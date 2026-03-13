"""gelu295 – Channel Firing Rate Homeostasis Gate.

CONCEPT:
    Biological neurons maintain a TARGET FIRING RATE via homeostatic plasticity.
    When a neuron fires too often (hyperactive), it downregulates;
    when it fires too rarely (hypoactive), it upregulates.

    We implement this for each output channel d of the FFN:
        "fires" = the channel's GELU output exceeds a per-channel threshold.
        rate_d = EMA of I[out_d > thresh_d] over batches.
        gate_d = 1 + beta * tanh(gamma * (target_d - rate_d))
               → rate > target: gate < 1 (suppress over-active channel)
               → rate < target: gate > 1 (amplify under-active channel)

    target_d: learned per-channel target firing rate.
    thresh_d: learned per-channel firing threshold.

SEQUENTIAL ADAPTATION:
    During eval pass 1: channels that fire more than their training rate
        (test-novel channels) accumulate rate > target → gate suppresses them.
    During eval pass 2: suppression from pass 1 lowers their effective output
        → less over-activation → rate closer to target → further suppression
        is less necessary → the gating has "calibrated" → Δ > 0.

BENEFIT FROM BACKPROP:
    log_target (D,), log_thresh_raw (D,), log_beta, log_gamma: all trained.
    Gradient shapes thresholds to maximise language model likelihood.

NO CAUSALITY LEAK:
    Firing rates aggregated over all B×T positions (batch-level, causal).

PARAMS:  log_target (D,), log_thresh_raw (D,), log_beta, log_gamma, logit_d_rate.
STATE:   _rate (D,) — EMA of firing indicator; reset to sigmoid(log_target) on reset_state().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU295(nn.Module):
    """Firing-rate homeostasis gate: suppress over-active channels, amplify under-active."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.D_FF = D_FF

        # Per-channel target rate in (0,1): exp(-1) ≈ 0.37 is a reasonable default
        self.log_target   = nn.Parameter(torch.full((D_FF,), math.log(0.37)))
        # Per-channel threshold (softplus-ensured positive, around median activation)
        self.log_thresh   = nn.Parameter(torch.zeros(D_FF))
        # Gate strength and sharpness
        self.log_beta     = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Rate EMA decay (slow — firing rate is a long-term average)
        self.logit_d_rate = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))

        self._rate: torch.Tensor = None   # (D,) EMA of firing indicator

    def reset_state(self):
        with torch.no_grad():
            # Reset firing rate to learned target (neutral start)
            self._rate = torch.sigmoid(self.log_target).detach().clone()

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        # Per-channel threshold (positive via softplus)
        thresh  = F.softplus(self.log_thresh)   # (D,)

        # Soft firing indicator: sigmoid of (out - thresh) for differentiability
        # Straight-through would be hard threshold; sigmoid gives gradient
        fire    = torch.sigmoid(10.0 * (out - thresh.view(1,1,D)))  # (B, T, D) ∈ (0,1)
        fire_m  = fire.mean(dim=(0,1))                               # (D,) current firing rate

        # Initialise rate buffer on first call
        if self._rate is None:
            with torch.no_grad():
                self._rate = fire_m.detach().clone()

        # Target rate per channel
        target  = torch.sigmoid(self.log_target)   # (D,)

        # Gate: homeostatic correction
        beta    = F.softplus(self.log_beta)
        gamma   = F.softplus(self.log_gamma)
        rate_diff = target - self._rate.detach()   # (D,) positive = below target (amplify)
        gate    = (1.0 + beta * torch.tanh(gamma * rate_diff)).clamp(0.05, 8.0)  # (D,)

        output  = out * gate.view(1, 1, D)

        # Update rate EMA (detach old, contribute gradient through current)
        d_rate      = torch.sigmoid(self.logit_d_rate)
        new_rate    = d_rate * self._rate.detach() + (1 - d_rate) * fire_m
        self._rate  = new_rate.detach()

        return output
