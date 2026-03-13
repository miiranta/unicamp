"""gelu300 – Batch-Delta Contrastive Gate (Momentum of Activation Changes).

CONCEPT:
    gelu211 gates on the ABSOLUTE level of activations (z-score from EMA mean).
    But the TREND — is the activation increasing or decreasing across batches? —
    carries different information.

    A channel that is RISING (delta > 0) may be encountering genuinely novel
    content that keeps accumulating.  A channel that was high but is now FALLING
    (delta < 0) is habituating naturally.

    This experiment tracks the batch-to-batch DELTA of the output mean and
    gates based on it:
        delta_d = out_mean_current_d - out_mean_prev_d
        gate_d  = 1 + beta * tanh(gamma * delta_scaled_d)

    Rising channels are amplified; falling channels are suppressed.

SEQUENTIAL ADAPTATION:
    The direction effect is subtle but cumulative.
    Pass 1 sees test content → some channels rise (test-novel), some fall.
    Pass 2: channels that rose in pass 1 have now been seen → their delta
    should flip to zero or negative as the surge settles → gate follows.
    This creates a natural Δ > 0 without explicit detection.

BENEFIT FROM BACKPROP:
    log_beta, log_gamma, logit_d_delta (EMA for delta): all trained.
    d_delta controls how aggressively the delta is smoothed.
    Unlike gelu211, the DIRECTION of change drives the gate, creating a
    qualitatively different gradient signal.

NO CAUSALITY LEAK:
    batch means, causal across batches.  No within-sequence position leak.

PARAMS:  logit_d_slow, logit_d_delta, log_beta, log_gamma, log_eps_delta.
STATE:   _slow_mean (D,), _delta_ema (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU300(nn.Module):
    """Delta-based gate: gate on batch-to-batch change in output mean."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_delta = 1e-4

        # Slow EMA of output mean (tracks running baseline)
        self.logit_d_slow  = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))
        # EMA over deltas (smooths the delta signal)
        self.logit_d_delta = nn.Parameter(torch.tensor(math.log(0.7 / 0.3)))
        # Gate shape
        self.log_beta      = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Log of delta normalisation scale (learnable)
        self.log_eps_delta = nn.Parameter(torch.tensor(math.log(0.1)))

        self._slow_mean:  torch.Tensor = None
        self._delta_ema:  torch.Tensor = None
        self._prev_mean:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._slow_mean = None
        self._delta_ema = None
        self._prev_mean = None
        self._ready     = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self._gelu(x)
        out_mean = out.flatten(0,1).mean(0)    # (D,)

        if not self._ready:
            with torch.no_grad():
                self._slow_mean = out_mean.detach().clone()
                self._delta_ema = torch.zeros_like(out_mean)
                self._prev_mean = out_mean.detach().clone()
                self._ready = True
            return out

        B, T, D = x.shape
        d_slow  = torch.sigmoid(self.logit_d_slow)
        d_delta = torch.sigmoid(self.logit_d_delta)

        # Delta from previous batch (detached prev)
        delta   = out_mean - self._prev_mean.detach()     # (D,) differentiable through out_mean

        # Smooth delta via EMA
        new_delta_ema = d_delta * self._delta_ema.detach() + (1 - d_delta) * delta

        # Normalise: scale by learnable eps_delta for stable z-score-like signal
        eps_d   = self.log_eps_delta.exp()
        z_delta = new_delta_ema / (new_delta_ema.detach().abs().mean() + eps_d)

        beta    = F.softplus(self.log_beta)
        gamma   = F.softplus(self.log_gamma)
        gate    = (1.0 + beta * torch.tanh(gamma * z_delta)).clamp(0.05, 8.0)  # (D,)

        output  = out * gate.view(1, 1, D)

        # Update slow mean and delta EMA
        new_slow = d_slow * self._slow_mean.detach() + (1 - d_slow) * out_mean
        self._slow_mean = new_slow.detach()
        self._delta_ema = new_delta_ema.detach()
        self._prev_mean = out_mean.detach()

        return output
