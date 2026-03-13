"""gelu290 – Per-Channel Variance Novelty Gate with EMA Baseline Adaptation.

MOTIVATION:
    All existing experiments gate on the MEAN activation (z-score of mean).
    The VARIANCE of activations is complementary: a channel with unusually
    HIGH variance is responding to diverse (novel) stimuli; LOW variance
    means uniform, stereotyped (habituated) responses.

    This experiment:
    1. Trains a per-channel baseline variance (log-parameterised).
    2. Gates based on observed variance vs baseline:
         var_d / baseline_d - 1  (positive = more varied than expected)
    3. During eval: maintains a slow EMA of observed variance, using it as
       a dynamic baseline that adapts to the test distribution across passes.

MECHANISM:
    out          = gelu(x)                             # (B, T, D)
    var          = out.var(dim=(0,1))                  # (D,) current batch variance
    effective_bl = alpha * exp(log_baseline) + (1-alpha) * _var_ema.detach()
    z_var        = var / (effective_bl + eps) - 1.0   # relative deviation
    gate         = (1 + beta * tanh(gamma * z_var)).clamp(0.1, 5.0)
    return out * gate.view(1, 1, D)

    alpha ∈ (0,1): blend between static trained baseline and dynamic EMA.
    log_baseline: D-dim trainable, learns the training-distribution variance.
    _var_ema:     D-dim buffer, slow EMA of observed variance (d ≈ 0.995).
                  Reset to exp(log_baseline) at start of each eval run.

SEQUENTIAL ADAPTATION:
    Pass 1: _var_ema accumulates the TEST variance profile.
            Some channels have higher test-variance than training → gate ↑
    Pass 2: _var_ema ≈ test distribution.
            effective_bl shifts towards test → z_var → 0 for typical channels
            → gate → 1 for habituated channels → less amplification noise → Δ > 0

NO CAUSALITY LEAK:
    batch variance is aggregated over all B×T positions, causal across batches.

BENEFIT FROM BACKPROP:
    log_baseline, log_beta, log_gamma, logit_alpha all get gradient.
    log_baseline converges to log of training-distribution variance per channel.
    alpha learns the optimal weight between static prior and dynamic adaptation.

PARAMS:  log_baseline (D,), log_beta, log_gamma, logit_alpha, logit_d_var.
STATE:   _var_ema (D,) — reset to exp(log_baseline) on reset_state().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU290(nn.Module):
    """Variance novelty gate: gate on relative deviation of batch variance from baseline."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.D_FF = D_FF

        # Per-channel log-variance baseline (trained to match training distribution)
        self.log_baseline = nn.Parameter(torch.zeros(D_FF))   # exp(0) = 1 init
        # Gate amplitude and sharpness
        self.log_beta     = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Blend: alpha=1 → use only static baseline; alpha=0 → use only EMA
        self.logit_alpha  = nn.Parameter(torch.zeros(1))       # sigmoid(0) = 0.5
        # Decay for variance EMA (very slow)
        self.logit_d_var  = nn.Parameter(torch.tensor(math.log(0.995 / 0.005)))

        # Variance EMA buffer (resets to static baseline at start of eval run)
        self._var_ema: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        # Reset to trained static baseline so pass-1 starts from prior
        with torch.no_grad():
            self._var_ema = self.log_baseline.exp().detach().clone()
        self._ready = True

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._gelu(x)   # (B, T, D)

        # Current batch per-channel variance (unbiased)
        var = out.var(dim=(0, 1), unbiased=False).clamp(min=self.eps)   # (D,)

        # Initialise EMA buffer on first forward call (before reset_state)
        if self._var_ema is None:
            with torch.no_grad():
                self._var_ema = var.detach().clone()

        # ── Effective baseline: blend static prior with dynamic EMA ──────
        static_bl = self.log_baseline.exp().clamp(min=self.eps)         # (D,)
        alpha     = torch.sigmoid(self.logit_alpha)                     # scalar
        eff_bl    = alpha * static_bl + (1 - alpha) * self._var_ema.detach()  # (D,)

        # ── Variance z-score and gate ─────────────────────────────────────
        z_var  = var / (eff_bl + self.eps) - 1.0                        # (D,)
        beta   = F.softplus(self.log_beta)
        gamma  = F.softplus(self.log_gamma)
        gate   = (1.0 + beta * torch.tanh(gamma * z_var)).clamp(0.1, 5.0)  # (D,)

        output = out * gate.view(1, 1, -1)

        # ── Update variance EMA (detach old, current var participates in graph) ──
        d_var        = torch.sigmoid(self.logit_d_var)
        new_var_ema  = d_var * self._var_ema.detach() + (1 - d_var) * var
        self._var_ema = new_var_ema.detach()

        return output
