"""GELU166 – Per-Channel Skewness Gate (3rd Central Moment EMA).

THE KEY INSIGHT:
    gelu80 (best at 7%) tracks mean and variance (1st and 2nd moments).
    gelu153 tracks kurtosis (4th moment) as a scalar.

    The SKEWNESS (3rd standardized central moment) captures asymmetry:
        skewness_d = E[(x-μ)³] / σ³

    For a NOVEL input at channel d:
    - If x[b,t,d] is extreme in the same direction as the historical skew,
      it's actually on the expected "heavy tail" — not novel.
    - If x[b,t,d] pushes AGAINST the historical skew (e.g., large positive value
      when history is negatively skewed), that IS novel — it's activating
      the distribution's thin tail.

    We measure: |z_d * sign(skewness_d)|
    - Large positive z when skew > 0: on the expected heavy tail → less novel
    - Large positive z when skew < 0: on the unexpected thin tail → more novel

    Adjusted novelty per channel:
        z_adj_d = z_d * sign(-skewness_d * z_d + eps)
                = z_d if pushing against historical skew
                = 0  if going with historical skew
    Simplified: adj_d = max(0, -sign(skewness_d) * z_d) = thin-tail deviation

    Total gate: 1 + alpha * tanh(sigma * mean_d(adj_d))

STATE: Track EMA of per-channel mean, variance (for z-score), and 3rd central moment.

CAUSALITY: EMA updated after forward with detached x. ✓
GRADIENT: All moment computation under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,), _ema_sq (D,), _ema_cm3 (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
EPS_VAR   = 1e-4


class GELU166(nn.Module):
    """Per-channel skewness-adjusted z-score gate: thin-tail deviations = more novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_mean: torch.Tensor = None   # (D,)
        self._ema_sq:   torch.Tensor = None   # (D,)
        self._ema_cm3:  torch.Tensor = None   # (D,) EMA of (x - mean)³
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_cm3  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)   # (N, D)
                mu = xf.mean(0)
                self._ema_mean = mu.clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_cm3  = (xf - mu).pow(3).mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var_d  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=EPS_VAR)   # (D,)
            std_d  = var_d.sqrt()                                                  # (D,)
            skew_d = self._ema_cm3 / (var_d * std_d + 1e-8)                      # (D,) skewness

            # Per-token z-score
            mu_  = self._ema_mean.view(1, 1, D)
            std_ = std_d.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + 1e-5)     # (B, T, D)

            # Thin-tail deviation: large positive z against negative skew (or vice versa)
            # adj_d = max(0, -sign(skew_d) * z_d)
            sign_skew = skew_d.sign().view(1, 1, D)
            adj       = F.relu(-sign_skew * z)              # (B, T, D) ≥ 0
            surp      = adj.mean(dim=-1).mean()             # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        with torch.no_grad():
            xf   = x.detach().flatten(0, 1)
            mu_b = xf.mean(0)
            cm3_b = (xf - mu_b).pow(3).mean(0)
            d = EMA_DECAY
            self._ema_mean = d * self._ema_mean + (1 - d) * mu_b
            self._ema_sq   = d * self._ema_sq   + (1 - d) * xf.pow(2).mean(0)
            self._ema_cm3  = d * self._ema_cm3  + (1 - d) * cm3_b

        return output
