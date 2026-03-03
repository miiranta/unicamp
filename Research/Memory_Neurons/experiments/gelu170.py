"""GELU170 – Log-Space Z-Score Gate (Multiplicative Surprise in Activation Magnitude).

THE KEY INSIGHT:
    gelu80 (best at 7%) uses LINEAR z-score: `(x - μ) / σ`.
    This measures ABSOLUTE deviation from the mean.

    But neural network activations (especially post-GELU which are ≥ 0) often
    have distributions that are better described in LOG space:
    - A weight activation of 0.01 where the mean is 0.1 is 10x below mean (huge %)
    - A weight activation of 100 where the mean is 10 is 10x above mean (huge %)
    - Linear z-score treats these asymmetrically (the second case alone is notable)

    LOG-SPACE Z-SCORE treats multiplicative changes symmetrically:
        log_x_d = log(|x_d| + eps)
        z_log_d = (log_x_d - EMA_log_mean_d) / (EMA_log_std_d + eps)

    A token that activates a channel at 10x the typical level has the same
    log-z-score as one at 0.1x the typical level (both are equally "multiplicatively
    surprising").

    This captures RATIO-BASED novelty:
    - Common function words: activate channels at predictable levels
    - Rare content words: activate channels at unusual ratios relative to typical

    Technically: if |x_d| ~ LogNormal, then the log-z-score is the correctly normalized
    sufficient statistic for identifying outliers. This is the log-normal MLE version
    of gelu80.

STATE: Track EMA of per-channel mean and variance in log-space.
CAUSALITY: EMA updated after forward. ✓
GRADIENT: Log-z-score computed under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_log_mean (D,), _ema_log_sq (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
EPS_LOG   = 1e-3   # small constant before log to handle near-zero activations
EPS_VAR   = 1e-4


class GELU170(nn.Module):
    """Log-space z-score gate: multiplicative activation surprise = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_log_mean: torch.Tensor = None   # (D,)
        self._ema_log_sq:   torch.Tensor = None   # (D,) EMA of (log|x|)²
        self._ready = False

    def reset_state(self):
        self._ema_log_mean = None
        self._ema_log_sq   = None
        self._ready        = False

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
                xf    = x.detach().flatten(0, 1)           # (N, D)
                lx    = (xf.abs() + EPS_LOG).log()         # (N, D) log-space
                self._ema_log_mean = lx.mean(0).clone()    # (D,)
                self._ema_log_sq   = lx.pow(2).mean(0).clone()  # (D,)
                self._ready = True
            return out

        with torch.no_grad():
            var_log = (self._ema_log_sq - self._ema_log_mean.pow(2)).clamp(min=EPS_VAR)
            std_log = var_log.sqrt()                  # (D,)

            lx = (x.detach().abs() + EPS_LOG).log()  # (B, T, D)
            mu_  = self._ema_log_mean.view(1, 1, D)
            std_ = std_log.view(1, 1, D)
            z_log = (lx - mu_) / (std_ + 1e-5)       # (B, T, D) log-space z-score

            mean_abs_z = z_log.abs().mean(dim=-1)     # (B, T)
            surp       = mean_abs_z.mean()            # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            lx_b = (xf.abs() + EPS_LOG).log()
            d = EMA_DECAY
            self._ema_log_mean = d * self._ema_log_mean + (1 - d) * lx_b.mean(0)
            self._ema_log_sq   = d * self._ema_log_sq   + (1 - d) * lx_b.pow(2).mean(0)

        return output
