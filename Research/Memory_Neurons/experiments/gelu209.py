"""GELU209 – Memory-Normalized GELU (No Gating).

THE MOST MINIMAL MEMORY HYPOTHESIS.

All previous experiments add a multiplicative gate on top of GELU(x).
This experiment asks: what if we just NORMALIZE the GELU input per-channel
using the EMA statistics, with NO gating at all?

    x_norm = (x − ema_mean) / ema_std     (per-channel z-score)
    out    = GELU(x_norm) × scale

    where scale = exp(log_scale) is a learnable output scalar.

INTUITION: the EMA-normalization centres each channel around its historical mean,
making the GELU nonlinearity more responsive to DEVIATIONS (surprises) relative
to the channel's typical activation level. The channel is now always operating in
the "interesting" regime of GELU (near 0), regardless of its DC offset.

This is like a channel-wise adaptive batch norm that runs online via EMA.

COMPARISON TARGET: pure GELU (control, ppl=172.34), gelu189 (best, ppl=161.11)
If gelu209 achieves comparable gains with only 2 parameters, normalization is the key.

PARAMS: logit_decay, log_scale (2 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU209(nn.Module):
    """Memory-normalized GELU: GELU((x − ema_mean) / ema_std) × scale."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # log(0.9/0.1) ≈ 2.197
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # scale = exp(0) = 1.0 at init → initially a no-op scale
        self.log_scale   = nn.Parameter(torch.zeros(1))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        scale = self.log_scale.exp()                             # scalar ≥ 0

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return self._gelu(x)

        with torch.no_grad():
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)                       # (1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)                  # (1, 1, D)

        # Normalise GELU input using historical per-channel statistics
        x_norm = (x - mu_) / (std + self.eps)                   # (B, T, D)
        output = self._gelu(x_norm) * scale                     # (B, T, D)

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)

        return output
