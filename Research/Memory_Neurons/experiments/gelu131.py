"""
GELU131 — Per-channel sigma (D-dimensional sensitivity).

Identical to gelu80 (per-channel z-score gate) but sigma is a D-dimensional
learnable vector instead of a global scalar.  Each channel can independently
set how sharply it responds to deviation from its running mean.

    gate = 1 + alpha * tanh(sigma_d * |z_d|)    (element-wise)
    out  = gelu(x) * gate

Params: log_alpha (scalar), log_sigma_raw (D,)  [2 + D parameters]
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU131(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        # scalar amplitude
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
        # per-channel sensitivity
        self.log_sigma   = nn.Parameter(torch.zeros(d_ff))   # sigma_d = exp(log_sigma_d)
        self._gelu = nn.GELU()

        # running EMA statistics (not parameters)
        self.register_buffer("_ema_mean", torch.zeros(d_ff))
        self.register_buffer("_ema_sq",   torch.ones(d_ff))
        self._decay = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        base = self._gelu(x)

        if self._warmup:
            batch_mean = x.detach().mean(dim=(0, 1))
            batch_sq   = (x.detach() ** 2).mean(dim=(0, 1))
            self._ema_mean.copy_(batch_mean)
            self._ema_sq.copy_(batch_sq)
            self._warmup = False
            return base

        # z-score using EMA statistics
        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z = (x - self._ema_mean) / ema_std          # (B, T, D)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)            # (D,) per-channel sensitivity

        gate = 1.0 + alpha * torch.tanh(sigma * z.abs())
        out  = base * gate

        # update EMA after forward
        with torch.no_grad():
            bm = x.detach().mean(dim=(0, 1))
            bq = (x.detach() ** 2).mean(dim=(0, 1))
            d  = self._decay
            self._ema_mean.mul_(d).add_(bm * (1 - d))
            self._ema_sq.mul_(d).add_(bq * (1 - d))

        return out
