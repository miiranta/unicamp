"""
GELU141 — Random-projection z-score (cross-channel surprise via cheap projections).

The per-channel z-score (gelu80) treats channels INDEPENDENTLY. It misses
cross-channel correlations: a token could deviate from all single-channel means
but still land on the familiar cross-channel manifold.

Random projections (Johnson-Lindenstrauss) cheaply capture cross-channel
structure: R=32 random unit vectors tile the hypersphere.  Track EMA mean
and variance in each projected direction.  A token that is novel in PROJECTION
SPACE is novel in cross-channel structure — something per-channel z-scores miss.

    M       = fixed random (D, R) unit-column matrix
    x_proj  = x @ M                              (B, T, R)
    z_proj  = (x_proj - ema_m) / ema_s           (B, T, R)
    surp    = tanh(sigma * mean_r(|z_proj_r|))   (B, T, 1)
    gate    = 1 + alpha * surp
    out     = gelu(x) * gate

M is registered as a buffer (non-trainable), saving O(D*R) from parameters.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _M (D, R) fixed,  _ema_m (R,), _ema_sq (R,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU141(nn.Module):
    R = 32   # number of random projections

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        # fixed orthogonal-ish random projection matrix  (D, R)
        M = torch.randn(d_ff, self.R)
        M = F.normalize(M, dim=0)   # unit columns
        self.register_buffer("_M",       M)
        self.register_buffer("_ema_m",   torch.zeros(self.R))
        self.register_buffer("_ema_sq",  torch.ones(self.R))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_m.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)
        x_proj = x @ self._M    # (B, T, R)

        if self._warmup:
            self._ema_m.copy_(x_proj.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x_proj.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return base

        ema_var = (self._ema_sq - self._ema_m ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z_proj  = (x_proj - self._ema_m) / ema_std    # (B, T, R)
        surp    = z_proj.abs().mean(dim=-1, keepdim=True)   # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)
        out   = base * gate

        with torch.no_grad():
            d  = self._decay
            bm = x_proj.detach().mean(dim=(0, 1))
            bs = (x_proj.detach() ** 2).mean(dim=(0, 1))
            self._ema_m.mul_(d).add_(bm * (1-d))
            self._ema_sq.mul_(d).add_(bs * (1-d))

        return out
