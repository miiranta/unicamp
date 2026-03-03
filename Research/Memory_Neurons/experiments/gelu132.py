"""
GELU132 — Per-channel alpha (D-dimensional amplification).

Like gelu80 but the amplification magnitude alpha is per-channel (D-dim)
rather than a global scalar.  Each channel independently learns how strongly
it should be up-weighted when novel.

    gate = 1 + alpha_d * tanh(sigma * |z_d|)    (element-wise)
    out  = gelu(x) * gate

Params: log_alpha_raw (D,),  log_sigma (scalar)  [D + 1 parameters]
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU132(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        # per-channel amplitude
        self.log_alpha   = nn.Parameter(torch.zeros(d_ff))
        # global sensitivity
        self.log_sigma   = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean", torch.zeros(d_ff))
        self.register_buffer("_ema_sq",   torch.ones(d_ff))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            self._ema_mean.copy_(x.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return base

        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z = (x - self._ema_mean) / ema_std          # (B, T, D)

        alpha = torch.exp(self.log_alpha)            # (D,) per-channel
        sigma = torch.exp(self.log_sigma)            # scalar

        gate = 1.0 + alpha * torch.tanh(sigma * z.abs())
        out  = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
