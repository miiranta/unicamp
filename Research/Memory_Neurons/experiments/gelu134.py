"""
GELU134 — Pre-GELU z-score gate (gate the GELU input, not its output).

gelu80 applies the gate AFTER GELU: out = gelu(x) * gate.
This variant applies the gate BEFORE GELU: out = gelu(x * gate).

Gating before the nonlinearity changes the GELU *operating point* — a large
gate shifts input into GELU's steeper / more saturated regime, making the
function effectively sharper or softer.  This is a qualitatively different
inductive bias from post-gating.

    gate = 1 + alpha * tanh(sigma * |z|)
    out  = gelu(x * gate)

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn


class GELU134(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
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
        if self._warmup:
            self._ema_mean.copy_(x.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return self._gelu(x)

        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z = (x - self._ema_mean) / ema_std          # (B, T, D)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * z.abs())

        # gate applied BEFORE the nonlinearity
        out = self._gelu(x * gate)

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
