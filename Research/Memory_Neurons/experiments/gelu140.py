"""
GELU140 — Homeostatic gate regulation (meta-adaptation).

gelu80's gate can persistently drift high (when the model processes genuinely
novel text for many steps) or low (during repeated/repetitive text).  This
biases ALL tokens in those stretches.

The fix: track EMA of the gate value itself and normalize.  The gate is
rescaled so its running average stays near 1.0 — homeostasis of the gate.

    raw_gate     = 1 + alpha * tanh(sigma * mean_d(|z_d|))    (same as gelu80)
    ema_gate     = EMA(mean(raw_gate))    ← track expected gate magnitude
    actual_gate  = raw_gate / (ema_gate + eps)                (normalize to mean=1)

Effect:
    • In novel stretches: raw_gate > 1, ema_gate rises, actual_gate stays ≈ 1
    • After novelty: raw_gate drops, ema_gate > raw_gate, actual_gate < 1 (familiar → suppress)
    • On average: actual_gate ≈ 1 → conservative energy usage
    • Relative within-batch discrimination is preserved (novel tokens still get higher gates)

This produces stronger RELATIVE contrast between familiar and novel tokens
even when the absolute novelty level is changing over training.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,), _ema_gate (scalar)
"""

import torch
import torch.nn as nn


class GELU140(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean",  torch.zeros(d_ff))
        self.register_buffer("_ema_sq",    torch.ones(d_ff))
        self.register_buffer("_ema_gate",  torch.tensor(1.0))   # init: expected gate = 1
        self._decay      = 0.99
        self._gate_decay = 0.99
        self._warmup     = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._ema_gate.fill_(1.0)
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
        z_abs   = (x - self._ema_mean).abs() / ema_std       # (B, T, D)
        surp    = z_abs.mean(dim=-1, keepdim=True)            # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)

        raw_gate = 1.0 + alpha * torch.tanh(sigma * surp)    # (B, T, 1)
        # normalize by running average gate value
        actual_gate = raw_gate / (self._ema_gate + 1e-6)
        out = base * actual_gate

        with torch.no_grad():
            d  = self._decay
            gd = self._gate_decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))
            # update gate EMA with current batch-mean gate
            batch_mean_gate = raw_gate.detach().mean()
            self._ema_gate.mul_(gd).add_(batch_mean_gate * (1-gd))

        return out
