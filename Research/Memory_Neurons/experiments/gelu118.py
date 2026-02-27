"""
gelu118 – Learned GELU shift gate  (bias the nonlinearity)
─────────────────────────────────────────────────────────────────────────────
An entirely different approach: instead of GATING the output, SHIFT the
input to GELU by a learned surprise-dependent bias:

    bias_t  = α × tanh(σ × mean|z_d|)          ∈ [–α, +α]   (α = softplus(log_α))
    result  = GELU(x + bias_t)

When a token is novel (large surprise), the bias shifts x rightward →
more of x falls in the GELU saturation zone → near-1 output for larger inputs.
When familiar, bias ≈ 0 → vanilla GELU.

This is a non-multiplicative, additive gate: it changes the OPERATING POINT
of the nonlinearity rather than scaling the output. Qualitatively different
from all prior experiments.
Parameters: logit_decay, log_alpha_raw, log_sigma_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU118(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))   # max shift amplitude
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))   # surprise scale
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        alpha  = F.softplus(self.log_alpha_raw) + 0.01
        sigma  = F.softplus(self.log_sigma_raw) + 0.01

        # ── surprise from z-score ──────────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std                     # (B,T,D) with grad
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))  # (B,T,1)

        # ── additive shift of GELU input ───────────────────────────────────
        bias   = alpha * surp                               # (B,T,1) broadcast
        result = self._gelu(x + bias)                      # shifted nonlinearity

        # ── EMA update ─────────────────────────────────────────────────────
        x_flat = x.detach().reshape(-1, D)
        xb, xsq = x_flat.mean(0), x_flat.pow(2).mean(0)
        decay    = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb)
                self._ema_sq.copy_(xsq)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1 - decay) * xb)
                self._ema_sq.mul_(decay).add_((1 - decay) * xsq)
        return result
