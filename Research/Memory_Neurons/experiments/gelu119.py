"""
gelu119 – Gated-residual modulation  (learn to be a residual shortcut)
─────────────────────────────────────────────────────────────────────────────
Rather than gating GELU(x), this module learns to BLEND between the identity
(residual shortcut) and GELU(x) based on how familiar x is:

    alpha_t = sigmoid(β × surp_t)              ∈ [0, 1]:  0=familiar, 1=novel
    result  = alpha_t × GELU(x) + (1 – alpha_t) × x

When familiar (low surprise → alpha≈0): result ≈ x  (identity / skip)
When novel   (high surprise → alpha≈1): result ≈ GELU(x)  (full nonlinearity)

Hypothesis: familiar context should pass through mostly unchanged (residual
stream preservation), while novel content should undergo full GELU transformation.
Parameters: logit_decay, log_sigma_raw, log_beta_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU119(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_beta_raw  = nn.Parameter(torch.tensor(0.0))   # blend sharpness
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        beta  = F.softplus(self.log_beta_raw)  + 0.1

        out = self._gelu(x)

        # ── z-score surprise ──────────────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std                              # (B,T,D)
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))   # (B,T,1)

        # ── blend gate: 0=identity, 1=GELU ────────────────────────────────
        alpha  = torch.sigmoid(beta * surp)                          # (B,T,1)
        result = alpha * out + (1.0 - alpha) * x

        # ── EMA update ─────────────────────────────────────────────────────
        x_flat = x.detach().reshape(-1, D)
        xb, xsq = x_flat.mean(0), x_flat.pow(2).mean(0)
        decay   = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb); self._ema_sq.copy_(xsq)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1 - decay) * xb)
                self._ema_sq.mul_(decay).add_((1 - decay) * xsq)
        return result
