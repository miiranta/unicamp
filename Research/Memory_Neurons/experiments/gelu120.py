"""
gelu120 – Softplus-family surprise gate  (differentiable GELU variant)
─────────────────────────────────────────────────────────────────────────────
All prior experiments apply a surprise gate to vanilla GELU.
This one replaces GELU with a PARAMETERIZED nonlinearity that is directly
shaped by surprise:

    sharpness_t = 1 + w × surp_t                   ∈ [1, 1+w]
    result_d    = x_d × sigmoid(sharpness_t × x_d)  ≡ Swish with learned β

This is the "Silu/Swish" family: x·σ(β·x). When β=1 → standard Swish≈GELU.
When β>1 (novel token) → sharper, more selective activation.
When β<1 → softer, more linear. (Since surp∈[0,1], β≥1 always — never softer.)

Hypothesis: novel tokens need sharper feature detection; familiar ones need
smoother interpolation. The activation function itself should adapt per-token.
Parameters: logit_decay, log_sigma_raw, log_w_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU120(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        # ── surprise ───────────────────────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))   # (B,T,1)

        # ── token-adaptive Swish: x × sigmoid(sharpness × x) ─────────────
        sharpness = 1.0 + w * surp                    # (B,T,1) ≥ 1
        result    = x * torch.sigmoid(sharpness * x)  # element-wise Swish(β)

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
