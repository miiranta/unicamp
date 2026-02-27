"""
gelu101 – Robust MAD z-score gate  (mean absolute deviation instead of std)
─────────────────────────────────────────────────────────────────────────────
gelu80 estimates spread via std = sqrt(E[x²] – μ²).  Variance is dominated by
outliers.  This variant tracks the running Mean Absolute Deviation (MAD):

    ema_mad_d  ←  EMA( |x_d – ema_mean_d| )      (batch mean of deviations)
    z_d        =  (x_d – ema_mean_d) / (ema_mad_d + eps)
    surp       =  tanh(σ × mean_d(|z_d|))
    gate       =  exp(–τ × cos_out) × (1 + w × surp)
    result     =  GELU(x) × gate

MAD is ~0.8 std for Gaussians so numerically comparable, but gives a much more
robust estimate in heavy-tailed activation distributions common in transformers.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU101(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_mad',  torch.ones(d_model) * 0.5)  # MAD init
        self.register_buffer('_ema_out',  torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── robust z-score using EMA MAD ───────────────────────────────────
        ema_mean = self._ema_mean.detach()          # (D,)
        ema_mad  = self._ema_mad.detach().clamp(min=1e-6)   # (D,)

        z        = (x - ema_mean) / ema_mad         # (B, T, D)  – keeps grad
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))  # (B,T,1)

        # ── cosine output gate ──────────────────────────────────────────────
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── EMA update ────────────────────────────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        x_bm     = x_flat.mean(0)
        x_bmad   = (x_flat - self._ema_mean.unsqueeze(0)).abs().mean(0)
        out_bm   = out_flat.mean(0)

        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(x_bm)
                self._ema_mad.copy_(x_bmad.clamp(min=1e-6))
                self._ema_out.copy_(out_bm)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1.0 - decay) * x_bm)
                self._ema_mad.mul_(decay).add_((1.0 - decay) * x_bmad)
                self._ema_out.mul_(decay).add_((1.0 - decay) * out_bm)

        return result
