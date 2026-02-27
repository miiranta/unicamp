"""
gelu106 – Layerwise Exponential Surprise (LES gate)
─────────────────────────────────────────────────────────────────────────────
Instead of tanh squashing the mean |z|, use an EXPONENTIAL:
    surp = exp(σ × mean_d(|z_d|)) – 1      (≈ σ·mean|z| for small z, blows up for large)
    surp_clipped = surp / (surp + 1)        (sigmoid of log(surp+1) — softclip to [0,1])
    gate = exp(–τ × cos_out) × (1 + w × surp_clipped)
    result = GELU(x) × gate

Hypothesis: tanh saturates too quickly for extreme novelty. An exponential
puts much more weight on rare, very surprising tokens — the opposite of tanh.
The softclip keeps it stable. This makes the gate highly non-linear in surprise.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU106(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
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

        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std                          # (B,T,D)
        mean_abs_z = z.abs().mean(-1, keepdim=True)              # (B,T,1)

        # exponential surprise, softclipped to [0,1]
        surp_raw     = torch.exp(sigma * mean_abs_z) - 1.0
        surp_clipped = surp_raw / (surp_raw + 1.0)              # ∈ [0,1)

        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        cos_out   = (F.normalize(out, dim=-1) * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp_clipped)
        result = out * gate

        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        decay    = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            xb, xsq, ob = x_flat.mean(0), x_flat.pow(2).mean(0), out_flat.mean(0)
            if not self._initialised:
                self._ema_mean.copy_(xb); self._ema_sq.copy_(xsq); self._ema_out.copy_(ob)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1 - decay) * xb)
                self._ema_sq.mul_(decay).add_((1 - decay) * xsq)
                self._ema_out.mul_(decay).add_((1 - decay) * ob)
        return result
