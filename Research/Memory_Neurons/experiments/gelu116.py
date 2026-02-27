"""
gelu116 – Channel-wise asymmetric EMA  (different decay per channel direction)
─────────────────────────────────────────────────────────────────────────────
gelu80 uses the same EMA decay for ALL channels.  But some channels may be
"fast-adapting" (highly variable) and others "slow-adapting" (stable).

This variant learns a DIFFERENT decay rate per channel half:
    channels [0:D//2]    → faster decay: sigmoid(logit_fast)
    channels [D//2:D]    → slower decay: sigmoid(logit_slow)

Then the z-score gate is computed normally across all D channels.
The different timescales mean the model separately tracks fast-moving
and slow-moving patterns in the feature space.

Parameters: logit_fast, logit_slow, log_tau, log_sigma_raw, log_w_raw  →  5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU116(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D    = d_model
        self.D2   = d_model // 2
        self.logit_fast    = nn.Parameter(torch.tensor(2.0))   # sigmoid ≈ 0.88
        self.logit_slow    = nn.Parameter(torch.tensor(4.0))   # sigmoid ≈ 0.98
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

        # ── z-score using current EMA (detached) ───────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))

        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        cos_out   = (F.normalize(out, dim=-1) * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── channel-asymmetric EMA update ─────────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        xb, xsq  = x_flat.mean(0), x_flat.pow(2).mean(0)
        ob       = out_flat.mean(0)
        d_fast   = torch.sigmoid(self.logit_fast).detach().item()
        d_slow   = torch.sigmoid(self.logit_slow).detach().item()

        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb)
                self._ema_sq.copy_(xsq)
                self._ema_out.copy_(ob)
                self._initialised = True
            else:
                D2 = self.D2
                # fast half
                self._ema_mean[:D2].mul_(d_fast).add_((1 - d_fast) * xb[:D2])
                self._ema_sq[:D2].mul_(d_fast).add_((1 - d_fast) * xsq[:D2])
                # slow half
                self._ema_mean[D2:].mul_(d_slow).add_((1 - d_slow) * xb[D2:])
                self._ema_sq[D2:].mul_(d_slow).add_((1 - d_slow) * xsq[D2:])
                self._ema_out.mul_(d_slow).add_((1 - d_slow) * ob)
        return result
