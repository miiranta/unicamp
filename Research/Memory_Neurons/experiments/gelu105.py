"""
gelu105 – Gate-then-GELU with hard clamp (novelty sharpening)
─────────────────────────────────────────────────────────────────────────────
Extends gelu90 (pre-GELU gate) with a HARD CLAMP on the input before GELU:
    clamped = clamp(x × gate, –c, +c)    c = softplus(log_c_raw) + 0.5
    result  = GELU(clamped)

The idea: familiar tokens (gate ≈ min_gate) get pushed toward zero → GELU
stays near-linear in [–c, c].  Novel tokens (gate > 1) get pushed toward the
saturation zone of GELU. The learned clamp c controls the operating point.
Hypothesis: controlling the nonlinearity operating point per-token is more
expressive than purely multiplicative gating.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw, log_c_raw  →  5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_MIN_GATE = 0.1


class GELU105(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.log_c_raw     = nn.Parameter(torch.tensor(0.0))   # learnable clamp radius
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self.register_buffer('_ema_out',  torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau    = torch.exp(self.log_tau)
        sigma  = F.softplus(self.log_sigma_raw) + 0.01
        w      = F.softplus(self.log_w_raw)
        c      = F.softplus(self.log_c_raw) + 0.5       # clamp radius ≥ 0.5

        # ── gate from EMA z-score + cosine ─────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std
        surp     = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))

        # Compute reference output for cosine gate
        out_ref  = self._gelu(x)
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out_ref, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(-1, keepdim=True)

        raw_gate = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        gate     = raw_gate.clamp(min=_MIN_GATE)

        # ── pre-GELU application with hard clamp ───────────────────────────
        x_gated  = x * gate
        x_clamped = x_gated.clamp(-c, c)
        result   = self._gelu(x_clamped)

        # ── EMA update ─────────────────────────────────────────────────────
        x_flat    = x.detach().reshape(-1, D)
        out_flat  = out_ref.detach().reshape(-1, D)
        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            xb, xsq, ob = x_flat.mean(0), x_flat.pow(2).mean(0), out_flat.mean(0)
            if not self._initialised:
                self._ema_mean.copy_(xb); self._ema_sq.copy_(xsq); self._ema_out.copy_(ob)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1.0 - decay) * xb)
                self._ema_sq.mul_(decay).add_((1.0 - decay) * xsq)
                self._ema_out.mul_(decay).add_((1.0 - decay) * ob)
        return result
