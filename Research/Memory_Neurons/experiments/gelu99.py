"""
gelu99 – Signed asymmetric surprise gate
─────────────────────────────────────────────────────────────────────────────
Innovation over gelu80: the scalar mean of |z_d| treats POSITIVE surprise
(activation above expectation) identically to NEGATIVE surprise (activation
below expectation). This variant separates them:

  pos_surp = tanh(σ × mean_d(relu( z_d)))   # unusually HIGH activation
  neg_surp = tanh(σ × mean_d(relu(-z_d)))   # unusually LOW activation
  gate     = exp(–τ × cos_out) × (1 + w_up × pos_surp + w_dn × neg_surp)
  result   = GELU(x) × gate

Hypothesis: excitation above the expected level (novel stimulus) and
inhibition below it (missing-expected stimulus) may need different
amplification strengths.  w_up and w_dn are learned separately.
Parameters: logit_decay, log_tau, log_sigma_raw, log_wup_raw, log_wdn_raw → 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU99(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay    = nn.Parameter(torch.tensor(0.0))
        self.log_tau        = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw  = nn.Parameter(torch.tensor(0.0))
        self.log_wup_raw    = nn.Parameter(torch.tensor(0.0))   # positive-surprise coeff
        self.log_wdn_raw    = nn.Parameter(torch.tensor(0.0))   # negative-surprise coeff
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
        w_up  = F.softplus(self.log_wup_raw)
        w_dn  = F.softplus(self.log_wdn_raw)

        out = self._gelu(x)

        # ── signed z-score gate ─────────────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()

        z        = (x - ema_mean) / std                              # (B,T,D)
        pos_surp = torch.tanh(sigma * F.relu( z).mean(-1, keepdim=True))  # (B,T,1)
        neg_surp = torch.tanh(sigma * F.relu(-z).mean(-1, keepdim=True))  # (B,T,1)

        # ── cosine output gate ──────────────────────────────────────────────
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w_up * pos_surp + w_dn * neg_surp)
        result = out * gate

        # ── EMA update ────────────────────────────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        x_bm   = x_flat.mean(0)
        x_bsq  = x_flat.pow(2).mean(0)
        out_bm = out_flat.mean(0)

        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(x_bm)
                self._ema_sq.copy_(x_bsq)
                self._ema_out.copy_(out_bm)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1.0 - decay) * x_bm)
                self._ema_sq.mul_(decay).add_((1.0 - decay) * x_bsq)
                self._ema_out.mul_(decay).add_((1.0 - decay) * out_bm)

        return result
