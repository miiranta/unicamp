"""
gelu103 – Top-K channel surprise gate
─────────────────────────────────────────────────────────────────────────────
gelu80 aggregates novelty as the MEAN of |z_d| over all D channels.  This
averages out the signal: a single extremely novel channel gets diluted by
D-1 familiar ones.

This variant uses the TOP-K largest |z_d| values (K = D // 8 = 16 for D=128):
    surp = tanh(σ × mean(topk(|z_d|, K)))
    gate = exp(–τ × cos_out) × (1 + w × surp)

The top-K agg is differentiable through σ and w since topk indices are
selected from z using .detach() but the values through sigma scale.
Hypothesis: a token is novel if ANY subset of its channels fires unusually,
not only if the AVERAGE channel is unusual.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_K_FRAC = 8   # use top (D // K_FRAC) channels


class GELU103(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D  = d_model
        self.K  = max(1, d_model // _K_FRAC)   # =16 for D=128
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

        # ── top-K z-score aggregation ──────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()

        # z keeps gradient through x; topk indices from detached magnitudes
        z       = (x - ema_mean) / std               # (B, T, D) with grad
        abs_z   = z.abs()
        # topk on detached values to get indices, then select from z to keep grad
        _, idx  = abs_z.detach().topk(self.K, dim=-1)   # (B, T, K)
        z_topk  = z.gather(-1, idx)                      # (B, T, K) – has grad

        topk_mean = z_topk.abs().mean(-1, keepdim=True)   # (B, T, 1)
        surp      = torch.tanh(sigma * topk_mean)

        # ── cosine output gate ──────────────────────────────────────────────
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── EMA update ──────────────────────────────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        x_bm     = x_flat.mean(0)
        x_bsq    = x_flat.pow(2).mean(0)
        out_bm   = out_flat.mean(0)

        decay  = torch.sigmoid(self.logit_decay).detach().item()
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
