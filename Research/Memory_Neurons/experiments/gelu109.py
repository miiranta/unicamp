"""
gelu109 – Gradient-of-EMA surprise  (second-order temporal novelty)
─────────────────────────────────────────────────────────────────────────────
First-order novelty: how far is x from the EMA mean?   (gelu80)
Second-order novelty: how fast is the EMA mean CHANGING?

    drift_d     = |ema_mean_d – ema_mean_prev_d|     (abs change in mean this step)
    ema_drift_d = EMA(drift_d)                         (typical rate of change)
    burst_d     = drift_d / (ema_drift_d + ε)          (drift ratio)
    surp        = tanh(σ × mean_d(burst_d – 1))       (excess drift)
    gate        = exp(–τ × cos_out) × (1 + w × relu(surp))
    result      = GELU(x) × gate

Hypothesis: when the EMA is changing FAST, the model is actively learning a
new distribution — those tokens deserve amplification because they're updating
the world model. When drift is low (stable regime), suppress familiar patterns.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU109(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_mean',      torch.zeros(d_model))
        self.register_buffer('_ema_mean_prev', torch.zeros(d_model))
        self.register_buffer('_ema_drift',     torch.full((d_model,), 0.01))
        self.register_buffer('_ema_sq',        torch.ones(d_model))
        self.register_buffer('_ema_out',       torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── second-order drift surprise ─────────────────────────────────────
        ema_mean = self._ema_mean.detach()
        ema_prev = self._ema_mean_prev.detach()
        ema_drft = self._ema_drift.detach().clamp(min=1e-6)   # (D,)

        drift_d  = (ema_mean - ema_prev).abs()                # (D,)
        burst_d  = drift_d / ema_drft                         # (D,) ratio
        # surp: scalar from D-mean of (burst-1)
        excess   = (burst_d - 1.0).mean()                     # scalar
        # Make differentiable via sigma: σ scales the feature, gradient flows
        # We recompute from x to retain grad path for sigma/w
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std
        surp_base = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))  # normal surp
        # Modulate by drift burst: when burst excess > 0, amplify surp
        burst_mod = torch.tensor(excess, device=x.device).clamp(min=0.0).detach()
        surp      = surp_base * (1.0 + burst_mod)

        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        cos_out   = (F.normalize(out, dim=-1) * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        xb, xsq, ob = x_flat.mean(0), x_flat.pow(2).mean(0), out_flat.mean(0)
        decay    = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb); self._ema_mean_prev.copy_(xb)
                self._ema_sq.copy_(xsq); self._ema_out.copy_(ob)
                self._initialised = True
            else:
                new_mean = self._ema_mean * decay + (1 - decay) * xb
                drift    = (new_mean - self._ema_mean).abs()
                self._ema_drift.mul_(decay).add_((1 - decay) * drift)
                self._ema_mean_prev.copy_(self._ema_mean)
                self._ema_mean.copy_(new_mean)
                self._ema_sq.mul_(decay).add_((1 - decay) * xsq)
                self._ema_out.mul_(decay).add_((1 - decay) * ob)
        return result
