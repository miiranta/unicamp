"""
gelu112 – Variance spike gate  (kurtosis-motivated)
─────────────────────────────────────────────────────────────────────────────
Standard z-score: how far is the MEAN of a token from expected?
This variant: how SPREAD is the token's activation (its own internal variance)?

    var_t  = mean_d(x_{t,d}²) – mean_d(x_{t,d})²    (variance of x across D)
    ema_var = EMA(batch_mean(var_t))                  (typical variance level)
    ratio   = var_t / (ema_var + ε)                   (1 = average; >1 = spread)
    surp    = tanh(σ × log(ratio))                    (log-ratio for scale-inv.)
    gate    = exp(–τ × cos_out) × (1 + w × surp)
    result  = GELU(x) × gate

High-variance tokens = high-amplitude activations across channels = energetic/novel.
Low-variance tokens = quiet, focused activations = routine processing.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU112(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_var', torch.tensor(0.5))   # running mean variance
        self.register_buffer('_ema_out', torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── per-token variance gate (keeps grad through sigma path) ────────
        x_mean = x.mean(-1, keepdim=True)
        x_var  = ((x - x_mean) ** 2).mean(-1, keepdim=True)      # (B,T,1)

        ema_var = self._ema_var.detach().clamp(min=1e-6)
        ratio   = x_var / ema_var                                  # (B,T,1)
        surp    = torch.tanh(sigma * torch.log(ratio.clamp(min=1e-6)))

        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        cos_out   = (F.normalize(out, dim=-1) * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        var_batch = x_var.detach().mean().item()
        out_flat  = out.detach().reshape(-1, D)
        ob        = out_flat.mean(0)
        decay     = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_var.fill_(var_batch)
                self._ema_out.copy_(ob)
                self._initialised = True
            else:
                self._ema_var.mul_(decay).add_((1 - decay) * var_batch)
                self._ema_out.mul_(decay).add_((1 - decay) * ob)
        return result
