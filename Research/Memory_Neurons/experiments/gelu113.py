"""
gelu113 – Residual-magnitude gate  (RMS energy gating)
─────────────────────────────────────────────────────────────────────────────
Simple but principled: track the running RMS amplitude of x; amplify tokens
whose RMS is unusually HIGH (energetic), suppress unusually LOW (quiet).

    rms_t    = sqrt(mean_d(x_{t,d}²))
    ema_rms  = EMA(batch_mean(rms_t))
    ratio    = rms_t / (ema_rms + ε)
    surp     = tanh(σ × (ratio – 1))             (0 at average, ±1 at extremes)
    gate     = 1 + w × surp
    result   = GELU(x) × gate

This is the simplest possible "energy novelty" gate — no per-channel stats,
just global amplitude relative to history. Baseline for more complex designs.
Parameters: logit_decay, log_sigma_raw, log_w_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU113(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_rms', torch.tensor(0.5))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        rms_t   = x.pow(2).mean(-1, keepdim=True).sqrt()         # (B,T,1)
        ema_rms = self._ema_rms.detach().clamp(min=1e-6)
        ratio   = rms_t / ema_rms
        surp    = torch.tanh(sigma * (ratio - 1.0))
        gate    = 1.0 + w * surp
        result  = out * gate

        rms_batch = rms_t.detach().mean().item()
        decay     = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_rms.fill_(rms_batch)
                self._initialised = True
            else:
                self._ema_rms.mul_(decay).add_((1 - decay) * rms_batch)
        return result
