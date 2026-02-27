"""
gelu117 – Prediction-error gate  (online linear predictor)
─────────────────────────────────────────────────────────────────────────────
Predictive coding: neurons should respond most to UNPREDICTED input.
This variant maintains a tiny ONLINE LINEAR PREDICTOR: given the running mean
activation, it predicts x via the EMA mean, then gates by prediction error:

    pred_t      = ema_mean                         (simplest linear prediction)
    error_t     = x_t – pred_t                     (residual)
    error_rms   = rms(error_t)                     (scalar per token)
    expected_rms = EMA(batch error_rms)            (running baseline)
    surp_t      = tanh(σ × (error_rms/expected_rms – 1))
    gate_t      = 1 + w × surp_t
    result      = GELU(x) × gate

The twist vs gelu80 (which also uses z-score): no per-channel std normalization.
The gate is purely on TOTAL ENERGY of the prediction error, not normalized
per-channel. This makes it sensitive to absolute magnitude changes.
Parameters: logit_decay, log_sigma_raw, log_w_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU117(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_mean',     torch.zeros(d_model))
        self.register_buffer('_ema_err_rms',  torch.tensor(0.5))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── prediction error gating ────────────────────────────────────────
        pred        = self._ema_mean.detach()                         # (D,)
        error       = x - pred                                        # (B,T,D)
        err_rms     = error.pow(2).mean(-1, keepdim=True).sqrt()      # (B,T,1)
        ema_err_rms = self._ema_err_rms.detach().clamp(min=1e-6)
        ratio       = err_rms / ema_err_rms
        surp        = torch.tanh(sigma * (ratio - 1.0))
        gate        = 1.0 + w * surp
        result      = out * gate

        # ── EMA update ─────────────────────────────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        xb       = x_flat.mean(0)
        err_flat = (x_flat - self._ema_mean.unsqueeze(0))
        err_rms_batch = err_flat.pow(2).mean().sqrt().item()
        decay    = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb)
                self._ema_err_rms.fill_(err_rms_batch)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1 - decay) * xb)
                self._ema_err_rms.mul_(decay).add_((1 - decay) * err_rms_batch)
        return result
