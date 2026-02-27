"""
gelu98 – Post-activation z-score gate (familiarity in activation space)
─────────────────────────────────────────────────────────────────────────────
Innovation over gelu80: track EMA statistics on GELU(x) rather than x.
Hypothesis: familiarity is better measured in the nonlinear activation space
where the model actually operates, not in the pre-nonlinearity (linear) space.

  out   = GELU(x)
  z_d   = (out_d – ema_gelu_d) / ema_gelu_std_d   (per-channel z-score on activation)
  surp  = tanh(σ × mean_d(|z_d|))
  gate  = exp(–τ × cos_out) × (1 + w × surp)
  result = out × gate

The EMA here tracks the running statistics of the GELU activations, so the
gate learns to suppress tokens whose activation PATTERNS (not just inputs)
are familiar, and amplify tokens with unusual activation profiles.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU98(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        # EMA statistics on GELU(x) activation space
        self.register_buffer('_ema_gelu_mean', torch.zeros(d_model))
        self.register_buffer('_ema_gelu_sq',   torch.full((d_model,), 0.1))
        # EMA of output direction (for cosine gate)
        self.register_buffer('_ema_out',       torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)      # (B, T, D) — keep gradient

        # ── z-score of GELU output against EMA activation stats ────────────
        ema_gm  = self._ema_gelu_mean.detach()     # (D,)
        ema_gsq = self._ema_gelu_sq.detach()       # (D,)
        std_g   = (ema_gsq - ema_gm.pow(2)).clamp(min=1e-6).sqrt()

        z_out         = (out - ema_gm) / std_g     # (B, T, D) with grad
        mean_abs_z    = z_out.abs().mean(dim=-1, keepdim=True)   # (B, T, 1)
        surp          = torch.tanh(sigma * mean_abs_z)           # (B, T, 1)

        # ── cosine output gate ──────────────────────────────────────────────
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(dim=-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── EMA update on GELU activation statistics ───────────────────────
        out_flat = out.detach().reshape(-1, D)
        out_bm   = out_flat.mean(0)
        out_bsq  = out_flat.pow(2).mean(0)

        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_gelu_mean.copy_(out_bm)
                self._ema_gelu_sq.copy_(out_bsq)
                self._ema_out.copy_(out_bm)
                self._initialised = True
            else:
                self._ema_gelu_mean.mul_(decay).add_((1.0 - decay) * out_bm)
                self._ema_gelu_sq.mul_(decay).add_((1.0 - decay) * out_bsq)
                self._ema_out.mul_(decay).add_((1.0 - decay) * out_bm)

        return result
