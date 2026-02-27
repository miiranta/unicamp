"""
gelu97 – Channel-attention z-score gate
─────────────────────────────────────────────────────────────────────────────
Innovation over gelu80: instead of a plain mean of |z_d|, use a LEARNED
per-channel softmax weight to aggregate channel-level z-scores.

  ch_weight_d = softmax(β_d),  β ∈ ℝ^D   (learnable, with gradient)
  weighted_abs_z = sum_d(ch_weight_d × |z_d|) × D    (preserves scale)
  surp = tanh(σ × weighted_abs_z)
  gate = exp(-τ × cos_out) × (1 + w × surp)
  result = GELU(x) × gate

Hypothesis: not all channels are equally informative for novelty detection;
learning channel importance lets the model focus on the most diagnostic features.
Parameters: log_beta_raw (D,), log_tau, log_sigma_raw, log_w_raw  →  D+3 params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU97(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        # Per-channel weight for z-score aggregation (has gradient via gate path)
        self.log_beta_raw = nn.Parameter(torch.zeros(d_model))   # (D,)
        # Scalar gate parameters
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        # Fixed-decay EMA state — no per-channel decay needed
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))     # sigmoid → 0.5 init
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self.register_buffer('_ema_out',  torch.zeros(d_model))
        self._initialised  = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── channel-attention weights (learnable, maintains gradient) ──────
        # softmax over D dims, scaled so sum = D (neutral baseline = 1.0 each)
        ch_weight = F.softmax(self.log_beta_raw, dim=-1) * D   # (D,)

        # ── z-score on input using detached EMA stats ──────────────────────
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()   # (D,)

        z          = (x - ema_mean) / std             # (B, T, D) — with grad
        w_abs_z    = (z.abs() * ch_weight)            # (B, T, D)
        mean_wz    = w_abs_z.mean(dim=-1, keepdim=True)   # (B, T, 1)
        surp       = torch.tanh(sigma * mean_wz)      # (B, T, 1)

        # ── cosine similarity of GELU output to EMA output unit vector ─────
        ema_out_u  = self._ema_out.detach()
        ema_out_u  = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm   = F.normalize(out, dim=-1)
        cos_out    = (out_norm * ema_out_u).sum(dim=-1, keepdim=True)   # (B,T,1)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── EMA state update (inside no_grad) ─────────────────────────────
        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        x_bm     = x_flat.mean(0)
        x_bsq    = x_flat.pow(2).mean(0)
        out_bm   = out_flat.mean(0)

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
