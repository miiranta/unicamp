"""
gelu102 – Position-aware familiarity gate
─────────────────────────────────────────────────────────────────────────────
Observation: the first token in a sequence, the last token, and middle tokens
have very different statistical profiles — yet gelu80 tracks a SINGLE global
EMA across all positions.

This variant maintains a per-position EMA (SEQ_LEN=64 independent buffers):
    ema_mean_t,  ema_sq_t     ∈ ℝ^D   for each position t ∈ {0…T-1}

    z_t,d = (x_{t,d} − ema_mean_t,d) / std_t,d
    surp_t = tanh(σ × mean_d(|z_t,d|))
    gate_t = exp(−τ × cos_out_t) × (1 + w × surp_t)
    result = GELU(x) × gate   (elementwise over time)

Hypothesis: positional context dramatically changes what "familiar" means,
and position-specific familiarity is a more informative novelty signal.
Parameters: logit_decay, log_tau, log_sigma_raw, log_w_raw  →  4 scalars.
State: (2 × SEQ_LEN × D + D) buffers  =  2×64×128+128 ≈ 16K floats — tiny.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEQ_LEN = 64


class GELU102(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau       = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        # Per-position EMA statistics
        self.register_buffer('_ema_mean', torch.zeros(_SEQ_LEN, d_model))
        self.register_buffer('_ema_sq',   torch.ones(_SEQ_LEN, d_model))
        self.register_buffer('_ema_out',  torch.zeros(d_model))   # global cosine gate
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        T_eff = min(T, _SEQ_LEN)
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── per-position z-score ───────────────────────────────────────────
        # Slice the first T_eff positions from the EMA buffers
        ema_mean = self._ema_mean[:T_eff].detach()     # (T_eff, D)
        ema_sq   = self._ema_sq[:T_eff].detach()       # (T_eff, D)
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()

        x_sub = x[:, :T_eff, :]                         # (B, T_eff, D)
        z     = (x_sub - ema_mean.unsqueeze(0)) / std.unsqueeze(0)  # (B, T_eff, D)
        surp  = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))  # (B, T_eff, 1)

        # Pad to T if T > T_eff (rare edge case)
        if T > T_eff:
            surp = F.pad(surp, (0, 0, 0, T - T_eff))

        # ── global cosine output gate ──────────────────────────────────────
        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        out_norm  = F.normalize(out, dim=-1)
        cos_out   = (out_norm * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)
        result = out * gate

        # ── per-position EMA update ────────────────────────────────────────
        # Compute per-position batch mean
        x_ptm  = x[:, :T_eff, :].detach().mean(0)          # (T_eff, D): mean over batch
        x_ptsq = x[:, :T_eff, :].detach().pow(2).mean(0)   # (T_eff, D)
        out_bm = out.detach().reshape(-1, D).mean(0)

        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean[:T_eff].copy_(x_ptm)
                self._ema_sq[:T_eff].copy_(x_ptsq)
                self._ema_out.copy_(out_bm)
                self._initialised = True
            else:
                self._ema_mean[:T_eff].mul_(decay).add_((1.0 - decay) * x_ptm)
                self._ema_sq[:T_eff].mul_(decay).add_((1.0 - decay) * x_ptsq)
                self._ema_out.mul_(decay).add_((1.0 - decay) * out_bm)

        return result
