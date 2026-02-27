"""
gelu114 – Causal sequence coherence gate
─────────────────────────────────────────────────────────────────────────────
Measures how much token t coheres with the CAUSAL running mean of all previous
tokens 0..t-1 in the same sequence:

    causal_mean_t = cumsum(x, dim=1)[:, t-1, :] / t   (mean of positions 0..t-1)
    coherence_t   = cosine(x_t, causal_mean_t)
    novelty_t     = 1 – coherence_t               ∈ [0, 2]
    surp_t        = tanh(σ × novelty_t)
    gate_t        = 1 + w × surp_t
    result        = GELU(x) × gate

Position 0 has no prior context → coherence = 1 (no novelty) by construction.
Fully stateless (no EMA). Parameters: log_sigma_raw, log_w_raw  →  2 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU114(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── causal running mean of x (positions 0..t-1) ────────────────────
        # cumsum[:, t, :] = sum of positions 0..t
        # shift right by 1 → cumsum[:, t-1, :] = sum of positions 0..t-1
        cum = torch.cumsum(x.detach(), dim=1)                         # (B, T, D)
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)  # [1..T]
        # shifted: index t gets sum of 0..t-1 (divide by t)
        cum_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), cum[:, :-1, :]], dim=1)
        counts_prev = torch.cat([torch.ones(1, device=x.device), counts[:-1]])  # avoid /0 at t=0
        causal_mean = cum_prev / counts_prev.view(1, T, 1)            # (B, T, D)

        xn        = F.normalize(x, dim=-1)
        cm_norm   = F.normalize(causal_mean, dim=-1)
        coherence = (xn * cm_norm).sum(-1, keepdim=True)              # (B, T, 1)
        # at t=0, causal_mean=0 → cm_norm=0 → coherence=0 → novelty=1 (treat as novel)
        novelty   = 1.0 - coherence

        surp   = torch.tanh(sigma * novelty)
        gate   = 1.0 + w * surp
        result = out * gate
        return result
