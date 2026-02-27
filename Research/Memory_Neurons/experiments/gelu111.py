"""
gelu111 – Neighbor-contrast gate  (within-sequence local novelty)
─────────────────────────────────────────────────────────────────────────────
A token is "locally novel" if it is DIFFERENT from its immediate neighbors.
Rather than comparing to a global EMA, compare each token to its two
neighbors via cosine similarity:

    sim_t = (cos(x_t, x_{t-1}) + cos(x_t, x_{t+1})) / 2     (clamped at edges)
    novelty_t = 1 – sim_t                                      ∈ [0, 2]
    surp_t    = tanh(σ × novelty_t)
    gate_t    = 1 + w × surp_t
    result    = GELU(x) × gate   (elementwise over time)

Hypothesis: locally distinctive tokens (peaks, starts of new topics) carry
more information and should be amplified, regardless of global statistics.
Completely stateless — no buffers needed.
Parameters: log_sigma_raw, log_w_raw  →  2 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU111(nn.Module):
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

        out  = self._gelu(x)
        xn   = F.normalize(x, dim=-1)   # (B, T, D)

        # causal: only left (previous) neighbor; pad position 0 with itself
        left    = torch.cat([xn[:, :1, :], xn[:, :-1, :]], dim=1)   # (B,T,D)
        sim     = (xn * left).sum(-1, keepdim=True)                  # (B,T,1)
        novelty = 1.0 - sim

        surp   = torch.tanh(sigma * novelty)
        gate   = 1.0 + w * surp
        result = out * gate
        return result
