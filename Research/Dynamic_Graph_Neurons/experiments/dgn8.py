"""
DGN-8  –  Contrastive Progressive Expansion  (DGN-4 × DGN-6 hybrid)
══════════════════════════════════════════════════════════════════════════════
Key idea: combine the two best ideas from the previous wave.

    DGN-4: dual binary neighbourhood (similar + dissimilar) with a contrast gate
    DGN-6: looped rounds where K grows each round (4 → 8 → 16)

In each round r:
    A_sim_r = causal top-K_r  most-similar  (reinforcing)
    A_con_r = causal bottom-K_r/2 least-similar (contrasting)

    msg_pos = unweighted_mean( h, A_sim_r )
    msg_neg = unweighted_mean( h, A_con_r )

    alpha_r = sigmoid(log_alpha_r)        per-round contrast gate  ∈ (0,1)
    ctx     = alpha_r * msg_pos + (1 - alpha_r) * msg_neg
    blended = mix_r * h + (1 - mix_r) * ctx
    h_new   = GELU( blended * gain_r + bias_r )
    h       = momentum * h + (1 - momentum) * h_new

delta = (h - x) * scale

K schedule: [4, 8, 16]  →  contrast K: [2, 4, 8]  (half of K_sim each round)
R = 3 rounds

Design rationale:
    • Progressive expansion (DGN-6) lets early rounds capture tight coherence
      and later rounds integrate broader context.
    • Contrastive signal (DGN-4) refines representations at every scale level.
    • Per-round alpha_r lets the model specialise: early rounds may prefer
      similarity-driven aggregation; later rounds may leverage contrast.

All edges remain binary (0/1).  No weight matrices.
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import unweighted_aggregate

# K grows: similar neighbourhood; contrast = half of K_sim each round
K_SIM_SCHEDULE = [4, 8, 16]
K_CON_SCHEDULE = [2, 4,  8]


def _dual_adjacency_r(x: torch.Tensor, k_sim: int, k_con: int):
    """Binary causal dual-adjacency for one round (same logic as DGN-4)."""
    B, T, D = x.shape
    with torch.no_grad():
        xn   = F.normalize(x.detach(), dim=-1)
        sim  = xn @ xn.transpose(-2, -1)                    # (B, T, T)
        past = torch.tril(torch.ones(T, T, device=x.device), diagonal=-1)
        sim  = sim * past - 1e9 * (1.0 - past)

        eff_sim = min(k_sim, T - 1)
        eff_con = min(k_con, max(0, T - 1 - eff_sim))

        A_sim = torch.zeros(B, T, T, device=x.device)
        A_con = torch.zeros(B, T, T, device=x.device)

        if eff_sim > 0:
            _, top_idx = sim.topk(eff_sim, dim=-1)
            A_sim.scatter_(-1, top_idx, 1.0)
            A_sim = A_sim * past

        if eff_con > 0:
            sim_con = sim.clone()
            sim_con = sim_con.masked_fill(A_sim.bool(), 1e9)
            _, bot_idx = sim_con.topk(eff_con, dim=-1, largest=False)
            A_con.scatter_(-1, bot_idx, 1.0)
            A_con = A_con * past
            A_con = A_con * (1.0 - A_sim)

    return A_sim, A_con


class DGN8(nn.Module):
    """
    Contrastive Progressive Expansion DGN (DGN-4 × DGN-6).

    R looped rounds with growing K.  Each round uses both similar and
    dissimilar aggregation blended by a per-round learned gate alpha_r.
    """

    def __init__(self, d_model: int,
                 k_sim_schedule: list = None,
                 k_con_schedule: list = None):
        super().__init__()
        if k_sim_schedule is None:
            k_sim_schedule = K_SIM_SCHEDULE
        if k_con_schedule is None:
            k_con_schedule = K_CON_SCHEDULE
        assert len(k_sim_schedule) == len(k_con_schedule)
        self.k_sim = k_sim_schedule
        self.k_con = k_con_schedule
        self.R     = len(k_sim_schedule)

        # ── Per-round parameters ───────────────────────────────────────────
        self.gain      = nn.ParameterList([
            nn.Parameter(torch.ones(d_model))   for _ in range(self.R)])
        self.bias      = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model))  for _ in range(self.R)])
        self.log_mix   = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))     for _ in range(self.R)])
        # Per-round contrast gate: how much weight to give similar vs dissimilar
        self.log_alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))     for _ in range(self.R)])

        # ── Global parameters ──────────────────────────────────────────────
        self.log_momentum = nn.Parameter(torch.tensor(1.0))   # ~0.73 init
        self.log_scale    = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta
        """
        momentum = torch.sigmoid(self.log_momentum)
        scale    = F.softplus(self.log_scale) + 0.01

        h = x.clone()

        for r in range(self.R):
            mix   = torch.sigmoid(self.log_mix[r])
            alpha = torch.sigmoid(self.log_alpha[r])    # similar weight ∈ (0,1)

            # ── Dual binary adjacency for this round ───────────────────────
            A_sim, A_con = _dual_adjacency_r(h, self.k_sim[r], self.k_con[r])

            msg_pos = unweighted_aggregate(h, A_sim)    # reinforcing  (B, T, D)
            msg_neg = unweighted_aggregate(h, A_con)    # contrasting  (B, T, D)

            # ── Gated context: per-round contrast gate ─────────────────────
            ctx     = alpha * msg_pos + (1.0 - alpha) * msg_neg
            blended = mix * h + (1.0 - mix) * ctx

            # ── Neuron update ──────────────────────────────────────────────
            h_new = F.gelu(blended * self.gain[r] + self.bias[r])

            # ── Momentum blend ─────────────────────────────────────────────
            h = momentum * h + (1.0 - momentum) * h_new

        return (h - x) * scale
