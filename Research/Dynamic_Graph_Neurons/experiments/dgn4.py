"""
DGN-4  –  Contrastive Dual-Graph (Reinforce + Contrast)
══════════════════════════════════════════════════════════════════════════════
Key idea: each token simultaneously aggregates information from two disjoint
neighbourhoods and blends them through a learned inhibitory gate.

    A_sim  = causal top-K_sim  most-similar  past tokens  (reinforcing)
    A_con  = causal top-K_con LEAST-similar  past tokens  (contrasting)

    msg_pos  = unweighted_mean( x, A_sim )   – "what agrees with me"
    msg_neg  = unweighted_mean( x, A_con )   – "what differs from me"

    alpha    = sigmoid( log_alpha )          ∈ (0, 1)  learned blend
    blended  = mix * x + alpha * (1-mix) * msg_pos + (1-alpha) * (1-mix) * msg_neg

    delta    = GELU( blended * gain + bias ) * scale

Motivation:
    Feeding in dissimilar context acts as a soft *lateral inhibition* signal.
    The model learns how much dissimilar context to suppress or leverage,
    which may sharpen representations and reduce over-smoothing.

Connectivity:
    K_sim = 8   (similar neighbours)
    K_con = 4   (dissimilar neighbours, fewer to avoid dominating)

All connections remain binary (0/1).  No weight matrices.
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import unweighted_aggregate

K_SIM = 8   # top-K most similar
K_CON = 4   # top-K LEAST similar (bottom-K)


def _causal_dual_adjacency(x: torch.Tensor, k_sim: int, k_con: int):
    """
    Returns two binary causal adjacency matrices.

    A_sim[b, t, s] = 1  iff  s is among the k_sim most-similar past tokens
    A_con[b, t, s] = 1  iff  s is among the k_con least-similar past tokens
                               AND s ∉ A_sim  (disjoint sets)

    Both shapes: (B, T, T).  No gradient.
    """
    B, T, D = x.shape
    with torch.no_grad():
        xn   = F.normalize(x.detach(), dim=-1)             # (B, T, D)
        sim  = xn @ xn.transpose(-2, -1)                   # (B, T, T)

        # strictly past mask (lower triangle, diagonal excluded)
        past = torch.tril(torch.ones(T, T, device=x.device), diagonal=-1)
        sim  = sim * past - 1e9 * (1.0 - past)             # mask future + self

        eff_sim = min(k_sim, T - 1)
        eff_con = min(k_con, max(0, T - 1 - eff_sim))      # can't exceed remaining pool

        A_sim = torch.zeros(B, T, T, device=x.device)
        A_con = torch.zeros(B, T, T, device=x.device)

        if eff_sim > 0:
            _, top_idx     = sim.topk(eff_sim, dim=-1)     # (B, T, k_sim)
            A_sim.scatter_(-1, top_idx, 1.0)
            A_sim = A_sim * past

        if eff_con > 0:
            # Mask positions already assigned to A_sim before picking bottom-K
            sim_con = sim.clone()
            sim_con = sim_con.masked_fill(A_sim.bool(), 1e9)   # push sim tokens away
            _, bot_idx = sim_con.topk(eff_con, dim=-1, largest=False)  # (B, T, k_con)
            A_con.scatter_(-1, bot_idx, 1.0)
            A_con = A_con * past
            A_con = A_con * (1.0 - A_sim)                  # enforce disjointness

    return A_sim, A_con


class DGN4(nn.Module):
    """
    Contrastive Dual-Graph neuron.

    Two separate binary adjacency pools (similar, dissimilar).
    Learned gate alpha controls the reinforce-vs-contrast ratio.
    """

    def __init__(self, d_model: int, k_sim: int = K_SIM, k_con: int = K_CON):
        super().__init__()
        self.k_sim = k_sim
        self.k_con = k_con

        # ── Neuron internal parameters ─────────────────────────────────────
        self.gain      = nn.Parameter(torch.ones(d_model))     # (D,) channel gain
        self.bias      = nn.Parameter(torch.zeros(d_model))    # (D,) channel bias
        self.log_mix   = nn.Parameter(torch.tensor(0.0))       # self vs msg blend
        self.log_alpha = nn.Parameter(torch.tensor(0.0))       # similar vs contrast
        self.log_scale = nn.Parameter(torch.tensor(0.0))       # output scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta
        """
        mix   = torch.sigmoid(self.log_mix)
        alpha = torch.sigmoid(self.log_alpha)     # weight on similar message
        scale = F.softplus(self.log_scale) + 0.01

        # ── 1. Dual binary adjacency ────────────────────────────────────────
        A_sim, A_con = _causal_dual_adjacency(x, self.k_sim, self.k_con)

        # ── 2. Separate unweighted messages ────────────────────────────────
        msg_pos = unweighted_aggregate(x, A_sim)              # reinforcing  (B, T, D)
        msg_neg = unweighted_aggregate(x, A_con)              # contrasting  (B, T, D)

        # ── 3. Gated blend: self + reinforce + contrast ─────────────────────
        ctx     = alpha * msg_pos + (1.0 - alpha) * msg_neg   # combined context
        blended = mix * x + (1.0 - mix) * ctx                 # (B, T, D)

        # ── 4. Neuron update ────────────────────────────────────────────────
        delta = F.gelu(blended * self.gain + self.bias) * scale

        return delta
