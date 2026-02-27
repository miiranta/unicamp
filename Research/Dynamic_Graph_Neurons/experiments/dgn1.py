"""
DGN-1  –  Basic Similarity-Based Dynamic Connectivity
══════════════════════════════════════════════════════════════════════════════
Connectivity criterion  : causal top-K cosine similarity
Edge weights            : NONE (binary 0/1)
Learnable params (internal to each neuron):
    gain    (D,)   – per-channel output amplification
    bias    (D,)   – per-channel additive bias
    log_mix  scalar – blend(self, message) ratio via sigmoid
    log_scale scalar – output scaling via softplus

Forward:
    A       = causal_topk( x, K=8 )            # binary (B,T,T) — NO grad
    msg     = unweighted_mean( x, A )          # (B,T,D)  — grad flows through x
    blended = mix * x  +  (1-mix) * msg        # per-token blend
    delta   = GELU( blended * gain + bias ) * scale
    return delta   (added as residual in DGNBlock)
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency, unweighted_aggregate

K_DEFAULT = 8   # how many past neighbours each neuron connects to


class DGN1(nn.Module):
    """
    Basic dynamic graph neuron.
    Connections: top-K cosine similarity, causal, binary.
    Neuron internal params: gain, bias, mix, scale.
    No weight matrices shared between neurons — all params are element-wise.
    """

    def __init__(self, d_model: int, K: int = K_DEFAULT):
        super().__init__()
        self.K = K
        # ── Internal neuron parameters  ────────────────────────────────────
        # These govern HOW each neuron processes its inputs.
        # There are NO (D×D) matrices coupling neurons — only element-wise ops.
        self.gain      = nn.Parameter(torch.ones(d_model))    # (D,) channel gain
        self.bias      = nn.Parameter(torch.zeros(d_model))   # (D,) channel bias
        self.log_mix   = nn.Parameter(torch.tensor(0.0))      # scalar: self vs msg blend
        self.log_scale = nn.Parameter(torch.tensor(0.0))      # scalar: output scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, T, D)  –  current token hidden states
        returns (B, T, D)  –  delta  (added as residual outside)
        """
        mix   = torch.sigmoid(self.log_mix)
        scale = F.softplus(self.log_scale) + 0.01

        # ── 1. Binary adjacency  (no gradient) ─────────────────────────────
        A   = causal_topk_adjacency(x, self.K)          # (B, T, T) binary

        # ── 2. Unweighted message aggregation  ─────────────────────────────
        #    Gradient flows through x (the values), NOT through A (the routing).
        msg = unweighted_aggregate(x, A)                # (B, T, D)

        # ── 3. Neuron internal processing  ─────────────────────────────────
        blended = mix * x + (1.0 - mix) * msg           # (B, T, D)
        delta   = F.gelu(blended * self.gain + self.bias) * scale

        return delta
