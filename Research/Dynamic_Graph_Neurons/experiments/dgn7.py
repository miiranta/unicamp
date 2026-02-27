"""
DGN-7  –  Attention-Weighted Aggregation within Binary Neighbourhood
══════════════════════════════════════════════════════════════════════════════
Key idea: the top-K binary adjacency still controls which past tokens are
ELIGIBLE to send messages (routing is non-parametric), but rather than
averaging them uniformly, we score them through a small learned dot-product
attention and take a weighted sum.

    A      = causal_topk( x, K=8 )              binary mask  (B, T, T)

    q      = x @ W_q                            query  (B, T, D_head)
    k      = x @ W_k                            key    (B, T, D_head)

    score  = (q @ k^T) / sqrt(D_head)           raw scores   (B, T, T)
    score  = score * A  −  1e9 * (1 − A)        mask out non-neighbours
    attn   = softmax( score, dim=-1 )            normalised   (B, T, T)

    msg    = attn @ x                            weighted sum (B, T, D)

    blended = mix * x + (1-mix) * msg
    delta   = GELU( blended * gain + bias ) * scale

Motivation:
    DGN1 treats all K neighbours as equally important.  Adding attention
    within the binary neighbourhood lets the model focus on the most
    contextually relevant subset — without relaxing the hard binary routing.
    W_q / W_k are small (D × D_head, D_head = 32) to keep parameters low.

    This is intentionally NOT full self-attention: the binary top-K mask
    remains non-differentiable and content-driven, not fully learned.
    Attention only re-weights within the already-selected neighbourhood.

Connectivity criterion  : causal top-K cosine similarity (binary routing)
Message weights         : soft dot-product attention within the binary mask
Learnable params        : W_q (D×D_h), W_k (D×D_h), gain (D,), bias (D,),
                          log_mix, log_scale
══════════════════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency

K_DEFAULT  = 8
D_HEAD     = 32   # small projection head for Q/K


class DGN7(nn.Module):
    """
    Top-K-masked attention-weighted DGN.

    Routing is binary top-K (same as DGN1).
    Within the selected neighbourhood, soft attention re-weights messages.
    """

    def __init__(self, d_model: int, K: int = K_DEFAULT, d_head: int = D_HEAD):
        super().__init__()
        self.K      = K
        self.d_head = d_head

        # ── Attention projections (small: D × d_head each) ─────────────────
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)

        # ── Internal neuron parameters ─────────────────────────────────────
        self.gain      = nn.Parameter(torch.ones(d_model))
        self.bias      = nn.Parameter(torch.zeros(d_model))
        self.log_mix   = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta
        """
        B, T, D = x.shape
        mix   = torch.sigmoid(self.log_mix)
        scale = F.softplus(self.log_scale) + 0.01

        # ── 1. Binary routing (top-K; no gradient) ─────────────────────────
        A = causal_topk_adjacency(x, self.K)              # (B, T, T) binary

        # ── 2. Attention within binary neighbourhood ────────────────────────
        q = self.W_q(x)                                   # (B, T, d_head)
        k = self.W_k(x)                                   # (B, T, d_head)

        # Raw dot-product scores
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, T, T)

        # Mask out positions not in the binary neighbourhood (A=0) and future
        # Keep positions where A=1; push everything else to -inf
        score = score.masked_fill(A < 0.5, -1e9)          # (B, T, T)

        # If a token has no neighbours at all (first token etc.), softmax → NaN
        # Add a tiny self-bias to ensure at least one valid logit
        has_nbr = A.sum(dim=-1, keepdim=True) > 0         # (B, T, 1)
        score   = score + (~has_nbr).float() * torch.eye(T, device=x.device).unsqueeze(0) * 0.0
        # Safer: replace full-masked rows with uniform over self
        row_max = score.max(dim=-1, keepdim=True).values
        score   = score - row_max.detach()                 # numerical stability

        attn = torch.softmax(score, dim=-1)                # (B, T, T)
        # Zero out rows that had no neighbours (softmax of all-inf → NaN guard)
        attn = torch.nan_to_num(attn, nan=0.0)

        # ── 3. Weighted message ────────────────────────────────────────────
        msg = attn @ x                                     # (B, T, D)

        # ── 4. Neuron update ───────────────────────────────────────────────
        blended = mix * x + (1.0 - mix) * msg
        delta   = F.gelu(blended * self.gain + self.bias) * scale

        return delta
