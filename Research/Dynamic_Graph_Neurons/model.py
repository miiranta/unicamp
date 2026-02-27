"""
Dynamic Graph Neuron Language Model  –  model.py
═══════════════════════════════════════════════════════════════════════════════
Core idea
─────────
Replace the FFN in each transformer layer with a DYNAMIC GRAPH MESSAGE PASSING
module.  Key design principles:

  • Connections are BINARY (0 or 1) — there are no learnable edge weights.
  • Connectivity is determined by a CONTENT CRITERION (similarity, novelty …),
    not by learning which neuron affects which.
  • Learnable parameters live INSIDE each neuron (gain, bias, blend, scale).
    They control HOW a neuron processes incoming messages, not WHERE they go.
  • All connectivity is strictly CAUSAL: token t may only read from 0 … t-1.

Architecture per layer:
  x  →  MaskedSelfAttention  →  norm  →  DGN(x)  →  norm  →  next layer
  (standard residual formulation — DGN returns a DELTA added to x)
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(BASE_DIR, "..", "Memory_Neurons", "dataset", "wikitext-2")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    D_MODEL    = 128
    N_HEADS    = 4
    N_LAYERS   = 4
    DROPOUT    = 0.1
    SEQ_LEN    = 64
    BATCH_SIZE = 32
    EPOCHS     = 15
    LR         = 3e-4
    GRAD_CLIP  = 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ──────────────────────────────────────────────────────────────────────────────
#  Shared graph utilities
# ──────────────────────────────────────────────────────────────────────────────
def causal_topk_adjacency(x: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute a binary causal adjacency matrix via top-K cosine similarity.

    For each token t, connect to the K most similar PAST tokens (positions < t).
    No edge weights — the result is 0/1.

    Returns A: (B, T, T)  where A[b, t, s] = 1  iff  s is one of the top-K
    most similar past tokens to t in batch item b.
    """
    B, T, D = x.shape
    with torch.no_grad():
        xn  = F.normalize(x.detach(), dim=-1)           # (B, T, D)
        sim = xn @ xn.transpose(-2, -1)                 # (B, T, T)

        # Strictly past mask (lower triangle, diagonal = 0)
        past = torch.tril(torch.ones(T, T, device=x.device), diagonal=-1)
        sim  = sim * past - 1e9 * (1.0 - past)          # mask future + self to -inf

        eff_K = min(K, T - 1)
        if eff_K <= 0:
            return torch.zeros(B, T, T, device=x.device)

        _, idx = sim.topk(eff_K, dim=-1)                # (B, T, K)
        A = torch.zeros(B, T, T, device=x.device)
        A.scatter_(-1, idx, 1.0)
        A = A * past                                     # ensure causal (edge case: first token)
    return A                                             # (B, T, T) binary


def unweighted_aggregate(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Unweighted mean aggregation.  Each neuron receives the plain mean of its
    connected neighbours' feature vectors — no edge weights.

    x : (B, T, D)
    A : (B, T, T)  binary adjacency
    returns : (B, T, D)  mean message (0-vector when no connections)
    """
    n   = A.sum(dim=-1, keepdim=True).clamp(min=1.0)    # (B, T, 1)
    return (A @ x) / n                                   # (B, T, D)


# ──────────────────────────────────────────────────────────────────────────────
#  DGN Block  (one transformer layer with DGN replacing the FFN)
# ──────────────────────────────────────────────────────────────────────────────
class DGNBlock(nn.Module):
    """
    Standard causal self-attention  →  LayerNorm  →  DGN message passing  →  LayerNorm.
    The DGN module's forward() must return a DELTA (same shape as x).
    """
    def __init__(self, d_model: int, nhead: int, dropout: float, dgn_module: nn.Module):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.dgn   = dgn_module

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # 1. Causal self-attention
        a, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.norm1(x + self.drop(a))
        # 2. DGN message passing (replaces FFN; returns delta)
        x = self.norm2(x + self.drop(self.dgn(x)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  DGN Language Model
# ──────────────────────────────────────────────────────────────────────────────
class DGNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, dgn_cls):
        """
        dgn_cls: callable  dgn_cls(d_model) → nn.Module
            Each layer gets its OWN DGN instance (independent learned params).
        """
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, cfg.DROPOUT)
        self.layers  = nn.ModuleList([
            DGNBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT, dgn_cls(cfg.D_MODEL))
            for _ in range(cfg.N_LAYERS)
        ])
        self.norm = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        out  = self.pos_enc(self.embed(x))
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1
        )
        for layer in self.layers:
            out = layer(out, causal_mask)
        return self.head(self.norm(out))   # (B, T, vocab)
