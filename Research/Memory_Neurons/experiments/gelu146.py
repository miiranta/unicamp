"""
GELU146 — Contrastive self-identification gate (stateless InfoNCE novelty).

Insight from contrastive learning: a token is "novel" if it is HARD TO IDENTIFY
among all other tokens in the batch — its representation is unique.

InfoNCE similarity: for each token (b,t), compute cosine similarity against all
other B×T-1 tokens in the batch. Build a soft-max distribution. The gate is based
on the negative log-likelihood of the token identifying ITSELF:

    sim[i,j] = cosine(x[i], x[j]) / tau          (N,N) where N=B*T
    probs[i]  = softmax(sim[i])                   (N,)
    uniqueness[i] = -log(softmax(sim[i])[i] + eps) / log(N)    ∈ [0,1]
                  = 0 when token is identical to all others (trivial to ID)
                  = 1 when token is completely unique (impossible to confuse)

    gate = 1 + alpha * tanh(sigma * uniqueness)

No EMA state needed — purely stateless, fully differentiable.

⚠ Complexity: O(N²) = O(B²T²) per layer. With B=32, T=64, N=2048: 4M sims per layer.
  This is expensive but rich. We subsample to max N=256 tokens for efficiency.

Params: log_alpha (scalar), log_sigma (scalar), log_tau (scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GELU146(nn.Module):
    MAX_N = 256   # subsample to this many tokens for the similarity matrix

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.log_tau   = nn.Parameter(torch.tensor(0.0))   # temperature
        self._gelu = nn.GELU()

    def reset_state(self):
        pass   # stateless

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        base = self._gelu(x)

        # flatten to (N, D)
        N = B * T
        xf = x.reshape(N, D)

        # subsample if too large
        if N > self.MAX_N:
            idx = torch.randperm(N, device=x.device)[:self.MAX_N]
            xs = xf[idx]    # (MAX_N, D) with grad
        else:
            xs = xf
            idx = None

        # pairwise cosine similarity matrix
        xn = F.normalize(xs, dim=-1)                           # (M, D)
        tau = torch.exp(self.log_tau).clamp(min=0.05)
        sim = (xn @ xn.T) / tau                                # (M, M)

        # diagonal = self-similarity → highest; compute log-softmax
        log_probs = torch.log_softmax(sim, dim=-1)             # (M, M)
        # self-identification loss = -log_probs[i,i]
        M = xs.shape[0]
        neg_log_self = -log_probs[torch.arange(M, device=x.device),
                                   torch.arange(M, device=x.device)]  # (M,)
        # normalize to [0,1] by dividing by log(M)
        uniqueness_sub = neg_log_self / math.log(max(M, 2))    # (M,) ~ [0,1]

        # scatter back to (N,): sampled positions get computed value, others get mean
        if idx is not None:
            mean_u = uniqueness_sub.detach().mean()
            # build full tensor: start from detached mean, then add grad-capable delta at idx
            base_full = torch.full((N,), 0.0, device=x.device)
            base_full = base_full.index_put_((idx,), uniqueness_sub)
            # fill non-selected positions with detached mean
            not_selected = torch.ones(N, dtype=torch.bool, device=x.device)
            not_selected[idx] = False
            fill = torch.full((N,), mean_u.item(), device=x.device)
            uniqueness_full = torch.where(not_selected, fill, base_full)
        else:
            uniqueness_full = uniqueness_sub

        uniqueness = uniqueness_full.reshape(B, T, 1)           # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * uniqueness)
        return base * gate
