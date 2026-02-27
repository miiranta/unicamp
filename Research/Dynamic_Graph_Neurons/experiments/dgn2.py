"""
DGN-2  –  Novelty-Gated Adaptive Connectivity
══════════════════════════════════════════════════════════════════════════════
Key idea: a neuron decides HOW MANY connections it needs based on novelty.

    Novel token    → seeks more context  → connects to K_high past neighbours
    Familiar token → self-sufficient     → connects to K_low  past neighbours

Novelty is measured via a per-channel EMA z-score (same principle as gelu80):
    z_d       = (x_d − ema_mean_d) / ema_std_d
    surp      = tanh(σ × mean_d |z_d|)          ∈ [0, 1]
    K_t       = K_low + round( (K_high−K_low) × surp_t )   per token

This implements a form of "dynamic synaptic pruning": familiar activations
retract their dendritic reach and become more autonomous; novel activations
extend their reach to integrate more context.

Connectivity criterion  : causal top-K(t) cosine similarity  (K varies per token)
Edge weights            : NONE
Learnable params (internal):
    ema_decay: logit,  log_sigma, gain (D,), bias (D,), log_mix, log_scale
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import unweighted_aggregate

K_LOW  = 2    # connections for fully familiar token
K_HIGH = 16   # connections for fully novel token


class DGN2(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        # ── Novelty detection (EMA z-score) ───────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))   # EMA decay
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))   # z-score sensitivity
        self.register_buffer('_ema_mean', torch.zeros(d_model))
        self.register_buffer('_ema_sq',   torch.ones(d_model))
        self._initialised = False
        # ── Neuron internal parameters ─────────────────────────────────────
        self.gain      = nn.Parameter(torch.ones(d_model))
        self.bias      = nn.Parameter(torch.zeros(d_model))
        self.log_mix   = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def _novelty(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-token surprise ∈ [0,1], shape (B, T)."""
        sigma    = F.softplus(self.log_sigma_raw) + 0.01
        ema_mean = self._ema_mean.detach()
        ema_sq   = self._ema_sq.detach()
        std      = (ema_sq - ema_mean.pow(2)).clamp(min=1e-6).sqrt()
        z        = (x - ema_mean) / std                          # (B, T, D)
        surp     = torch.tanh(sigma * z.abs().mean(-1))          # (B, T)
        return surp

    def _adaptive_adjacency(self, x: torch.Tensor, surp: torch.Tensor) -> torch.Tensor:
        """
        Build a binary causal adjacency matrix where each token t gets
        K_t connections determined by its novelty score.
        Fully vectorised — no Python loops over B or T.
        """
        B, T, D = x.shape
        # Per-token K: (B, T), fully detached from graph
        K_t = (K_LOW + (K_HIGH - K_LOW) * surp.detach()).round().long()
        K_t = K_t.clamp(min=0, max=min(K_HIGH, T - 1))          # (B, T)

        with torch.no_grad():
            xn   = F.normalize(x.detach(), dim=-1)
            sim  = xn @ xn.transpose(-2, -1)                     # (B, T, T)
            past = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool),
                              diagonal=-1)                        # (T, T)
            # Mask out self + future with -inf so they never win a rank
            sim  = sim.masked_fill(~past.unsqueeze(0), -1e9)     # (B, T, T)

            # Argsort descending → rank[b, t, j] = rank of position j
            order = sim.argsort(dim=-1, descending=True)         # (B, T, T)
            rank  = torch.empty_like(order)
            rank.scatter_(-1, order,
                          torch.arange(T, device=x.device)
                               .view(1, 1, T).expand(B, T, T))  # (B, T, T)

            # A[b,t,j]=1 iff rank of j < K_t[b,t]  AND  j is a past token
            A = (rank < K_t.unsqueeze(-1)).float()               # (B, T, T)
            A = A * past.unsqueeze(0).float()                    # enforce causality

        return A     # (B, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mix   = torch.sigmoid(self.log_mix)
        scale = F.softplus(self.log_scale) + 0.01

        # ── 1. Per-token novelty (carries grad via sigma) ──────────────────
        surp = self._novelty(x)                                  # (B, T)

        # ── 2. Novelty-adaptive binary adjacency ───────────────────────────
        A   = self._adaptive_adjacency(x, surp)                  # (B, T, T)
        msg = unweighted_aggregate(x, A)                         # (B, T, D)

        # ── 3. Neuron internal processing  ─────────────────────────────────
        blended = mix * x + (1.0 - mix) * msg
        delta   = F.gelu(blended * self.gain + self.bias) * scale

        # ── 4. EMA state update ────────────────────────────────────────────
        xf    = x.detach().reshape(-1, D)
        xb    = xf.mean(0)
        xsq   = xf.pow(2).mean(0)
        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_mean.copy_(xb)
                self._ema_sq.copy_(xsq)
                self._initialised = True
            else:
                self._ema_mean.mul_(decay).add_((1 - decay) * xb)
                self._ema_sq.mul_(decay).add_((1 - decay) * xsq)

        return delta
