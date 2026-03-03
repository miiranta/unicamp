"""GELU151 – EMA Prototype Bank (VQ-style Novelty Gate).

THE KEY INSIGHT:
    All previous experiments measure surprise relative to a SINGLE running statistic
    (mean, variance, etc.). But the FF layer sees many different "modes" of input —
    positional patterns, token types, syntactic roles — each with its own distribution.

    A single mean/variance collapses all these modes into one blob, which means
    tokens from any mode close to the global mean look "familiar" even if they're
    actually far from their own mode's center.

    Solution: maintain K prototypes that partition the activation space.
    Each prototype tracks a different cluster of inputs.
    Novelty = distance to the nearest prototype (how far from any known cluster).

IMPLEMENTATION:
    K=16 prototypes, each D-dimensional, EMA-updated (no gradients).

    Initialization: first batch → K-means-like pick (random subsample).

    Per forward:
        1. Normalize both x and prototypes to unit sphere.
        2. Cosine distance to nearest prototype per token: d_i = 1 - max_k(cos_sim_k)
        3. gate = 1 + alpha * tanh(sigma * mean_{b,t}(d_i))
        4. output = GELU(x) * gate (broadcast scalar gate over B, T, D)

    After forward (no grad):
        5. Assign each token to nearest prototype.
        6. EMA-update each prototype centroid using its assigned tokens.
        7. If a prototype has received no tokens in this batch, leave it unchanged.

CAUSALITY: Prototypes updated AFTER forward, using detached x. ✓
GRADIENT: gate depends on alpha, sigma via detached distance scalar. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _protos (K, D) unit vectors, _ready flag.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PROTOS = 16


class GELU151(nn.Module):
    """EMA prototype bank: novelty = distance to nearest learned cluster centroid."""

    def __init__(self, d_ff: int, ema_decay: float = 0.95):
        super().__init__()
        self.d_ff  = d_ff
        self.decay = ema_decay

        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # alpha = exp(0) = 1
        self.log_sigma = nn.Parameter(torch.tensor(0.0))   # sigma = exp(0) = 1

        self._protos: torch.Tensor = None   # (K, D) unit normalized prototypes
        self._ready = False

    def reset_state(self):
        self._protos = None
        self._ready  = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if not self._ready:
            with torch.no_grad():
                # Initialize prototypes from random subsample of first batch
                xf = x.detach().flatten(0, 1)   # (N, D)
                N  = xf.shape[0]
                idx = torch.randperm(N, device=x.device)[:K_PROTOS]
                protos = F.normalize(xf[idx], dim=-1)   # (K, D)
                self._protos = protos.clone()
                self._ready  = True
            return out

        with torch.no_grad():
            xf   = x.detach().flatten(0, 1)          # (N, D)
            xn   = F.normalize(xf, dim=-1)            # (N, D) unit-normalized tokens
            pn   = F.normalize(self._protos, dim=-1)  # (K, D) unit-normalized protos

            # Cosine similarity matrix: (N, K)
            sim = xn @ pn.T                            # (N, K)
            max_sim, assign = sim.max(dim=-1)          # (N,)

            # Novelty = 1 - max cosine similarity, averaged over tokens
            novelty = (1.0 - max_sim).mean()           # scalar ∈ [0, 2], typically [0, 0.5]

        gate   = 1.0 + alpha * torch.tanh(sigma * novelty)
        output = out * gate

        # EMA prototype update (no grad, after forward)
        with torch.no_grad():
            decay = self.decay
            for k in range(K_PROTOS):
                mask = assign == k
                if mask.any():
                    centroid = xf[mask].mean(0)        # (D,)
                    centroid = F.normalize(centroid, dim=0)
                    self._protos[k] = decay * self._protos[k] + (1 - decay) * centroid
                    self._protos[k] = F.normalize(self._protos[k], dim=0)

        return output
