"""gelu289 – Prototype Attention Gate (Per-Channel).

MOTIVATION:
    gelu284 uses K=8 prototypes and a scalar gate from the MAX cosine
    similarity.  Two limitations:
    1. Scalar gate: same suppression/amplification for all channels.
    2. Max aggregation: only the closest prototype contributes.

    This experiment uses SOFT ATTENTION over K=8 prototypes to produce
    a PER-CHANNEL gate vector.  Each prototype has an associated gate
    vector G_k ∈ R^D.  The final gate is the attention-weighted blend
    of gate vectors.

MECHANISM:
    out    = gelu(x)                              # (B, T, D)
    out_m  = normalize(out.mean(dim=(0,1)))       # (D,) query
    P_n    = normalize(protos, dim=-1)            # (K, D) normalised keys
    alpha  = softmax(scale * P_n @ out_m)         # (K,) attention weights
    g      = alpha @ gates                        # (D,) weighted gate vector
    gate_d = 1 + beta * tanh(g)                   # (D,) gate centred at 1
    return out * gate_d.view(1, 1, D)            # per-channel, uniform across B×T

    protos ∈ R^{K×D}: prototype keys
    gates  ∈ R^{K×D}: per-prototype gate vectors
    log_scale, log_beta: scalar learnable parameters

    When out_m closely matches prototype k, alpha_k → 1 and gate = 1 + beta*tanh(G_k).
    Multiple prototypes blend smoothly; each prototype can amplify or suppress
    different SETS of channels.

NO CAUSALITY LEAK:
    Prototypes are fixed learned parameters; batch-mean query same as gelu284.

BENEFIT FROM BACKPROP:
    Jointly trains K prototype keys AND K gate vectors — far richer than a
    scalar gate.  Channel d's habituating response is a weighted combination
    of K learned "habituation patterns", each with their own channel profile.

SEQUENTIAL ADAPTATION:
    Stateless — Δ ≈ 0.  Benefit is in base PPL from per-channel expressiveness.

PARAMS:  protos (K×D), gates (K×D), log_scale, log_beta.
STATE:   none.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PROTO = 8


class GELU289(nn.Module):
    """Soft-attention prototype gate: per-channel gate from attention-weighted prototypes."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.D_FF  = D_FF

        # Prototype keys — attention query keys
        self.protos    = nn.Parameter(torch.randn(K_PROTO, D_FF) * 0.02)
        # Per-prototype gate vectors — small init so gate starts near 1
        self.gates     = nn.Parameter(torch.zeros(K_PROTO, D_FF))
        # Temperature for attention (start at 1.0)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        # Gate amplitude
        self.log_beta  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._gelu(x)   # (B, T, D)

        # Batch-mean query
        out_mean = out.mean(dim=(0, 1))                                 # (D,)
        query_n  = F.normalize(out_mean.unsqueeze(0), dim=-1)          # (1, D)

        # Normalised prototype keys
        proto_n  = F.normalize(self.protos, dim=-1)                    # (K, D)

        # Soft attention: alpha = softmax(scale * cos(query, key))
        scale  = self.log_scale.exp()
        logits = scale * (query_n @ proto_n.T).squeeze(0)              # (K,)
        alpha  = torch.softmax(logits, dim=0)                          # (K,)

        # Weighted blend of gate vectors
        g    = alpha @ self.gates                                       # (D,)
        beta = F.softplus(self.log_beta)
        gate = (1.0 + beta * torch.tanh(g)).clamp(0.05, 8.0)          # (D,)

        return out * gate.view(1, 1, -1)
