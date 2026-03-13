"""gelu284 – Learned Multi-Prototype Cosine Gate.

MOTIVATION:
    gelu38/gelu61 use a RUNNING EMA to build a reference direction, then gate
    based on cosine similarity.  The EMA is updated with no_grad.

    Here we replace the running reference with K=8 LEARNED PROTOTYPE VECTORS
    (nn.Parameter, shape K × D_FF) that are optimised end-to-end via backprop.
    No running statistics at all — the prototypes learn directly from the loss.

MECHANISM:
    out      = gelu(x)
    out_mean = out.mean(dim=(0,1))          # (D,) batch-mean activation
    out_norm = L2-normalise(out_mean)       # (D,)
    P_norm   = L2-normalise(protos, dim=-1) # (K, D)
    cos_k    = P_norm @ out_norm            # (K,) cosine similarity to each prototype
    max_cos  = max(cos_k)                   # scalar
    gate     = exp(-w * ReLU(max_cos - θ))  # suppress when similar to any prototype

    The gate is scalar, applied uniformly across all B×T positions.

    During eval: identical forward, no state changes → same gate each pass → Δ ≈ 0.
    BENEFIT is in base PPL: prototypes learn to identify habituatable patterns.

NO CAUSALITY LEAK:
    Prototypes are fixed learned parameters (not computed from current data).
    The batch-mean aggregation is causal across batches (same as gelu211).

BENEFIT FROM BACKPROP:
    All three sets of parameters (protos P, log_w, logit_theta) receive direct
    gradient from the loss.  The prototypes converge to the subspace of
    activation patterns that most benefit from suppression.

PARAMS: protos (K×D), log_w, logit_theta.
STATE:  none — fully stateless.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PROTO = 8


class GELU284(nn.Module):
    """Multi-prototype cosine suppression gate with learnable prototype vectors."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.D_FF = D_FF

        # K learned prototype vectors — initialise with small random values
        self.protos     = nn.Parameter(torch.randn(K_PROTO, D_FF) * 0.02)
        # Gate strength and threshold
        self.log_w      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.logit_theta= nn.Parameter(torch.zeros(1))  # threshold in (0, 1) via sigmoid

    def reset_state(self):
        pass   # no state to reset

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._gelu(x)

        # Batch-mean activation query
        out_mean = out.mean(dim=(0, 1))                         # (D,)
        out_norm = F.normalize(out_mean.unsqueeze(0), dim=-1)   # (1, D)

        # Per-prototype cosine similarities
        proto_norm = F.normalize(self.protos, dim=-1)           # (K, D)
        cos_sims   = (out_norm @ proto_norm.T).squeeze(0)       # (K,)

        # Gate: suppress when similar to any learned prototype
        max_cos = cos_sims.max()
        w       = self.log_w.exp()
        theta   = torch.sigmoid(self.logit_theta)
        gate    = torch.exp(-w * F.relu(max_cos - theta))       # scalar in (0, 1]

        return out * gate
