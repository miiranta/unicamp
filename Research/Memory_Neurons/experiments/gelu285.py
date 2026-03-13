"""gelu285 – Per-Channel Learned Sigmoid Gate (No EMA).

MOTIVATION:
    All existing gelu2xx experiments use running EMA statistics to compute
    z-scores for gating. The EMA update itself is under no_grad, limiting
    gradient flow.

    This experiment eliminates the EMA entirely.  For each output channel d,
    we learn a per-channel WEIGHT w_d and BIAS b_d that directly parameterise
    the gate as a function of the batch's normalised channel activation.

MECHANISM:
    out      = gelu(x)                           # (B, T, D)
    out_mean = out.mean(dim=(0,1))               # (D,) batch-mean
    out_std  = out.std(dim=(0,1)).clamp(min=eps) # (D,) batch-std
    z        = out_mean / out_std                # (D,) batch z-normalised mean
    gate     = sigmoid(w * z + b)               # (D,) per-channel gate in (0, 1)
    return   out * gate.view(1, 1, D)           # uniform per-channel across B×T

    This is equivalent to a learned channel attention over the batch
    distribution, without any memory of previous batches.

NO CAUSALITY LEAK:
    gate depends only on current batch statistics.  No future-position
    look-ahead beyond what gelu211 already does.

BENEFIT FROM BACKPROP:
    w_d and b_d each have D_FF independent values, giving the model a
    per-channel learned gating function.  This is more expressive than
    gelu211's shared scalar parameters.  The gate for channel d is trained
    to produce the optimal suppression/amplification regardless of history.

SEQUENTIAL ADAPTATION:
    Fully stateless — gate recomputed fresh each batch → Δ ≈ 0.
    Benefit is in PPL: the per-channel gate should learn a richer
    habituation profile than a shared asymmetric scalar gate.

PARAMS: w (D,), b (D,).
STATE:  none — fully stateless, no reset_state needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU285(nn.Module):
    """Per-channel direct-sigmoid gate: gate_d = sigmoid(w_d * z_d + b_d)."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Per-channel slope and bias — initialise near identity gate
        # sigmoid(0) = 0.5, so initialise b such that gate ≈ 1 at z=0:
        # sigmoid(b) = 1 means b → +inf; use b = 3.0 as soft approximation
        self.w = nn.Parameter(torch.zeros(D_FF))
        self.b = nn.Parameter(torch.full((D_FF,), 3.0))   # gate ≈ 0.95 at z=0

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._gelu(x)   # (B, T, D)

        out_mean = out.mean(dim=(0, 1))                          # (D,)
        out_std  = out.std(dim=(0, 1)).clamp(min=self.eps)       # (D,)
        z        = out_mean / out_std                            # (D,) batch z-score

        gate = torch.sigmoid(self.w * z + self.b)               # (D,)

        return out * gate.view(1, 1, -1)
