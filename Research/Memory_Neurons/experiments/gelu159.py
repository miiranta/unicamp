"""GELU159 – Tiny GRU Surprise Integrator (Causal Sequence Accumulation).

THE KEY INSIGHT:
    All existing gates treat each token independently — the gate at position t only
    uses statistics from the current token (or the batch). But surprise is a
    *sequential* phenomenon: a token that follows a long string of familiar tokens
    is MORE surprising than the same token occurring after many novel tokens.

    A human reading text becomes progressively more alert when familiar context is
    suddenly broken. The "contextual surprise" accumulates over the sequence.

    Solution: Run a tiny GRU along the SEQUENCE DIMENSION (T=64) within each forward
    pass. The GRU input is the per-token raw surprise (mean |z-score| against EMA).
    The GRU hidden state accumulates "tension" from the surprise history within this
    specific sequence context.

    Gate = linear(GRU output) — causally depends on positions 0..t-1 ✓

    Since h_0 = 0 always: the GRU is fully stateless across batches.
    The EMA (for computing z-score inputs) IS maintained across batches.

ARCHITECTURE:
    EMA stats (for z-score): _ema_mean (D,), _ema_sq (D,)
    GRU: input_size=1 (scalar surprise), hidden_size=H=16, 1 layer
    Linear: H → 1 (scale) → sigmoid → gate multiplier

    Per forward:
        1. Compute per-token z-score surprise: s[b,t] = mean_d(|z[b,t,d]|)  (B, T)
        2. Run GRU on s.unsqueeze(-1) with h_0=0:
               h, _ = gru(s, h_0)   — (B, T, H)
        3. gate_val = 1 + alpha * sigmoid(linear(h))  — (B, T, 1)
        4. output = GELU(x) * gate_val

CAUSALITY:
    - GRU at position t uses h_{t-1} = function(positions 0..t-1) ✓
    - No cross-batch contamination (h_0 = 0 each forward) ✓
    - EMA states updated AFTER forward ✓

GRADIENT:
    - GRU parameters, linear, alpha: full gradient ✓
    - z-score computation done with no_grad on detached x ✓
    - Gate value is differentiable w.r.t. GRU params ✓

Params: GRU (input=1, hidden=H=16): ~3*(1+H+H)*H ≈ 816 params + linear (H→1) + log_alpha
State: _ema_mean (D,), _ema_sq (D,), _ready flag.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
H_GRU     = 16


class GELU159(nn.Module):
    """Tiny GRU surprise integrator: accumulates per-position surprise along sequence."""

    def __init__(self, d_ff: int):
        super().__init__()

        # Learnable GRU + linear projection for gate
        self.gru    = nn.GRU(input_size=1, hidden_size=H_GRU, batch_first=True)
        self.linear = nn.Linear(H_GRU, 1)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

        # EMA state for computing z-score input to GRU
        self._ema_mean: torch.Tensor = None   # (D,)
        self._ema_sq:   torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return out

        # ── Compute per-token z-score surprise (no grad) ──────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=1e-4)
            std  = var.sqrt()                                         # (D,)
            mu_  = self._ema_mean.view(1, 1, D)
            std_ = std.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + 1e-5)               # (B, T, D)
            surp = z.abs().mean(dim=-1, keepdim=True)                # (B, T, 1) ∈ [0, ∞)

        # ── Causal GRU over T positions ────────────────────────────────────
        # h_0 = zeros (reset each forward — no cross-batch state in GRU)
        gru_out, _ = self.gru(surp)                                  # (B, T, H)
        gate_raw   = self.linear(gru_out)                            # (B, T, 1)
        gate       = 1.0 + alpha * torch.sigmoid(gate_raw)          # (B, T, 1) ∈ (1, 2)
        output     = out * gate

        # ── Update EMA stats after forward ────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = EMA_DECAY * self._ema_mean + (1 - EMA_DECAY) * xf.mean(0)
            self._ema_sq   = EMA_DECAY * self._ema_sq   + (1 - EMA_DECAY) * xf.pow(2).mean(0)

        return output
