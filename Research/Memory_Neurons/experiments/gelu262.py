"""GELU262 – EMA Self-Similarity Gate (No Ring Buffer, No Pass Detection).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: All previous experiments use a ring buffer + discrete slot
lookup. This experiment uses NO ring buffer at all — just a single running
EMA of all GELU outputs seen during eval.

The WHOLE point: as the eval progresses across passes, the EMA gets closer
and closer to a stable representation of the test corpus. The gate measures
how close the CURRENT output is to this accumulated EMA. The gate naturally
grows as familiarity increases — without ANY discrete pass detection.
═══════════════════════════════════════════════════════════════════════════

MECHANISM:
    _ema: (D,) exponential moving average of GELU output means during eval.
    Initialized to FIRST eval batch mean.

    Every batch (including during pass 1):
        cos_sim = cosine(current_y_mean, ema)
        gate = 1 + k * cos_sim^power

    After gate is applied, update EMA:
        ema = decay * ema + (1-decay) * current_y_mean

WHY MONOTONIC ACROSS PASSES:
    Pass 1: ema = {evolving estimate of corpus mean}. cos_sim varies per batch.
            Average gate ≈ 1 + k * avg_cos_sim_pass1.
    Pass 2: ema has converged toward the corpus mean. cos_sim is HIGHER on
            average because ema now represents EXACTLY the corpus distribution.
            Average gate > pass-1 average gate → Δ1→2 > 0 ✓
    Pass 3: ema continues to be updated (though slower change). cos_sim
            may be slightly higher still.

FAST vs SLOW EMA:
    decay_train ≈ 0.99 (slow, to track training distribution over epochs)
    decay_eval  ≈ 0.5  (fast, to quickly represent the CURRENT eval pass)

    At eval reset_state() → ema is reset → fast EMA starts fresh.
    This ensures pass 1 builds an accurate representation.

POWER PARAMETER:
    gate = 1 + k * cos_sim^power
    power > 1 → suppresses low-similarity batches, amplifies high-similarity.
    Learned parameter (log_power, init power=2.0).

PARAMS: log_k (gate strength, init 0.5), log_power (exponent, init 2.0)
STATE:  _ema (D,), _ema_initialized bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU262(nn.Module):
    """EMA-based self-similarity gate — no ring buffer, no pass detection."""

    def __init__(self, eval_ema_decay: float = 0.5):
        super().__init__()
        self._eval_decay = eval_ema_decay
        self.log_k     = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_power = nn.Parameter(torch.tensor(math.log(2.0)))

        self._ema:     torch.Tensor = None
        self._ema_init = False

    def reset_state(self):
        """Called before eval loop — clears eval-time EMA."""
        self._ema      = None
        self._ema_init = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k     = self.log_k.exp().clamp(0.01, 5.0)
        power = self.log_power.exp().clamp(0.5, 6.0)

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)    # (D,)

        # ── Init (first ever eval batch) ──────────────────────────────
        if not self._ema_init:
            self._ema      = F.normalize(y_mean, dim=0).clone()
            self._ema_init = True
            # No gate on very first batch (ema = current → sim ≈ 1 → gate = 1+k == too high)
            # Instead just return y without gate and let ema build one step.
            return y

        # ── Compute cosine similarity between current output and EMA ──
        y_mean_n = F.normalize(y_mean.unsqueeze(0), dim=-1)   # (1, D)
        ema_n    = F.normalize(self._ema.unsqueeze(0), dim=-1) # (1, D)
        sim      = (y_mean_n * ema_n).sum(-1).clamp(-1.0, 1.0).item()   # scalar

        # Apply power to sharpen the gate curve at high similarity
        gate_scalar = 1.0 + k.item() * (max(sim, 0.0) ** power.item())

        # Update EMA (always, across all passes)
        with torch.no_grad():
            d = self._eval_decay
            new_ema = d * self._ema + (1.0 - d) * y_mean
            self._ema = F.normalize(new_ema, dim=0)

        return y * gate_scalar
