"""GELU175 – Per-Channel Vote-Balance (Directional Fatigue) Gate.

THE KEY INSIGHT:
    Over training, neurons develop "preferred directions" — they tend to be
    consistently above or below their mean activation. This is a form of
    "directional fatigue": the neuron always votes the same way.

    A stably biased neuron (fatigue_d → 0 or 1) is in a FAMILIAR mode.
    A genuinely undecided neuron (fatigue_d → 0.5) is in a NOVEL mode.

    fatigue_d = EMA of P(x_d[b,t] > ema_mean_d)        ∈ [0, 1]

    Directional instability (novelty):
        novelty_d = 0.5 - |fatigue_d - 0.5|             ∈ [0, 0.5]
          = 0   when fatigue_d = 0 or 1 (perfectly stable direction)
          = 0.5 when fatigue_d = 0.5   (maximally undecided)

    Average novelty across channels:
        surp = 2 * mean_d(novelty_d)                     ∈ [0, 1]
    (scaled to [0,1] so sigma doesn't need compensating)

    gate = 1 + alpha * tanh(sigma * surp)
    output = GELU(x) * gate

    TRAINING DYNAMICS:
        Early training:  all channels near 0.5 → high novelty → strong gating
        Late training:   channels settle into preferred territories → lower novelty
        This provides a natural "curriculum": stronger gating pressure when more novel

    DISTINCT FROM EXISTING:
    - gelu163: detects momentary sign FLIP (current vs previous batch)
    - gelu175: tracks long-run STABILITY of preferred direction (integral over time)
    - gelu161: rank extremeness — purely spatial (within batch), not temporal

CAUSALITY: vote is computed from current x vs EMA mean (historical). ✓
           EMA updated after forward, detached. ✓
GRADIENT: fatigue computation entirely under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,), _ema_fatigue (D,).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EMA_FAST = 0.95   # For mean tracking (fast-ish)
EMA_SLOW = 0.995  # For fatigue tracking (very slow — long-run statistic)


class GELU175(nn.Module):
    """Per-channel vote-balance gate: undecided neurons = novel regime."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_mean: torch.Tensor    = None   # (D,)
        self._ema_fatigue: torch.Tensor = None   # (D,) in [0,1]
        self._ready = False

    def reset_state(self):
        self._ema_mean    = None
        self._ema_fatigue = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        # ── Initialise on first call ──────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                flat   = x.detach().flatten(0, 1)          # (B*T, D)
                mean_d = flat.mean(0)                       # (D,)
                self._ema_mean    = mean_d.clone()
                self._ema_fatigue = torch.full_like(mean_d, 0.5)   # unbiased init
                self._ready = True
            return out

        # ── Compute gate using CURRENT x vs HISTORICAL mean ──────────────────
        with torch.no_grad():
            flat   = x.detach().flatten(0, 1)              # (B*T, D)
            mean_d = flat.mean(0)                           # (D,)

            # vote: fraction of current batch tokens above historical mean
            above  = (flat > self._ema_mean).float().mean(0)   # (D,) in [0,1]

            # directional instability (closer to 0.5 = more novel)
            novelty_d = 0.5 - (self._ema_fatigue - 0.5).abs()  # (D,) in [0,0.5]
            surp      = 2.0 * novelty_d.mean()                  # scalar in [0,1]

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        # ── Update EMAs (strictly after forward, detached) ────────────────────
        with torch.no_grad():
            flat_nw   = x.detach().flatten(0, 1)
            mean_d_nw = flat_nw.mean(0)
            above_nw  = (flat_nw > self._ema_mean).float().mean(0)

            self._ema_mean    = EMA_FAST * self._ema_mean    + (1 - EMA_FAST) * mean_d_nw
            self._ema_fatigue = EMA_SLOW * self._ema_fatigue + (1 - EMA_SLOW) * above_nw

        return output
