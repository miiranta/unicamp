"""GELU127 – Per-Channel Excess Kurtosis Gate.

THE CORE IDEA — STATISTICAL SHAPE AS NOVELTY SIGNAL:
    All prior gates measure WHERE a channel's value is relative to its mean/history.
    GELU127 asks a fundamentally different question: WHAT IS THE SHAPE of the channel's
    distribution over time?

    Kurtosis = E[X^4] / E[X^2]^2
    - Gaussian: kurtosis ≈ 3 (excess kurtosis ≈ 0)
    - Heavy-tailed: kurtosis >> 3 (excess kurtosis >> 0) — rare extreme events dominate
    - Platykurtic (light-tailed): kurtosis < 3 — values cluster near mean (familiar, boring)

    INSIGHT: Channels with HIGH excess kurtosis:
        - Have rare but EXTREME activations → those extreme events carry unusual/surprising info
        - Are "spiky" — usually quiet but occasionally fire strongly
        - Encode SPECIFIC, RARE patterns (like rare words, unusual constructs)

    Channels with LOW excess kurtosis (near-Gaussian or uniform):
        - Respond similarly to most inputs (always active at some level)
        - Encode COMMON, FAMILIAR patterns (common words, predictable structure)
        - Should be suppressed to reduce familiar signal

THE MECHANISM:
    Track per-channel 2nd and 4th moments via EMA:
        ema_m2[d] = EMA(x_d²)
        ema_m4[d] = EMA(x_d⁴)
    
    Compute kurtosis:
        kurt[d] = ema_m4[d] / (ema_m2[d]² + eps)          ≥ 0
        excess[d] = kurt[d] - 3                             ∈ (-3, ∞)
    
    Gate per channel:
        gate[d] = 1 + alpha × sigmoid(beta × excess[d])     ∈ (1, 1+alpha)
    
    High excess kurtosis (rare/spiky channel) → gate close to 1+alpha → amplified
    Negative excess kurtosis (smooth/uniform) → gate close to 1 → neutral

WHY THIS IS DIFFERENT FROM Z-SCORE GATES:
    Z-score gates: "Is THIS SPECIFIC TOKEN surprising?"  (varies by token)
    Kurtosis gate: "Is THIS CHANNEL generally informative?" (varies by channel, not token)
    
    The kurtosis gate IDENTIFIES WHICH CHANNELS tend to carry rare, specific information,
    regardless of whether the current token activates them.
    
    Combined effect: kurtosis-high channels fire rarely → when they DO fire at any level,
    their GELU output is amplified by the gate. This amplifies by channel identity.

CROSS-BATCH LEARNING:
    Kurtosis evolves over training — channels that were familiar early may become
    more specific as learning proceeds. The gate adapts automatically.

TEMPORAL STABILITY:
    M4 EMA stabilizes slowly (4th moment is sensitive to outliers).
    Use slow decay (0.99) to smooth out individual outlier batches.
    beta controls how strongly kurtosis difference translates to gate strength.

BIOLOGICAL ANALOGY:
    In neuroscience, neurons with high "sparsity" (rarely fire but fire strongly
    when they do) are associated with high-level, specific representations.
    Neurons that fire continuously with low variance are more "utility" neurons.
    Kurtosis gate implements this: amplify sparse/specific, suppress generic/continuous.

INITIALIZATION:
    ema_m2, ema_m4 initialized from first batch.
    beta=0 initially → gates all 1.0 → gradual adaptation.
    log_alpha: α ≈ 0.5 → gate range [1, 1.5].
    log_beta: β ≈ 0.3 → moderate kurtosis sensitivity.

Params: logit_decay, log_alpha, log_beta = 3 scalars.
State: _ema_m2 (D,), _ema_m4 (D,), _ready (bool).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU127(nn.Module):
    """Per-channel excess kurtosis gate: amplify channels with heavy-tailed (specific) distributions."""

    def __init__(self, d_ff: int = 1024, ema_decay: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        # alpha: max amplification, init ≈ 0.5
        self.log_alpha = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        # beta: kurtosis sensitivity (sigmoid(beta * excess_kurt))
        self.log_beta_raw  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

        self._ema_m2: torch.Tensor = None
        self._ema_m4: torch.Tensor = None
        self._ready: bool = False

    def reset_state(self):
        self._ema_m2 = None
        self._ema_m4 = None
        self._ready  = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        d     = torch.sigmoid(self.logit_decay).detach().item()
        alpha = F.softplus(self.log_alpha)
        beta  = F.softplus(self.log_beta_raw)

        # Compute batch+sequence moments for EMA update
        x_flat = x.detach().flatten(0, -2)            # (B*T, D)
        m2_batch = x_flat.pow(2).mean(0)              # (D,)
        m4_batch = x_flat.pow(4).mean(0)              # (D,)

        if not self._ready:
            self._ema_m2 = m2_batch.clone()
            self._ema_m4 = m4_batch.clone()
            self._ready  = True
            return self._gelu(x)

        # ── Compute kurtosis from current EMA state (uses PAST statistics) ──
        # kurt[d] = E[x^4] / E[x^2]^2  (raw kurtosis, Gaussian ≈ 3)
        kurt_raw = self._ema_m4 / (self._ema_m2.pow(2) + self.eps)  # (D,) ≥ 0
        excess   = kurt_raw - 3.0                                      # (D,) centered at 0 for Gaussian

        # Gate: 1 + alpha * sigmoid(beta * excess)
        # High excess → sigmoid → near 1.0 → gate near 1+alpha (amplified)
        # Negative/zero excess → sigmoid → near 0.5/0.0 → gate near 1 (neutral/slightly suppressed)
        gate = 1.0 + alpha * torch.sigmoid(beta * excess)              # (D,) ∈ (1, 1+alpha)

        # ── Update EMA moments AFTER gate computation (causal) ──────────
        self._ema_m2 = d * self._ema_m2 + (1.0 - d) * m2_batch
        self._ema_m4 = d * self._ema_m4 + (1.0 - d) * m4_batch

        # Broadcast gate to all tokens: gate is per-channel only (B, T, D) ← (D,)
        return self._gelu(x * gate)
