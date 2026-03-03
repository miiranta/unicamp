"""GELU124 – Predictive EMA with Learned Low-Rank Transform.

THE CORE IDEA — PREDICTION ERROR AS NOVELTY:
    Predictive coding theory (Rao & Ballard, 1999): the brain doesn't process
    raw sensory input. It processes PREDICTION ERRORS — deviations from what
    was expected. Familiar, predictable inputs get suppressed; unexpected
    deviations propagate up the hierarchy and trigger learning.
    
    All prior EMA gates compute: novelty = dist(x, ema)
    where ema is a PASSIVE tracker of past inputs.
    
    GELU124 uses an ACTIVE predictor:
        h_t = d × h_{t-1} + (1-d) × x̄_t      ← EMA state accumulation
        x_pred_t = B(ReLU(A(h_{t-1})))          ← LEARNED prediction from past state
        error_t = ||x[b,t] - x_pred_t|| / ||x_pred_t||  ← normalized prediction error
        gate = 1 + alpha × tanh(σ × error_t)    ← amplify surprises

THE KEY DIFFERENCE FROM gelu71/gelu80:
    gelu71: surprise = ||x - ema_x|| / ema_norm
        → direct distance to running mean (linear predictor: x_pred = ema_x)
    
    gelu124: x_pred = B(ReLU(A(h)))
        → NONLINEAR prediction through a rank-k transform
        → A maps state to a k-dimensional "expectation code"
        → B maps back to D-dimensional prediction
        → The predictor can learn "given I usually see pattern A in state h,
           predict that pattern A will continue next" — nonlinear expectation
    
    Example: if state h encodes "we're in the middle of a list pattern", 
    the predictor can learn to predict "another list item" → list items have low error,
    but suddenly a topic shift has HIGH error even if the specific words were seen before.

THE MECHANISM:
    Setup:
        A: D → rank (rank=8), no bias — compresses state to task-relevant features
        B: rank → D, no bias — decompresses prediction
        logit_decay → d ∈ (0,1), learned EMA rate
        log_sigma, log_alpha → shape and strength parameters
    
    Forward pass:
        x_mean = mean(x, dims=[B,T])              ← batch+sequence mean
        
        if not initialized:
            h = x_mean; return GELU(x)
        
        x_pred = B(ReLU(A(h))).unsqueeze(B, T)    ← prediction broadcast to all tokens
        
        error[b,t] = ||x[b,t] - x_pred|| / ||x_pred||  ← per-token prediction error
        
        surp[b,t] = tanh(σ × error[b,t])          ← bounded surprise ∈ (0,1)
        gate = 1 + alpha × surp                    ← amplify unexpected tokens
        output = GELU(x × gate)
        
        h ← d × h + (1-d) × x_mean               ← update state AFTER forward

WHY LOW-RANK (rank=8)?
    Full D×D prediction: A is 1024×1024 = 1M params (too expensive).
    Rank=8 bottleneck: 2×1024×8 = 16K params. ~1% overhead.
    The 8 rank factors encode the "key predictive dimensions" of the input.
    
    Oja/PCA theory: with good training, A will learn the top-8 principal
    directions of the EMA state covariance — the most predictive features.

CAUSAL GUARANTEE:
    h is updated AFTER the forward pass using PAST x_mean values.
    x_pred is computed from h[t-1] only — strictly causal.
    First call: h initialized from x_mean, output = GELU(x) unchanged.

SELF-REGULATION:
    If model sees many similar batches → error stays low → h ≈ x → 
    x_pred ≈ A→B mapping of mean → low error → gate ≈ 1.
    Sudden distributional shift → x_pred outdated → high error → gate boosted.

Params: A (D×rank) + B (rank×D) + logit_decay + log_sigma + log_alpha = 2Dk + 3.
For D=1024, rank=8: 16,387 per layer × 4 = 65,548 total extra.
State: _h (D,), _ready (bool).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU124(nn.Module):
    """Predictive EMA with learned low-rank transform: amplify prediction error per token."""

    def __init__(self, d_ff: int = 1024, rank: int = 8, ema_decay: float = 0.95):
        super().__init__()
        # Low-rank predictor: state h → x_pred via rank-k bottleneck
        self.A = nn.Linear(d_ff, rank, bias=False)   # D → rank
        self.B = nn.Linear(rank, d_ff, bias=False)   # rank → D

        # Learnable parameters
        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_sigma = nn.Parameter(torch.tensor(math.log(1.0)))   # sensitivity
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0)))   # amplification strength

        # EMA state
        self._h:    torch.Tensor = None
        self._ready: bool = False

        # Small init for stability
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)

    def reset_state(self):
        self._h     = None
        self._ready = False

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
        d     = torch.sigmoid(self.logit_decay)
        d_val = d.detach().item()
        sigma = F.softplus(self.log_sigma)
        alpha = F.softplus(self.log_alpha)

        # Batch+sequence mean (detach: EMA update is non-differentiable bookkeeping)
        x_mean = x.detach().mean(dim=(0, 1))  # (D,)

        if not self._ready:
            self._h     = x_mean.clone()
            self._ready = True
            return self._gelu(x)

        # ── Generate prediction from past EMA state ──────────────────
        # Detach h: prediction is computed from past state, gradient flows through A, B via x
        h_detach = self._h.detach()
        x_pred = self.B(F.relu(self.A(h_detach)))  # (D,) — learned nonlinear prediction

        # ── Per-token prediction error ────────────────────────────────
        # Broadcast prediction to all tokens
        pred_expanded = x_pred.unsqueeze(0).unsqueeze(0)      # (1, 1, D)
        error = (x - pred_expanded).norm(dim=-1)               # (B, T)

        # Normalize error by prediction magnitude for scale invariance
        pred_norm = x_pred.norm().clamp(min=1e-6)
        error_norm = error / pred_norm                          # (B, T), dimensionless

        # Bounded surprise ∈ (0, 1)
        surp = torch.tanh(sigma * error_norm)                  # (B, T)

        # Gate: amplify tokens with high prediction error
        gate = 1.0 + alpha * surp.unsqueeze(-1)                # (B, T, 1)

        # ── Update EMA state AFTER forward (causal) ──────────────────
        self._h = d_val * self._h + (1.0 - d_val) * x_mean

        return self._gelu(x * gate)
