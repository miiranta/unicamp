"""GELU130 – Dual-Mode Gate: Within-Sequence Pre-Gate × Cross-Batch Post-Gate.

THE CORE IDEA — TWO GATING POINTS, TWO NOVELTY SIGNALS:
    Prior experiments apply ONE gate at ONE location:
        - Pre-GELU:  output = GELU(x × gate)  — gelu96, gelu123
        - Post-GELU: output = GELU(x) × gate  — gelu80, gelu78, gelu71, etc.
    
    GELU130 applies TWO SEPARATE gates using DIFFERENT novelty signals:

        pre_gate  based on WITHIN-SEQUENCE novelty (gelu126 mechanism)
        post_gate based on CROSS-BATCH EMA novelty (gelu80 mechanism)
    
        output = GELU(x × pre_gate) × post_gate

WHY TWO GATES AT DIFFERENT LOCATIONS AFFECT DIFFERENT THINGS:

    Pre-GELU gate changes the OPERATING POINT of the nonlinearity:
        - Standard GELU: GELU(x) for x ~ N(0,1) activates on ~50% of inputs
        - With pre_gate=2×: GELU(2x) operates in STEEPER region → sharper discrimination
        - With pre_gate=0.5×: GELU(0.5x) operates near saturation → smoother
        
        For WITHIN-SEQUENCE novel tokens: pre_gate > 1 → steeper GELU → MORE nonlinear processing
        For WITHIN-SEQUENCE familiar tokens: pre_gate closer to 1 → normal GELU
    
    Post-GELU gate scales the OUTPUT magnitude:
        - Standard: full amplitude passes
        - With post_gate < 1: suppress output (reduce influence on subsequent layers)
        - With post_gate > 1: amplify output (increase influence)
        
        For CROSS-BATCH novel tokens: post_gate > 1 → amplify influence on final output
        For CROSS-BATCH familiar tokens: post_gate ≈ 1 → normal influence

INTERACTION BETWEEN PRE AND POST:
    A token that is novel by BOTH measures:
        → pre_gate > 1: processed with steeper GELU (nonlinear amplification)
        → post_gate > 1: result further amplified
        → Product effect: richer representation AND higher OUTPUT magnitude
    
    A token familiar globally but novel locally:
        → pre_gate > 1: steeper GELU (sharper discrimination)
        → post_gate ≈ 1: normal output scale
        → The model "thinks harder" about this token but doesn't amplify output
    
    A token novel globally but familiar locally:
        → pre_gate ≈ 1: normal nonlinearity (fits current context)
        → post_gate > 1: amplified output (unusual for training distribution)
        → The model passes through normally but scales up the unusual pattern

WITHIN-SEQUENCE SIGNAL (pre_gate):
    Causal EMA scan along T dimension (same as gelu126):
        z_local[b,t,d] = (x[b,t,d] - ema_l[b,d,t⁻]) / (std_l[b,d,t⁻] + eps)
        pre_gate = 1 + alpha_l × tanh(sigma_l × |z_local|)
    
    Stateless (no cross-batch state), causal.

CROSS-BATCH SIGNAL (post_gate):
    Per-channel EMA z-score (same as gelu80):
        z_global[b,t,d] = (x[b,t,d] - ema_g[d]) / (std_g[d] + eps)
        post_gate = 1 + alpha_g × tanh(sigma_g × |z_global|)
    
    Uses cross-batch EMA state.

STABILITY:
    pre_gate, post_gate ≥ 1 always (both amplification-only).
    Combined: GELU output scaled by up to (1+alpha_l) × (1+alpha_g) ≈ 2.25×.
    GELU is bounded for large positive inputs (saturates to x), so no explosion.
    For negative inputs, GELU → 0, so gate doesn't cause instability there.

PARAMETER COUNT:
    6 scalars: logit_decay_l, log_sigma_l, log_alpha_l, logit_decay_g, log_sigma_g, log_alpha_g
    State: _ema_mean_g (D,), _ema_sq_g (D,) — only global state.

Params: 6 scalars.
State: _ema_mean_g (D,), _ema_sq_g (D,), _ready_g (bool).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU130(nn.Module):
    """Dual-mode gate: within-sequence pre-GELU gate × cross-batch post-GELU gate."""

    def __init__(self, d_ff: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        # Within-sequence scan (pre-gate) hyperparameters
        self.logit_decay_l = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_sigma_l = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_alpha_l = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ≈ 0.5

        # Cross-batch EMA (post-gate) hyperparameters
        self.logit_decay_g = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_sigma_g = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_alpha_g = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ≈ 0.5

        # Global EMA state
        self._ema_mean_g: torch.Tensor = None
        self._ema_sq_g:   torch.Tensor = None
        self._ready_g: bool = False

    def reset_state(self):
        self._ema_mean_g = None
        self._ema_sq_g   = None
        self._ready_g    = False

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
        B, T, D = x.shape

        # ── LOCAL (within-sequence) pre-gate ─────────────────────────
        d_l     = torch.sigmoid(self.logit_decay_l).detach().item()
        sigma_l = F.softplus(self.log_sigma_l)
        alpha_l = F.softplus(self.log_alpha_l)

        ema_mean_l = torch.zeros(B, D, device=x.device)
        ema_sq_l   = torch.zeros(B, D, device=x.device)
        pre_gates  = []

        for t in range(T):
            xt = x[:, t, :].detach()
            if t == 0:
                pre_gates.append(torch.ones(B, D, device=x.device))
                ema_mean_l = xt.clone()
                ema_sq_l   = xt.pow(2)
            else:
                var_l  = (ema_sq_l - ema_mean_l.pow(2)).clamp(min=self.eps_var)
                std_l  = var_l.sqrt()
                z_l    = (xt - ema_mean_l) / (std_l + self.eps)
                pre_gates.append(1.0 + alpha_l * torch.tanh(sigma_l * z_l.abs()))
                ema_mean_l = d_l * ema_mean_l + (1.0 - d_l) * xt
                ema_sq_l   = d_l * ema_sq_l   + (1.0 - d_l) * xt.pow(2)

        pre_gate = torch.stack(pre_gates, dim=1)   # (B, T, D)

        # ── Apply pre-gate BEFORE GELU ─────────────────────────────────
        x_gated  = x * pre_gate                      # (B, T, D)
        gelu_out = self._gelu(x_gated)               # (B, T, D)

        # ── GLOBAL (cross-batch) post-gate ────────────────────────────
        d_g     = torch.sigmoid(self.logit_decay_g).detach().item()
        sigma_g = F.softplus(self.log_sigma_g)
        alpha_g = F.softplus(self.log_alpha_g)

        x_flat    = x.detach().flatten(0, -2)
        x_mean    = x_flat.mean(0)
        x_sq_mean = x_flat.pow(2).mean(0)

        if not self._ready_g:
            self._ema_mean_g = x_mean.clone()
            self._ema_sq_g   = x_sq_mean.clone()
            self._ready_g    = True
            post_gate = torch.ones(1, 1, 1, device=x.device)  # neutral on first call
        else:
            var_g     = (self._ema_sq_g - self._ema_mean_g.pow(2)).clamp(min=self.eps_var)
            std_g     = var_g.sqrt()
            # Use ORIGINAL x (not pre-gated) for global z-score (compare to history)
            z_g       = (x - self._ema_mean_g) / (std_g + self.eps)  # (B, T, D)
            post_gate = 1.0 + alpha_g * torch.tanh(sigma_g * z_g.abs())

        # Update global EMA after gate computation
        self._ema_mean_g = d_g * self._ema_mean_g + (1.0 - d_g) * x_mean
        self._ema_sq_g   = d_g * self._ema_sq_g   + (1.0 - d_g) * x_sq_mean

        # ── Apply post-gate AFTER GELU ─────────────────────────────────
        return gelu_out * post_gate
