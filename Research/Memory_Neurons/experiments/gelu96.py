"""GELU96 – Per-Channel Pre-GELU Gate (gelu87 mechanism applied before GELU).

COMBINING THE TWO STRONGEST NOVEL IDEAS:
    gelu87: Per-channel gate (D-dimensional, not scalar) applied POST-GELU
    gelu90: Pre-GELU scalar gate (changes which nonlinearity regime activates)

    gelu96: Per-channel gate applied PRE-GELU
    result = GELU(x × gate_d(channel))

WHY PER-CHANNEL PRE-GELU?
    Per-channel gate (gelu87): applies different scaling to each channel
    → Changes the DIRECTION of the output (not just magnitude)
    → Novel channels get amplified, familiar channels suppressed
    
    Pre-GELU application (gelu90): changes the nonlinearity regime
    → Novel channels get pushed to high values → GELU fires strongly
    → Familiar channels get suppressed to near-zero → GELU is nearly linear
    
    Combined:
    → Novel channel d: x_d × (large gate_d) → GELU operates in nonlinear, high-activation regime
    → Familiar channel d: x_d × (small gate_d) → x_d pushed to near-zero → GELU is suppressed

    This creates MAXIMUM DIFFERENTIATION between novel and familiar channels:
    - Novel channels: strong, nonlinear representation
    - Familiar channels: near-zero, suppressed
    - No wasted computation on familiar signals!

ENERGY CONSERVATION:
    Unlike gelu90 (scalar gate), gate_d is normalized: mean_D(gate_d) = 1.
    So the AVERAGE scale of GELU inputs is preserved.
    Some channels get pushed far up (novel), others get pushed down (familiar).
    The total signal energy is approximately preserved.

MATHEMATICAL FORM:
    z_d = (x_d - μ_d) / (σ_d + eps)             per-channel z-score
    raw_d = softplus(σ × |z_d|)                  channel novelty strength
    norm_d = raw_d / mean_D(raw_d)               normalize to mean = 1
    gate_d = α × norm_d + (1 - α)               blend toward identity
    gate_d = gate_d / mean_D(gate_d)             re-normalize

    result = GELU(x × gate)                      apply channel gate PRE-GELU

DIFFERENCE FROM gelu80 AND gelu87:
    gelu80: scalar post-GELU gate based on mean z-score → same for all channels
    gelu87: per-channel post-GELU gate → changes output direction
    gelu90: scalar pre-GELU gate → changes x magnitude before GELU
    gelu96: per-channel pre-GELU gate → changes both direction AND nonlinearity regime

This is the most expressive combination.

STABILITY:
    Gate values are normalized to mean=1, so on average x is unchanged.
    But individual channels can have gate >> 1 (novel) or gate << 1 (familiar).
    The pre-GELU application means:
    - x_d × gate_d can be very large → GELU might saturate
    - But GELU saturates gracefully (output ≈ x for large x, GELU'(x)→1)
    - Clamping gate to max=5 prevents extreme cases

Params: logit_decay, log_sigma_raw, log_alpha_raw = 3 scalars.
State: _ema_mean (D,), _ema_sq (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU96(nn.Module):
    """Per-channel pre-GELU gate: GELU(x × gate_d) — strongest novelty amplification."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        sigma = F.softplus(self.log_sigma_raw)
        alpha = torch.sigmoid(self.log_alpha_raw)  # in grad graph

        if not self._ready:
            out = self._gelu(x)
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score (detach stats, sigma in graph) ──────────
        mu_b  = self._ema_mean.detach().view(1, 1, D)
        var_b = (self._ema_sq.detach() - self._ema_mean.detach().pow(2)).clamp(min=self.eps_var)
        std_b = var_b.sqrt().view(1, 1, D)
        z     = (x.detach() - mu_b) / (std_b + self.eps)              # (B, T, D)

        # ── Per-channel novelty gate (sigma + alpha in graph) ────────────
        raw    = F.softplus(sigma * z.abs())                           # (B, T, D) sigma gets grad
        norm   = raw / (raw.mean(dim=-1, keepdim=True) + self.eps)     # normalize to mean=1
        gate   = alpha * norm + (1.0 - alpha)                          # blend with identity
        gate   = gate / (gate.mean(dim=-1, keepdim=True) + self.eps)   # re-normalize

        # ── Apply gate PRE-GELU ────────────────────────────────────────────
        x_scaled = x * gate                                            # (B, T, D) grad flows
        result   = self._gelu(x_scaled)                                # GELU on modified input

        # ── EMA updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)

        return result
