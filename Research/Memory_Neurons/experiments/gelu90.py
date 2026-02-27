"""GELU90 – Pre-GELU Familiarity Gate: GELU(x × gate) instead of GELU(x) × gate.

THE CRITICAL DISTINCTION FROM ALL PRIOR EXPERIMENTS:
    EVERY prior experiment (gelu80, gelu78, gelu85...):
        output = GELU(x) × gate   ← POST-GELU multiplicative scaling

    This experiment:
        output = GELU(x × gate)   ← PRE-GELU input scaling

WHY THIS MATTERS — THE NONLINEARITY REGIME ARGUMENT:
    GELU has distinct regimes:
        x << 0: nearly zero (gating)               GELU'(x) ≈ 0
        x ≈ 0:  nearly linear                      GELU'(x) ≈ 0.5
        x ≈ 0.5: maximum curvature (most nonlinear)
        x >> 1: nearly linear (passes through)     GELU'(x) ≈ 1

    Post-GELU gating ONLY changes magnitude. The nonlinear behavior of GELU is
    fixed — it fires or not based on x alone. Novel tokens get a louder version
    of the SAME representation.

    Pre-GELU gating changes WHICH REGIME of the nonlinearity operates:
    - Familiar token (gate = 0.3): GELU(0.3x) → input is small → GELU is near-linear
      → output ≈ 0.15x → bland, linear representation → gradient = 0.15 × GELU'(0.3x)
    - Novel token (gate = 2.0): GELU(2.0x) → input is large → GELU is nonlinear
      → RICH, curved representation → gradient = 2.0 × GELU'(2.0x)

    The model LITERALLY LEARNS DIFFERENT FEATURES for novel vs familiar tokens.
    Novel tokens engage GELU's nonlinear computation fully.
    Familiar tokens get linearly blended down (essentially just linear activation).

    This is analogous to:
    - Familiarization in psychology: automatic/stimulus-driven (linear) vs deliberate/effortful (nonlinear)
    - In neuroscience: habituated firing (gain-reduced, linear) vs novel-driven (bursting, nonlinear)

THE GATE MECHANISM (same as gelu80):
    Per-channel z-score: z_d = (x_d - μ_d) / (σ_d + eps)
    Scalar surprise: surp = tanh(σ × mean_d(|z_d|))
    Output cosine: cos_out = cosine(GELU(x), ema_out)
    gate_scalar = exp(-τ × cos_out) × (1 + w × surp)

    Then: output = GELU(x × clamp(gate_scalar, min=min_gate).unsqueeze(-1))

WHY CLAMP?
    gate_scalar can be close to 0 (strong familiar suppression).
    GELU(x × 0) = 0 for all x — no information passed through.
    min_gate = 0.1 ensures some signal always passes.
    max_gate: unbounded but GELU naturally saturates for large inputs.

GRADIENT ANALYSIS:
    d_output/d_x = d/dx GELU(x × g) = GELU'(x × g) × g
    When g >> 1: gradient amplified by g (strong learning signal for novel)
    When g << 1: gradient multiplied by g → small (weak update for familiar)
    This is AUTOMATIC GRADIENT FOCUSING: the model updates strongly from novel inputs
    and weakly from familiar ones — WITHOUT explicit meta-learning!

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_out (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU90(nn.Module):
    """Pre-GELU familiarity gate: GELU(x × gate) — changes which nonlinearity regime activates."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5, min_gate: float = 0.1):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.min_gate = min_gate
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        # Compute standard GELU for warm-up and EMA tracking
        out_standard = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out_standard.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(of.mean(0), dim=0).clone()
                self._ready    = True
            return out_standard

        # ── Per-channel z-score (detach stats, sigma in graph) ──────────
        mu_b  = self._ema_mean.detach().view(1, 1, D)
        var_b = (self._ema_sq.detach() - self._ema_mean.detach().pow(2)).clamp(min=self.eps_var)
        std_b = var_b.sqrt().view(1, 1, D)
        z     = (x.detach() - mu_b) / (std_b + self.eps)             # (B, T, D)
        mean_abs_z = z.abs().mean(dim=-1)                             # (B, T)
        surp  = torch.tanh(sigma * mean_abs_z)                        # (B, T) sigma gets grad

        # ── Output cosine of standard GELU ──────────────────────────────
        out_norm = F.normalize(out_standard.detach(), dim=-1)
        ema_norm = self._ema_out.detach().view(1, 1, D)
        cos_out  = (out_norm * ema_norm).sum(dim=-1)                  # (B, T)

        # ── Scalar gate ───────────────────────────────────────────────
        gate = torch.exp(-tau * cos_out) * (1.0 + w * surp)          # (B, T) tau,w get grad
        gate = gate.clamp(min=self.min_gate)                          # prevent complete suppression

        # ── PRE-GELU scaling: x_scaled = x × gate  ───────────────────
        x_scaled = x * gate.unsqueeze(-1)                             # (B, T, D) — grad flows
        result   = self._gelu(x_scaled)                               # GELU on modified input

        # ── EMA updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out_standard.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out  = F.normalize(d_val * self._ema_out + (1 - d_val) * of.mean(0), dim=0)

        return result
