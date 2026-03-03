"""GELU129 – Dual-Reference Novelty Gate: Cross-Batch × Within-Sequence.

THE CORE IDEA — TWO ORTHOGONAL NOVELTY SIGNALS:
    gelu80 (best so far at 7%): per-channel z-score vs CROSS-BATCH EMA history.
        "Is this channel unusual relative to everything seen in training so far?"
    
    gelu126: per-channel z-score vs WITHIN-SEQUENCE causal EMA scan.
        "Is this position unusual relative to earlier positions in THIS sequence?"
    
    These measure DIFFERENT axes of novelty:
    
        High global z + High local z = DOUBLY NOVEL: rare in training AND rare in context
        High global z + Low local z  = GLOBALLY NOVEL but fits current context
        Low global z  + High local z = LOCALLY NOVEL: common word in unusual context position
        Low global z  + Low local z  = DOUBLY FAMILIAR: common word, expected here
    
    Only gelu129 can handle all 4 cases with appropriate amplification.
    
    Key insight: this lets the model distinguish between e.g.:
    - "cat" (common word, low global z) appearing at the START of a list (locally fresh)
      → slight local amplification only
    - A proper noun (rare globally) mid-sentence (locally consistent)
      → global amplification only
    - Technical jargon (rare globally) + topic shift (locally unusual)
      → maximum double amplification

THE MECHANISM — PRODUCT GATE:
    Global signal (cross-batch EMA per-channel z-score):
        z_global[b,t,d] = (x[b,t,d] - ema_mean_g[d]) / (ema_std_g[d] + eps)
        surp_global[b,t,d] = tanh(σ_g × |z_global[b,t,d]|)    ∈ (0,1)
    
    Local signal (within-sequence causal EMA scan):
        z_local[b,t,d] = (x[b,t,d] - ema_mean_l[b,d,t⁻]) / (ema_std_l[b,d,t⁻] + eps)
        surp_local[b,t,d] = tanh(σ_l × |z_local[b,t,d]|)     ∈ (0,1)
    
    Combined gate (additive blend):
        gate[b,t,d] = 1 + α_g × surp_global[b,t,d]
                        + α_l × surp_local[b,t,d]
    
    Parameters α_g and α_l are learned separately — the model determines
    whether global or local novelty is more useful for this particular layer.

WHY ADDITIVE NOT MULTIPLICATIVE:
    Multiplicative: gate = (1 + α_g × surp_g) × (1 + α_l × surp_l)
        → doubly novel tokens amplified by up to (1+α_g)×(1+α_l) ≈ 2.25×
        → but gradient harder to optimize (product of two learned quantities)
    
    Additive: gate = 1 + α_g × surp_g + α_l × surp_l
        → maximum amplification = 1 + α_g + α_l ≈ 2.0×
        → gradient cleaner: ∂gate/∂α_g and ∂gate/∂α_l are independent
        → model learns which signal matters more via α magnitudes

STABILITY:
    gate ≥ 1 always (surp ≥ 0, α > 0 via softplus).
    gate ≤ 1 + α_g + α_l.
    Initial α_g ≈ 0.5, α_l ≈ 0.3 → max gate ≈ 1.8 initially.
    Both surp signals ∈ (0,1) → bounded output.

GLOBAL EMA DECAY vs LOCAL EMA DECAY:
    Global (cross-batch): slow decay needed to accumulate training history.
        logit_decay_g → d_g ≈ 0.9 (window ~10 batches) — learned.
    
    Local (within-sequence): decay controls "memory length" within sequence.
        logit_decay_l → d_l ≈ 0.9 (window ~10 positions in T=64) — learned.

Params: logit_decay_g, logit_decay_l, log_sigma_g, log_sigma_l, log_alpha_g, log_alpha_l = 6 scalars.
State: _ema_mean_g (D,), _ema_sq_g (D,), _ready_g (bool).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU129(nn.Module):
    """Dual-reference novelty: cross-batch per-channel z-score + within-sequence causal EMA z-score."""

    def __init__(self, d_ff: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        # Global (cross-batch) EMA parameters
        self.logit_decay_g = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
        self.log_sigma_g = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_alpha_g = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ≈ 0.5

        # Local (within-sequence scan) EMA parameters
        self.logit_decay_l = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))  # same init, learned separately
        )
        self.log_sigma_l = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_alpha_l = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # ≈ 0.3

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

        d_g     = torch.sigmoid(self.logit_decay_g).detach().item()
        sigma_g = F.softplus(self.log_sigma_g)
        alpha_g = F.softplus(self.log_alpha_g)

        d_l     = torch.sigmoid(self.logit_decay_l).detach().item()
        sigma_l = F.softplus(self.log_sigma_l)
        alpha_l = F.softplus(self.log_alpha_l)

        # ── GLOBAL SIGNAL: cross-batch per-channel z-score ───────────
        # Batch+sequence mean for EMA update
        x_flat    = x.detach().flatten(0, -2)   # (B*T, D)
        x_mean    = x_flat.mean(0)              # (D,)
        x_sq_mean = x_flat.pow(2).mean(0)       # (D,)

        if not self._ready_g:
            self._ema_mean_g = x_mean.clone()
            self._ema_sq_g   = x_sq_mean.clone()
            self._ready_g    = True
            # Skip global gate on first call (no history)
            surp_global = torch.zeros(B, T, D, device=x.device)
        else:
            # Per-channel global z-score
            var_g    = (self._ema_sq_g - self._ema_mean_g.pow(2)).clamp(min=self.eps_var)
            std_g    = var_g.sqrt()
            z_g      = (x - self._ema_mean_g) / (std_g + self.eps)   # (B, T, D) — gradient ok
            surp_global = torch.tanh(sigma_g * z_g.abs())             # (B, T, D) ∈ (0,1)

        # Update global EMA AFTER computing gate
        self._ema_mean_g = d_g * self._ema_mean_g + (1.0 - d_g) * x_mean
        self._ema_sq_g   = d_g * self._ema_sq_g   + (1.0 - d_g) * x_sq_mean

        # ── LOCAL SIGNAL: within-sequence causal EMA scan ─────────────
        ema_mean_l = torch.zeros(B, D, device=x.device)
        ema_sq_l   = torch.zeros(B, D, device=x.device)
        surp_local_list = []

        for t in range(T):
            xt = x[:, t, :].detach()  # (B, D)
            if t == 0:
                surp_local_list.append(torch.zeros(B, D, device=x.device))
                ema_mean_l = xt.clone()
                ema_sq_l   = xt.pow(2)
            else:
                var_l    = (ema_sq_l - ema_mean_l.pow(2)).clamp(min=self.eps_var)
                std_l    = var_l.sqrt()
                z_l      = (xt - ema_mean_l) / (std_l + self.eps)     # (B, D)
                surp_local_list.append(torch.tanh(sigma_l * z_l.abs()))
                ema_mean_l = d_l * ema_mean_l + (1.0 - d_l) * xt
                ema_sq_l   = d_l * ema_sq_l   + (1.0 - d_l) * xt.pow(2)

        surp_local = torch.stack(surp_local_list, dim=1)               # (B, T, D) ∈ (0,1)

        # ── Combined additive gate ────────────────────────────────────
        gate = 1.0 + alpha_g * surp_global + alpha_l * surp_local      # (B, T, D)

        return self._gelu(x * gate)
