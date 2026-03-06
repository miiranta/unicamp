"""GELU230 – Ring Buffer with Familiarity Floor (Double-Suppression for Known Content).

MOTIVATION — MAXIMIZING ADAPTATION BY EXTREME SUPPRESSION:
    gelu54 suppresses familiar content: gate = (1-α) + α × exp(-τ × cos_sim)
    For α=0.3, τ=2: minimum gate ≈ 0.7 even for perfectly familiar content (cos=1).

    What if we push the gate lower for VERY familiar content?
    The model may be able to handle near-zero FFN contribution if the attention
    layers have learned to be self-sufficient.

    gelu230 adds a FAMILIARITY FLOOR — applied only to content above a high-similarity threshold:
        gate_base = (1-α) + α × exp(-τ × cos_sim)      ← normal gelu54 gate
        very_familiar = cos_sim > θ_high                 ← boolean mask
        gate_floor = clamp(gate_base, floor, 1.0)        ← clamp DOWN to floor when very familiar
                   = min(gate_base, f + (1-f) × ...) ...

    Actually: implement as a MIXTURE:
        gate_gelu54 = (1-α) + α × exp(-τ × cos_sim)
        gate_floor  = sigmoid(-β × (cos_sim - θ_high))   ← ≈ 1 when cos < θ, ≈ 0 when cos > θ
        gate        = gate_gelu54 × gate_floor           ← product: floor factor applied on top

    When cos_sim > θ_high (very familiar, i.e., pass 2):
        gate_floor ≈ 0 → gate ≈ 0 → extreme suppression
    When cos_sim < θ_high (less familiar, i.e., pass 1 with training buffer):
        gate_floor ≈ 1 → gate = gate_gelu54 → normal gelu54 behavior

    Since training content is at most moderately familiar (cos ≈ 0.3-0.7), gate_floor ≈ 1
    and the gate behaves like gelu54. The model calibrates to gelu54-like behavior.

    On test pass 2: test content has cos ≈ 0.85+ → gate_floor → 0 → extreme suppression.
    This should give much stronger Δ than gelu54 if the model can work with near-zero FFN.

PARAMS: log_tau, log_blend (α), log_beta_floor (sharpness of floor cutoff), logit_theta_high
STATE:  ring buffer (N, D), mask (N,), pointer (int)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU230(nn.Module):
    """Ring buffer with double-suppression floor: familiar content gets near-zero gate on pass 2."""

    def __init__(self, buffer_size: int = 32):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend      = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α ≈ 0.3
        # Floor gate: sigmoid(-β_floor × (cos - θ_high))
        self.log_beta_floor = nn.Parameter(torch.tensor(math.log(8.0)))         # sharpness
        self.logit_theta_hi = nn.Parameter(torch.tensor(math.log(0.75 / 0.25)))  # θ_high ≈ 0.75

    def reset_state(self):
        self._buf   = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        tau       = self.log_tau.exp()
        alpha     = torch.sigmoid(self.log_blend)
        beta_fl   = self.log_beta_floor.exp().clamp(1.0, 20.0)
        theta_hi  = torch.sigmoid(self.logit_theta_hi)           # ∈ (0,1)

        y = self._gelu(x)   # (B, T, D)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Find nearest episode ──────────────────────────────────────
        m_norm = F.normalize(m_curr.unsqueeze(0), dim=-1)         # (1, D)
        buf_n  = F.normalize(self._buf, dim=-1)                   # (N, D)
        sims   = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
        nearest_idx = sims.argmax()
        ep_n   = buf_n[nearest_idx].unsqueeze(0)                  # (1, D)

        # Per-token cosine to nearest episode
        y_flat = y.flatten(0, 1).detach()                         # (B*T, D)
        y_n    = F.normalize(y_flat, dim=-1)
        tok_sim= (y_n * ep_n).sum(-1)                             # (B*T,)

        # ── Base gate (gelu54 smooth gate) ────────────────────────────
        gate_base = (1.0 - alpha) + alpha * torch.exp(-tau * tok_sim)   # (B*T,)

        # ── Floor gate (extra suppression for very-familiar content) ─
        # sigmoid(-β × (cos - θ_hi)):
        #   cos << θ_hi (novel/moderate): floor_gate ≈ 1.0  → no extra effect
        #   cos >> θ_hi (very familiar):  floor_gate ≈ 0.0  → kill the gate
        floor_gate = torch.sigmoid(-beta_fl * (tok_sim - theta_hi))     # (B*T,)

        # ── Combined gate ─────────────────────────────────────────────
        gate = (gate_base * floor_gate).clamp(0.02, 1.05).view(B, T, 1)

        output = y * gate

        # ── Update ring buffer ────────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
