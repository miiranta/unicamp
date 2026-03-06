"""GELU229 – Ring Buffer with Hard Sigmoid Suppression Gate.

MOTIVATION — SHARPER FAMILIARITY DETECTION THAN gelu54:
    gelu54 uses: gate = (1-α) + α × exp(-τ × cos_sim)
    For cos_sim ∈ [0, 1]:
        cos_sim = 0.0 (orthogonal/novel):     gate = (1-α) + α × 1.0 = 1.0
        cos_sim = 0.5 (moderately familiar):  gate = (1-α) + α × exp(-τ/2) ≈ 0.72 (τ=2)
        cos_sim = 0.9 (very familiar):        gate = (1-α) + α × exp(-0.9τ) ≈ 0.51 (α=0.3)

    The transition from "novel" to "familiar" is SOFT — gate decreases smoothly.
    This means on pass 2 (high cos_sim ≈ 0.9), the gate is still quite high (0.51).

    gelu229 uses a SIGMOID with learnable sharpness and threshold:
        gate_t = σ(-sharpness × (cos_sim - θ))        ← sigmoid centered at threshold θ
        gate   = gate_min + (gate_max - gate_min) × gate_t

    For sharpness >> 1:
        cos_sim < θ (novel):     gate_t ≈ 1 → gate ≈ gate_max (≈1.0)
        cos_sim ≈ θ (boundary):  gate_t = 0.5 → gate ≈ (gate_min + gate_max)/2
        cos_sim > θ (familiar):  gate_t ≈ 0 → gate ≈ gate_min (≈0.1)

    ADAPTATION ADVANTAGE:
        With sharpness=10 and θ=0.7:
        Pass 1 (test novel vs training buffer, cos≈0.3): gate ≈ gate_max ≈ 1.0
        Pass 2 (test familiar vs test buffer,   cos≈0.85): gate ≈ gate_min ≈ 0.1

        The CONTRAST between pass 1 and pass 2 is dramatically larger than gelu54's smooth gate.
        This should give MUCH stronger positive Δ if the model can handle near-zero gate.

    CAUTION: gate_min is learned; if it goes too low, training may become unstable.
             We enforce gate_min ∈ [0.05, 0.5] and gate_max = 1 + small_buffer.

PARAMS: log_sharpness, logit_threshold, logit_gate_min = 3 scalars
STATE:  ring buffer (N, D), mask (N,), pointer (int)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU229(nn.Module):
    """Hard sigmoid suppression ring buffer gate: sharp novel/familiar transition."""

    def __init__(self, buffer_size: int = 32):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None   # (N, D) normalized episode means
        self._mask: torch.Tensor = None   # (N,) bool
        self._ptr   = 0
        self._ready = False

        # Sharpness: init at 5 (moderately sharp), can grow sharper
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(5.0)))
        # Threshold: cos_sim at which gate transitions (init 0.6)
        self.logit_threshold = nn.Parameter(torch.tensor(math.log(0.6 / 0.4)))  # ≈0.6
        # Minimum gate value when very familiar (init 0.2, constrained to [0.05, 0.5])
        self.logit_gate_min  = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))  # ≈0.2

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
        sharpness = self.log_sharpness.exp().clamp(1.0, 20.0)
        threshold = torch.sigmoid(self.logit_threshold)                  # ∈ (0,1)
        gate_min  = 0.05 + 0.45 * torch.sigmoid(self.logit_gate_min)    # ∈ (0.05, 0.5)

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

        # ── Find nearest buffer episode ───────────────────────────────
        m_norm = F.normalize(m_curr.unsqueeze(0), dim=-1)         # (1, D)
        buf_n  = F.normalize(self._buf, dim=-1)                   # (N, D)
        sims   = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
        nearest_idx = sims.argmax()

        # Per-token cosine similarity to nearest episode
        y_flat  = y.flatten(0, 1).detach()                        # (B*T, D)
        y_n     = F.normalize(y_flat, dim=-1)                     # (B*T, D)
        ep_n    = buf_n[nearest_idx].unsqueeze(0)                 # (1, D)
        tok_sim = (y_n * ep_n).sum(-1)                            # (B*T,)

        # ── Hard sigmoid gate ─────────────────────────────────────────
        # sigmoid(-sharpness × (cos - θ)):
        #   cos < θ → sigmoid(positive) → ≈1 (novel → gate_max=1)
        #   cos > θ → sigmoid(negative) → ≈0 (familiar → gate_min)
        gate_t = torch.sigmoid(-sharpness * (tok_sim - threshold))  # (B*T,) ∈ (0,1)
        # Scale from gate_min to 1.0
        gate_s = gate_min + (1.0 - gate_min) * gate_t               # ∈ [gate_min, 1.0]
        gate   = gate_s.view(B, T, 1)                               # (B, T, 1)

        output = y * gate

        # ── Update ring buffer ────────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
