"""GELU266 – Hysteresis LTP Gate (Long-Term Potentiation Analog).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Biological LTP (Long-Term Potentiation) is characterized
by a THRESHOLD CROSSING mechanism: once a synapse exceeds a potentiation
threshold, it "locks in" at a high state and decays only slowly.
This experiment models that sharp threshold-crossing behavior with hysteresis.

Prior eperiments have CONTINUOUS facilitation (facil × 2 each pass).
This experiment is BINARY: each slot is either UNPOTENTIATED or POTENTIATED.
Once potentiated, it stays potentiated until reset.
═══════════════════════════════════════════════════════════════════════════

MECHANISM:
    _potentiated[slot]: bool — has this slot undergone LTP?
    _hit_count[slot]:   int  — how many times has this slot been retrieved?

    On retrieval in pass 2+:
        hit_count[nearest] += 1
        if hit_count[nearest] >= LTP_THRESHOLD:
            potentiated[nearest] = True

    Gate:
        if potentiated[nearest]:
            gate = G_HIGH  (e.g., 2.5 — the "potentiated" state)
        else:
            gate = 1 + k * (hit_count / LTP_THRESHOLD) * sim  (rising ramp)

WHY HYSTERESIS?
    Standard facilitation grows exponentially (facil × 2 each pass).
    With 3 passes, facil = 1, 2, 4 → gate = 1, 1.5, 2.5 (with k=0.5).
    This is smooth but may be too gradual.

    Hysteresis creates a SHARP TRANSITION:
        Pass 2: hit_count=1/2 of threshold → gate = 1 + k*(0.5)*sim (moderate)
        Pass 3: hit_count=2/2 = threshold  → SNAP to G_HIGH (e.g., 2.5)
    This guarantees: pass-3 gate >> pass-2 gate → Δ1→3 >> Δ1→2 ✓

    Additionally, once potentiated, the strong gate persists on ALL subsequent
    passes (not just the next one), which could help multi-pass scenarios.

BIOLOGICALLY: Ca²⁺ accumulation at synapse → exceeds threshold → AMPA
    receptor trafficking → persistent synaptic strengthening.

LTP_THRESHOLD = 2 (requires 2 hits — fires in pass 2, locks in pass 3).
    With 3 eval passes, this means:
    Pass 2: 1st hit  → ramp (gate moderate)
    Pass 3: 2nd hit  → fully potentiated (gate high)
    → Δ1→3 > Δ1→2 ✓

PARAMS: log_k_ramp (ramp gate strength), log_g_high (potentiated gate)
STATE:  _buf (N,D), _hit_count (N,) int, _potentiated (N,) bool,
        _mask, _ptr, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH    = 0.85
LTP_THRESHOLD  = 2    # hits required to trigger full potentiation


class GELU266(nn.Module):
    """Hysteresis LTP gate: binary potentiation with threshold crossing."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        # Ramp gate: 1 + k_ramp*(hit_frac)*sim while approaching threshold
        self.log_k_ramp  = nn.Parameter(torch.tensor(math.log(1.0)))
        # Potentiated state gate (fully locked-in boost)
        self.log_g_high  = nn.Parameter(torch.tensor(math.log(1.5)))   # g_high = 2.5 init

        self._buf:         torch.Tensor = None
        self._hit_count:   torch.Tensor = None   # (N,) int → stored as float
        self._potentiated: torch.Tensor = None   # (N,) bool
        self._mask:        torch.Tensor = None
        self._ptr          = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf          = None
        self._hit_count    = None
        self._potentiated  = None
        self._mask         = None
        self._ptr          = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_ramp = self.log_k_ramp.exp().clamp(0.01, 5.0)
        g_high = 1.0 + self.log_g_high.exp().clamp(0.1, 5.0)   # gate > 1 always

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf         = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hit_count   = torch.zeros(self._N, device=x.device, dtype=y.dtype)
                self._potentiated = torch.zeros(self._N, device=x.device, dtype=torch.bool)
                self._mask        = torch.zeros(self._N, device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]  = F.normalize(y_mean, dim=0)
                        self._mask[self._ptr] = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]  = F.normalize(y_mean, dim=0)
                    self._mask[0] = True
                    self._ptr = 1
                    return y
            if not self._pass1_complete:
                return y

        # ── PASS-2+ PHASE ─────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            sim_val     = sims[nearest_idx].clamp(0.0, 1.0).item()

            if sim_val > FIRE_THRESH:
                self._hit_count[nearest_idx] += 1
                if self._hit_count[nearest_idx] >= LTP_THRESHOLD:
                    self._potentiated[nearest_idx] = True

            is_potentiated = self._potentiated[nearest_idx].item()
            hit_frac       = min(self._hit_count[nearest_idx].item() / LTP_THRESHOLD, 1.0)

        if is_potentiated:
            gate = g_high.item()
        else:
            gate = 1.0 + k_ramp.item() * hit_frac * sim_val

        return y * gate
