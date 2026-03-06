"""GELU271 – Hit-Count Saturation Gate.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: gelu249 uses EXPONENTIAL facilitation  (×2 each pass →
1, 2, 4, 8, ...)  which can over-boost with many passes.
This experiment uses CUMULATIVE HIT COUNT with tanh saturation:

    gate = 1 + k * tanh(hit_count / saturation)

This gives a SMOOTH, BOUNDED facilitation that automatically saturates.
═══════════════════════════════════════════════════════════════════════════

MOTIVATION (from neuroscience):
    Short-Term Facilitation (STF) in synapses saturates — the first few
    activations cause large increases in synaptic strength, but subsequent
    activations have diminishing returns. Mathematically, STF is often
    modeled as a saturating function of presynaptic calcium concentration.
    tanh(count / sat) is the simplest saturating model.

GATE FORMULA:
    hit_count[slot] = number of times slot has been retrieved (int)

    gate = 1 + k * tanh(hit_count / saturation)

    hit_count=0  → gate = 1.0                         (pass 1, exact zero)
    hit_count=1  → gate = 1 + k * tanh(1/sat)         (pass 2)
    hit_count=2  → gate = 1 + k * tanh(2/sat)         (pass 3, larger)
    hit_count=∞  → gate → 1 + k                       (asymptote, safe)

    With sat=1.0, k=1.0:
        pass2: gate = 1 + tanh(1.0) ≈ 1.76
        pass3: gate = 1 + tanh(2.0) ≈ 1.96
        pass4: gate = 1 + tanh(3.0) ≈ 2.00 (saturated)

    Δ1→3 > Δ1→2 ✓ (tanh is concave → largest jump between 0 and 1)

COMPARE TO PRIOR APPROACHES:
    gelu249 (exponential): facil=1,2,4 → gate=1, 1.5, 2.5  (grows without bound)
    gelu252 (soft ceiling): gate=1+k*tanh((facil-1)*soft) → uses facil not count
    gelu271 (this):        gate=1+k*tanh(count/sat)       → directly counts passes

CRITICAL DIFFERENCE:
    The hit_count accumulates GLOBALLY per slot. After 3 passes:
        Every slot has been hit at most ~2-3 times.
        gate ≈ 1 + k*tanh(3/sat) = bounded regardless of k.
    No unbounded growth. Numerically stable.

PARAMS: log_k_gate (asymptote above 1, init k=1.0),
        log_sat    (saturation count, init sat=1.0)
STATE:  _buf (N,D) normalized keys,  _hit_count (N,) int,
        _mask (N,) bool,  _ptr int,  _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
N_BUF       = 512
MAX_GATE    = 8.0


class GELU271(nn.Module):
    """Hit-count saturation gate: gate = 1 + k * tanh(count/sat)."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N = buffer_size

        self.log_k_gate = nn.Parameter(torch.tensor(math.log(1.0)))    # k=1.0
        self.log_sat    = nn.Parameter(torch.tensor(math.log(1.0)))    # sat=1.0

        self._buf:       torch.Tensor = None
        self._hit_count: torch.Tensor = None
        self._mask:      torch.Tensor = None
        self._ptr        = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf        = None
        self._hit_count  = None
        self._mask       = None
        self._ptr        = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_gate = self.log_k_gate.exp().clamp(0.01, 5.0)
        sat    = self.log_sat.exp().clamp(0.1, 10.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf       = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hit_count = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (self._buf * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]       = F.normalize(m_curr, dim=0)
                        self._hit_count[self._ptr] = 0
                        self._mask[self._ptr]      = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]       = F.normalize(m_curr, dim=0)
                    self._hit_count[0] = 0
                    self._mask[0]      = True
                    self._ptr          = 1
                    return y

        # ── Pass-2+ hit-count saturation gate ─────────────────────────
        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (self._buf * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()

            if sims[nearest_idx].item() > FIRE_THRESH:
                # PRE-FIRE: increment hit count BEFORE computing gate
                self._hit_count[nearest_idx] += 1

            count = self._hit_count[nearest_idx].item()

        # gate = 1 + k * tanh(count / sat)
        gate = min(1.0 + k_gate.item() * math.tanh(count / sat.item()), MAX_GATE)
        return y * gate
