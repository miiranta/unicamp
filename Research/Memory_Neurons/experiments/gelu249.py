"""GELU249 – Detection-based Frozen Buffer + PRE-FIRE Facilitation.

═══════════════════════════════════════════════════════════════════════════
ROOT CAUSE FIX FOR TINY Δ IN gelu239-248
═══════════════════════════════════════════════════════════════════════════

WHY gelu239-248 FAIL TO GIVE MONOTONIC IMPROVEMENT:

  In gelu239, the depletion fires AFTER gate is computed:
    1. Find nearest slot  (depl=1.0 for a fresh pass-1 slot)
    2. gate = exp(-k*(1-1.0)) = 1.0   ← NO EFFECT on pass 2
    3. Apply gate
    4. THEN: depl *= 0.5  (fires after the gate was already applied)

  → Pass 2 gate = 1.0 (no adaption at all!)
  → Pass 3 gate = exp(-k*0.5) ≈ 0.37 (suppression, HURTS PPL)

  Result: Δ1→2 ≈ 0, Δ1→3 < 0 (model gets WORSE on pass 3).

═══════════════════════════════════════════════════════════════════════════
FIX 1 – DETECTION-BASED BUFFER FREEZE
═══════════════════════════════════════════════════════════════════════════

  When pass 2 starts, the first batch finds its pass-1 counterpart with
  cos_sim ≈ 1.0 > FIRE_THRESH. At this point we KNOW pass 1 is complete.

  Solution:
    • While _pass1_complete=False: write slots normally, gate=1.0.
    • When FIRST high-sim match detected: set _pass1_complete=True.
    • From then on: STOP writing new slots (buffer is frozen with pass-1 data).
    • All pass-2 and pass-3 lookups hit pass-1 entries exclusively.

═══════════════════════════════════════════════════════════════════════════
FIX 2 – PRE-FIRE STATE UPDATE
═══════════════════════════════════════════════════════════════════════════

  Fire state update BEFORE computing gate (not after):
    1. Find nearest slot
    2. PRE-FIRE: _facil[nearest] *= FACIL_RATE  (e.g., ×2.0)
    3. gate = 1 + k * (_facil[nearest] - 1.0)  ← uses UPDATED facil
    4. Apply gate

  → Pass 2: facil=1.0→2.0, gate = 1+k  (moderate boost)
  → Pass 3: facil=2.0→4.0, gate = 1+3k (strong boost)
  → Pass 4: facil=4.0→8.0, gate = 1+7k (very strong boost)

  This gives MONOTONICALLY INCREASING facilitation across passes.

═══════════════════════════════════════════════════════════════════════════
FIX 3 – FACILITATION INSTEAD OF DEPLETION
═══════════════════════════════════════════════════════════════════════════

  Depletion (gate<1) SUPPRESSES familiar patterns → hurts PPL.
  Facilitation (gate>1) BOOSTS familiar patterns → potentially helps PPL.

  Biological basis: SHORT-TERM SYNAPTIC FACILITATION (STF).
  In STF, repeated activation of a synapse INCREASES its response due to
  calcium accumulation. Neurons that have already processed a pattern
  fire MORE strongly when they encounter it again (not less).

  In the LM context: familiar text patterns should receive STRONGER
  activation, giving the model more signal about expected structure.

GUARANTEE Δ1→2 < Δ1→3:
  Since facilitation level strictly increases each pass, the gate strictly
  increases each pass. If facilitation helps PPL, then:
    ppl_2 < ppl_1         (some improvement at pass 2)
    ppl_3 < ppl_2 < ppl_1 (more improvement at pass 3)
  → Δ1→3 = ppl_1-ppl_3 > ppl_1-ppl_2 = Δ1→2 > 0  ✓

PASS-1 GUARANTEE: buffer building, no fires below FIRE_THRESH (0.85).
  gate=1.0 throughout pass 1 → zero PPL change on pass 1 ✓

PARAMS: log_k_fac (gate strength, init k=0.5, range [0.01, 5.0])
STATE:  _buf (N,D), _facil (N,) starts 1.0, _mask (N,), _ptr int,
        _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH  = 0.85
FACIL_RATE   = 2.0   # facil multiplied by this on each hit (×2 per pass)
MAX_GATE     = 8.0   # safety cap on gate value


class GELU249(nn.Module):
    """Detection-based frozen buffer + pre-fire facilitation gate > 1."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        # Gate = 1 + k*(facil-1). k=0.5 → pass2: gate=1.5, pass3: gate=2.5
        self.log_k_fac = nn.Parameter(torch.tensor(math.log(0.5)))

        self._buf:  torch.Tensor = None  # (N, D)
        self._facil: torch.Tensor = None  # (N,) facilitation level, starts 1.0
        self._mask: torch.Tensor = None  # (N,) bool
        self._ptr  = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf    = None
        self._facil  = None
        self._mask   = None
        self._ptr    = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_fac = self.log_k_fac.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask  = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                # Check if any existing slot fires (→ entering pass 2)
                if self._mask.any():
                    m_n    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    buf_n  = F.normalize(self._buf, dim=-1)
                    sims   = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    max_sim = sims.max().item()
                    if max_sim > FIRE_THRESH:
                        self._pass1_complete = True  # Buffer frozen!
                        # Fall through to PASS-2+ phase below
                    else:
                        # Still in pass 1: write slot, gate=1.0
                        self._buf[self._ptr]   = F.normalize(m_curr, dim=0)
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    # Very first batch
                    self._buf[0]   = F.normalize(m_curr, dim=0)
                    self._facil[0] = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y

        # ── PASS-2+ PHASE: frozen buffer, pre-fire facilitation ────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if max_sim > FIRE_THRESH:
                # PRE-FIRE: update facilitation BEFORE computing gate
                self._facil[nearest_idx] *= FACIL_RATE

            facil_level = self._facil[nearest_idx].item()

        # gate = 1 + k * (facil - 1)
        # facil=1.0 → gate=1.0, facil=2.0 → gate=1+k, facil=4.0 → gate=1+3k
        gate = min(1.0 + k_fac.item() * (facil_level - 1.0), MAX_GATE)
        return y * gate
