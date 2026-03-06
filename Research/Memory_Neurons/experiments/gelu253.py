"""GELU253 – Detection-based Buffer + PRE-FIRE Depletion (accelerating suppression).

CONTROL EXPERIMENT: What if we PRE-FIRE depletion (gate < 1)?

With DEPLETION + PRE-FIRE:
    Pass 2: depl=1.0→0.5 (pre-fire), gate=exp(-k*0.5) ← SUPPRESSION on pass 2
    Pass 3: depl=0.5→0.25 (pre-fire), gate=exp(-k*0.75) ← MORE SUPPRESSION on pass 3

So: gate monotonically DECREASES (more suppression each pass).

Expected: ppl_2 > ppl_1 and ppl_3 > ppl_2 → both deltas negative (model gets worse).
  Δ1→2 < 0, Δ1→3 < Δ1→2 (both negative, and pass 3 even worse).

This is the OPPOSITE of what the user wants.

WHY INCLUDE IT: As a counterpoint to gelu249 (facilitation).
By comparing gelu249 (facilitation+pre-fire) and gelu253 (depletion+pre-fire)
with the same buffer frozen architecture, we can ISOLATE the effect of
facilitation vs depletion.

If gelu249 shows Δ > 0 and gelu253 shows Δ < 0 with similar magnitudes, it
confirms that FACILITATION is the correct direction.
If both show Δ < 0, then the effect is dominated by something else (e.g.,
the frozen buffer change itself causing worse PPL due to stale representations).

PARAMS: log_k (depletion sharpness, init k=2.0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH   = 0.85
DEPL_RATE     = 0.5


class GELU253(nn.Module):
    """Pre-fire depletion: gate<1 applied from pass 2, stronger at pass 3."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k = nn.Parameter(torch.tensor(math.log(2.0)))

        self._buf:  torch.Tensor = None
        self._depl: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr  = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf    = None
        self._depl   = None
        self._mask   = None
        self._ptr    = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_gate = self.log_k.exp().clamp(0.1, 8.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        if self._buf is None:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._depl = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n  = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (F.normalize(self._buf, dim=-1) * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]   = F.normalize(m_curr, dim=0)
                        self._depl[self._ptr]  = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]   = F.normalize(m_curr, dim=0)
                    self._depl[0]  = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y

        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            # PRE-FIRE depletion: deplete BEFORE gate computation
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= DEPL_RATE
            depl_level = self._depl[nearest_idx].item()

        # gate = exp(-k * (1 - depl)) < 1 for depleted slots
        gate = math.exp(-k_gate.item() * (1.0 - depl_level))
        return y * gate
