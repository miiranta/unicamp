"""GELU257 – Detection-based Frozen Buffer + PRE-FIRE Facilitation + Sigmoid Transition.

SMOOTH PROGRESSIVE FACILITATION that only kicks in AFTER a delay.

Problem with gelu249: at pass 2 (facil=2.0 after 1 fire), gate=1+k immediately.
But we want Δ1→2 SMALL and Δ1→3 LARGE (not equal facilitation at both passes).

Solution: make the gate a sigmoid function of (facil-1) such that:
    - At facil=2.0 (pass 2, 1 fire): gate is small  → Δ1→2 ≈ 0
    - At facil=4.0 (pass 3, 2 fires): gate is large → Δ1→3 > Δ1→2

GATE FORMULA:
    gate = 1 + k * sigmoid(sharpness * (facil - threshold))

    threshold=3.5 → only kicks in significantly when facil > 3.5
    With FACIL_RATE=2.0: facil=1→2→4→8. threshold=3.5 is between 2 and 4.
    
    pass 1: facil=1.0 → gate = 1 + k*σ(sharpness*(1-3.5)) ≈ 1 + k*~0  ≈ 1.0
    pass 2: facil=2.0 → gate = 1 + k*σ(sharpness*(2-3.5)) 
                              = 1 + k*σ(-1.5*sharpness) ← small
    pass 3: facil=4.0 → gate = 1 + k*σ(sharpness*(4-3.5))
                              = 1 + k*σ(+0.5*sharpness) ← moderate to large
    pass 4: facil=8.0 → gate = 1 + k*σ(sharpness*(8-3.5)) ← near maximum

EXAMPLE with k=2.0, sharpness=2.0, threshold=3.5:
    pass 1: gate = 1 + 2*σ(-5.0) ≈ 1 + 2*0.007 ≈ 1.01  (tiny)
    pass 2: gate = 1 + 2*σ(-3.0) ≈ 1 + 2*0.047 ≈ 1.09  (small)
    pass 3: gate = 1 + 2*σ(+1.0) ≈ 1 + 2*0.731 ≈ 2.46  (large!)

Δ1→2 ≈ small, Δ1→3 ≈ much larger. This is EXACTLY what the user wants.

PARAMS:
    log_k (boost amplitude)
    log_sharpness (sigmoid steepness)
    log_threshold (where sigmoid crossover happens in facil space)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0


class GELU257(nn.Module):
    """Sigmoid-transition facilitation: small Δ1→2, large Δ1→3 by design."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k         = nn.Parameter(torch.tensor(math.log(2.0)))    # amplitude
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(2.0)))    # steepness
        self.log_threshold = nn.Parameter(torch.tensor(math.log(3.5)))    # crossover

        self._buf:  torch.Tensor = None
        self._facil: torch.Tensor = None
        self._mask: torch.Tensor = None
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

    def _gate_from_facil(self, facil: float, k: float, sharpness: float, threshold: float) -> float:
        """Sigmoid-gated facilitation. gate→1.0 for facil≤threshold, →1+k for facil>>threshold."""
        import math
        sig = 1.0 / (1.0 + math.exp(-sharpness * (facil - threshold)))
        return 1.0 + k * sig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k         = self.log_k.exp().clamp(0.01, 5.0)
        sharpness = self.log_sharpness.exp().clamp(0.1, 10.0)
        threshold = self.log_threshold.exp().clamp(1.1, 20.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        if self._buf is None:
            with torch.no_grad():
                self._buf   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask  = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
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
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]   = F.normalize(m_curr, dim=0)
                    self._facil[0] = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y

        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            if max_sim > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE
            facil_level = self._facil[nearest_idx].item()

        gate = self._gate_from_facil(facil_level, k.item(), sharpness.item(), threshold.item())
        return y * gate
