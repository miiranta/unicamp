"""GELU252 – Detection-based Frozen Buffer + PRE-FIRE Soft Facilitation Cap.

VARIANT OF gelu249 with a LEARNABLE CEILING on the gate value.

Problem with gelu249: gate = 1 + k*(facil-1) grows unboundedly as facil doubles
each pass. After 5 passes: facil=32, gate=1+31k. For k=0.5: gate=16.5. Too large.

FIX: Use a soft ceiling via tanh or sigmoid:
    gate = 1 + k * tanh((facil - 1) * softness)

    facil=1.0: gate = 1 + k*tanh(0) = 1.0  ← pass 1 exactly
    facil=2.0: gate = 1 + k*tanh(softness) ← first fire, moderate boost
    facil=4.0: gate = 1 + k*tanh(3*softness) ← second fire, larger boost
    facil→∞:  gate → 1 + k           ← asymptotes at 1+k (controlled max)

MONOTONICITY: tanh is monotonically increasing, so gate increases each pass.
  Δ1→2 < Δ1→3 guaranteed if facilitation helps PPL.

ADDITIONAL FEATURE: learnable 'softness' controls how quickly the ceiling is reached.
  - softness=0.1: slow approach to ceiling, very gradual across many passes
  - softness=1.0: reaches ~95% of ceiling after facil=4.0

PARAMS:
    log_k       (max boost amplitude, init 1.0 → 1+k=2.0 ceiling)
    log_softness (rate of approach to ceiling, init 0.7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0


class GELU252(nn.Module):
    """Soft-ceiling facilitation: gate = 1 + k*tanh((facil-1)*softness)."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k        = nn.Parameter(torch.tensor(math.log(1.0)))    # max boost
        self.log_softness = nn.Parameter(torch.tensor(math.log(0.7)))    # approach rate

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k        = self.log_k.exp().clamp(0.01, 5.0)
        softness = self.log_softness.exp().clamp(0.01, 3.0)

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

        # gate = 1 + k * tanh((facil-1) * softness)
        gate = 1.0 + k.item() * math.tanh((facil_level - 1.0) * softness.item())
        return y * gate
