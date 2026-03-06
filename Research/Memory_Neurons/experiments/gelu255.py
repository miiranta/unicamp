"""GELU255 – Detection-based Frozen Buffer + PRE-FIRE Facilitation (Large k, High Ceiling).

VARIANT OF gelu249 with STRONGER facilitation: k=2.0 instead of 0.5.

    gate = 1 + k * (facil - 1)  with k=2.0

    facil=1.0 (pass 1): gate = 1.0
    facil=2.0 (pass 2): gate = 1 + 2.0*(2.0-1.0) = 3.0  ← TRIPLE output
    facil=4.0 (pass 3): gate = 1 + 2.0*3.0 = 7.0         ← 7x output

    Cap at MAX_GATE=3.0 to prevent numerical explosion:
    gate = min(3.0, 1 + k*(facil-1))

WHY TEST LARGE FACILITATION:
    gelu249 uses k=0.5 (init) → pass-2 gate=1.5 (50% boost).
    This may be too conservative to see meaningful PPL improvement.
    gelu255 uses k=2.0 (init) → pass-2 gate=3.0 (200% boost, capped).
    If the model benefits from stronger activation for familiar contexts,
    this experiment will show it.

RISK: The model was trained without such extreme gating. A 3x activation
boost might collapse the model's predictions (all probabilities become
very peaked). If this happens, PPL will be catastrophically bad.

But that's useful information: it tells us the SENSITIVITY of the model
to facilitation strength.

PARAMS: log_k (larger init), gate capped at 3.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 3.0


class GELU255(nn.Module):
    """Strong facilitation: gate capped at 3.0, tests sensitivity to boost magnitude."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k = nn.Parameter(torch.tensor(math.log(2.0)))   # k=2.0 (large)

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
        k = self.log_k.exp().clamp(0.01, 5.0)

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

        gate = min(1.0 + k.item() * (facil_level - 1.0), MAX_GATE)
        return y * gate
