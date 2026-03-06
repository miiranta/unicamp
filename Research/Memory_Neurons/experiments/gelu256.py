"""GELU256 – Detection-based Frozen Buffer + PRE-FIRE Facilitation (Small k, Gentle).

GENTLE VARIANT of gelu249 for stability comparison: k=0.1 instead of 0.5.

    gate = 1 + k * (facil - 1)  with k=0.1

    facil=1.0 (pass 1): gate = 1.0
    facil=2.0 (pass 2): gate = 1.1  (10% boost — very gentle)
    facil=4.0 (pass 3): gate = 1.3  (30% boost)

MOTIVATION:
    gelu249 (k=0.5) gives 50% boost at pass 2, 250% at pass 3 (capped to 8x).
    gelu255 (k=2.0) gives 300% boost at pass 2 (immediately hits ceiling).
    gelu256 (k=0.1) gives 10% boost at pass 2, 30% at pass 3.

    The gentle variant tests whether even SMALL facilitation leads to
    measurable, monotonic PPL improvement without disrupting the model's
    learned distribution too severely.

    Insight from existing experiments: gelu211's EMA gate changes gelu_output
    by ~10-20% and achieves PPL 159. If facilitation can provide a similar-scale
    boost in the right direction, it might give comparable improvement.

PARAMS: log_k (small init, k=0.1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0


class GELU256(nn.Module):
    """Gentle facilitation: k=0.1 init, 10%/30% boost at pass 2/3."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k = nn.Parameter(torch.tensor(math.log(0.1)))   # k=0.1 gentle

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
