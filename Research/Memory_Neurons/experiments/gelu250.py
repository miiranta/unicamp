"""GELU250 – Detection-based Frozen Buffer + PRE-FIRE Per-Token Facilitation.

Same detection + frozen buffer architecture as gelu249, but applies facilitation
PER TOKEN based on cosine similarity between the individual token activation
and the stored nearest slot mean.

GATE FORMULA:
    tok_sim(b,t) = cos(gelu(x)[b,t], nearest_slot_mean)  ∈ [0, 1]
    gate(b,t)    = 1 + k * (facil - 1) * tok_sim(b,t)

    When facil=1.0 (pass 1): gate(b,t) = 1.0  ∀ tokens  ← zero change ✓
    When facil=2.0 (pass 2): gate(b,t) = 1 + k * tok_sim(b,t)
        familiar tokens  (tok_sim=0.8): gate ≈ 1 + 0.8k → strong boost
        unfamiliar tokens (tok_sim=0.1): gate ≈ 1 + 0.1k → mild boost
    When facil=4.0 (pass 3): gate(b,t) = 1 + 3k * tok_sim(b,t)
        familiar tokens: gate ≈ 1 + 2.4k → even stronger
        unfamiliar: barely affects

WHY PER-TOKEN IS RICHER THAN BATCH-LEVEL (gelu249):
    In pass 2, the SAME text is re-read but individual token positions carry
    different semantic loads. Per-token facilitation selectively amplifies
    the activations for tokens that are most representative of the stored
    context, while leaving less-similar tokens unaffected.

    This is more aligned with the thesis's similarity function:
        s(x, c) = per-token cosine similarity to stored context c.
    The depletion/facilitation function m() then scales based on s(x,c).

MONOTONIC GUARANTEE:
    Since facil increases each pass (pre-fire), and tok_sim captures how
    similar each token is to the stored context, the boost accelerates:
        pass 2: gate ∝ (facil=2.0) → moderate selective boost
        pass 3: gate ∝ (facil=4.0) → triple selective boost

PARAMS: log_k_fac (1 scalar, token-selective facilitation strength)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0


class GELU250(nn.Module):
    """Per-token facilitation: gate(b,t) = 1 + k*(facil-1)*tok_sim(b,t)."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N  = buffer_size
        self.log_k_fac = nn.Parameter(torch.tensor(math.log(0.5)))

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
        k_fac = self.log_k_fac.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        if self._buf is None:
            with torch.no_grad():
                self._buf   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask  = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # PASS-1: build buffer, gate=1.0
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

        # PASS-2+: frozen buffer, pre-fire, per-token facilitation
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            if max_sim > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE
            facil_level  = self._facil[nearest_idx].item()
            nearest_vec  = self._buf[nearest_idx].clone()

        # Per-token sim to nearest slot mean
        y_flat  = y.detach().flatten(0, 1)
        y_n     = F.normalize(y_flat, dim=-1)
        nv_n    = F.normalize(nearest_vec.view(1, D), dim=-1)
        tok_sim = (y_n * nv_n).sum(-1).clamp(0.0, 1.0).view(B, T)   # (B, T)

        # gate(b,t) = clip(1 + k*(facil-1)*tok_sim, 1, MAX_GATE)
        gate = (1.0 + k_fac * (facil_level - 1.0) * tok_sim).clamp(1.0, MAX_GATE)
        return y * gate.unsqueeze(-1)
