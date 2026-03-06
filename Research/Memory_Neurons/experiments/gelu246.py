"""GELU246 – Output Substitution Depletion Gate.

NOVEL MECHANISM: Instead of gating the GELU output by a scalar, on pass 2
the familiar token's activation is BLENDED with the STORED PASS-1 ACTIVATION
(from the buffer). This is "output recall" rather than "output suppression."

MOTIVATION:
    Current gates multiply output by a scalar < 1 (suppression).
    This reduces the magnitude of familiar activations, which may have side effects
    on subsequent layers (normalization, attention, etc.) that destabilize PPL.

    Alternative: blend output with cached value:
        output = (1 - blend) * gelu(x) + blend * cached_mean

    This preserves the DIRECTION of the activation (same as pass-1 cached mean)
    while varying the weight. On pass 1 (blend=0): output = gelu(x) exactly.
    On pass 2 (blend based on depletion): output = partial recall of pass-1 mean.

BLEND FORMULA:
    blend_k = 1 - depl_k
    output = (1-blend) * gelu(x) + blend * cached_mean_k
           = depl_k * gelu(x) + (1 - depl_k) * cached_mean_k

    depl=1.0 (fresh, pass 1):  output = gelu(x)                 ← exact
    depl=0.5 (fired, pass 2):  output = 0.5*gelu(x) + 0.5*mean  ← blend
    depl=0.25 (pass 3):        output = 0.25*gelu(x) + 0.75*mean ← mostly recall

INTERESTING PROPERTIES:
    - Different from suppression: magnitude stays similar but direction pulled
      toward stored mean → less "surprising" output in familiar context.
    - More stable than scalar gate for downstream layers (shape preserved).
    - Can be thought of as "completion" of a familiar pattern from memory.

PASS-1 GUARANTEE: depl=1.0 → blend=0 → output = gelu(x) exactly. Zero change.

PARAMS: none beyond buffer state (no learnable gate — depletion directly controls blend)
STATE:  _buf (N,D) normalized means, _depl (N,), _mask (N,), _ptr int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU246(nn.Module):
    """Output blending with cached pass-1 mean based on depletion level."""

    def __init__(self, buffer_size: int = 512, depletion_rate: float = 0.5):
        super().__init__()
        self._N  = buffer_size
        self._DR = depletion_rate

        # Magnitude of cached mean: buffer stores normalized vectors.
        # We need to store the raw (unnormalized) mean for blending.
        # _norm_buf stores ||m_curr|| per slot for scale recovery.
        # Alternative: store unnormalized directly (no log_k needed).
        # Here we use a learned blend strength amplifier:
        self.log_blend_amp = nn.Parameter(torch.tensor(0.0))  # blend scale ≈ 1.0

        self._buf:      torch.Tensor = None   # (N, D) normalized means
        self._buf_raw:  torch.Tensor = None   # (N, D) UNnormalized means (for blending)
        self._depl:     torch.Tensor = None   # (N,)
        self._mask:     torch.Tensor = None
        self._ptr  = 0
        self._ready = False

    def reset_state(self):
        self._buf     = None
        self._buf_raw = None
        self._depl    = None
        self._mask    = None
        self._ptr     = 0
        self._ready   = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        blend_amp = torch.sigmoid(self.log_blend_amp)   # ∈ (0, 1), default ≈ 0.5

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,) unnormalized

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf     = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_raw = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._depl    = torch.ones(self._N,    device=x.device, dtype=y.dtype)
                self._mask    = torch.zeros(self._N,   device=x.device, dtype=torch.bool)
                self._buf[0]     = F.normalize(m_curr, dim=0)
                self._buf_raw[0] = m_curr
                self._depl[0]    = 1.0
                self._mask[0]    = True
            self._ptr  = 1
            self._ready = True
            return y   # output = gelu(x) exactly

        # ── Nearest slot lookup ────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()
            cached_mean = self._buf_raw[nearest_idx].clone()   # (D,) unnormalized

        # ── Output blending based on depletion ──────────────────────────
        # blend = blend_amp * (1 - depl)
        # depl=1.0: blend=0 → output=gelu(x) exactly
        # depl=0.5: blend=0.5*blend_amp → partial recall
        blend_coeff = blend_amp * (1.0 - depl_level)   # scalar
        # Expand cached mean to (B, T, D) for blending
        cached_expanded = cached_mean.view(1, 1, D).expand(B, T, D)
        output = (1.0 - blend_coeff) * y + blend_coeff * cached_expanded

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= self._DR
            self._buf[self._ptr]     = F.normalize(m_curr, dim=0)
            self._buf_raw[self._ptr] = m_curr
            self._depl[self._ptr]    = 1.0
            self._mask[self._ptr]    = True
            self._ptr = (self._ptr + 1) % self._N

        return output
