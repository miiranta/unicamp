"""GELU267 – Orthogonal Complement Suppression (OCS).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: gelu258 amplifies the component of GELU output ALONG the
stored context direction v. Components perpendicular to v are unchanged.

This experiment does BOTH:
    - AMPLIFY  component along v       (y_par  *= boost,  boost > 1)
    - SUPPRESS component perp to v     (y_perp *= damp,   damp  < 1)

This is a ROTATION in activation space: the output vector is "pulled toward"
the stored context direction while "damping" orthogonal directions.
The net effect is a REDISTRIBUTION of activation energy, not just scaling.
═══════════════════════════════════════════════════════════════════════════

MATHEMATICS:
    v = stored context direction (unit vector from pass-1 batch mean)

    y_par  = (y · v̂) * v̂     [projection onto v̂]
    y_perp = y - y_par        [orthogonal complement]

    output = y_par * boost + y_perp * damp

    where:
        boost = 1 + k_boost * (facil - 1) * sim   [> 1, amplifies context]
        damp  = 1 - k_damp  * (facil - 1) * sim   [< 1, suppresses noise]

    Note: output = y + (boost-1)*y_par + (damp-1)*y_perp
                 = y + (boost-1)*(y·v̂)*v̂ + (damp-1)*(y - (y·v̂)*v̂)
                 = y_par*boost + y_perp*damp   ✓

COMPARE TO gelu258 (directional facilitation):
    gelu258: output = y + (boost-1)*(y·v̂)*v̂   [only ADDS to parallel]
    gelu267: output = y_par*boost + y_perp*damp [ADDS to parallel, SUBTRACTS from perp]
    gelu267 is STRONGER: same boost + additional suppression of irrelevant dims.

WHY SUPPRESS PERPENDICULAR?
    In pass 2, the model is processing the SAME text again. Activation
    components NOT aligned with the stored context direction represent:
    - Random noise at the token level
    - Features from OTHER passages not relevant here
    Suppressing these sharpens the model's "focus" on the familiar content,
    potentially reducing loss on the expected next tokens.

ENERGY CONSERVATION OPTION:
    damp = 2.0 - boost (so that |output|² ≈ |y|² when y_par ⊥ y_perp)
    This is "pure rotation" mode. We give damp its own parameter instead.

MONOTONICITY: pre-fire facilitation → boost & damp both monotonically change.
    Pass 1: facil=1.0 → boost=1, damp=1 → output=y ✓
    Pass 2: facil=2.0 → boost>1, damp<1 → redistribution starts
    Pass 3: facil=4.0 → stronger redistribution ✓

PARAMS: log_k_boost (amplification), log_k_damp (suppression, init 0.3)
STATE:  _buf (N,D) context directions, _facil (N,), _mask, _ptr, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0


class GELU267(nn.Module):
    """Orthogonal complement suppression: amplify context direction, damp rest."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_boost = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_k_damp  = nn.Parameter(torch.tensor(math.log(0.3)))

        self._buf:   torch.Tensor = None   # (N, D) unit context vectors
        self._facil: torch.Tensor = None
        self._mask:  torch.Tensor = None
        self._ptr    = 0
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
        k_boost = self.log_k_boost.exp().clamp(0.01, 4.0)
        k_damp  = self.log_k_damp.exp().clamp(0.01, 0.9)   # damp ≤ 1 safety

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)

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
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]   = F.normalize(y_mean, dim=0)   # unit vec
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]   = F.normalize(y_mean, dim=0)
                    self._facil[0] = 1.0
                    self._mask[0]  = True
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
                self._facil[nearest_idx] *= FACIL_RATE   # PRE-FIRE

            facil_level = self._facil[nearest_idx].item()
            v           = self._buf[nearest_idx].clone()   # (D,) unit context vector

        # Decompose y into parallel and perpendicular components
        # v̂ is already unit-normalized in the buffer
        v_bcast = v.view(1, 1, D)                              # (1, 1, D)
        proj    = (y * v_bcast).sum(-1, keepdim=True)          # (B, T, 1)  y·v̂
        y_par   = proj * v_bcast                               # (B, T, D)  (y·v̂)v̂
        y_perp  = y - y_par                                    # (B, T, D)

        # Scale factors modulated by facilitation and similarity
        mod    = (facil_level - 1.0) * sim_val
        boost  = 1.0 + k_boost.item() * mod
        damp   = max(0.01, 1.0 - k_damp.item() * mod)   # never negative

        output = y_par * boost + y_perp * damp
        return output
