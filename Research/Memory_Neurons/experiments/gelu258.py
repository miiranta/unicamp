"""GELU258 – Detection-based Frozen Buffer + Directional Facilitation.

NOVEL APPROACH: Facilitate ONLY along the direction of the stored context.

Instead of uniformly scaling all GELU dimensions (gelu249-256), this experiment
boosts activations ALONG THE STORED MEAN DIRECTION and leaves perpendicular
directions unchanged.

MECHANISM:
    Let v = stored_mean[nearest] (normalized)
    Let y = gelu(x)  (B, T, D)

    Project each token activation onto v:
        y_proj = (y · v) v   [component along stored direction]
        y_perp = y - y_proj  [perpendicular component]

    Facilitated output:
        output = y_perp + gate * y_proj
               = y + (gate-1) * y_proj
               = y + (gate-1) * (y·v) * v

    gate > 1: amplifies the stored-direction component
    gate = 1: no change (including pass 1)
    gate < 1: suppresses the stored-direction component

INTUITION:
    The stored mean direction v represents the "typical activation pattern"
    for this context. By amplifying the component of the current activation
    along v, we strengthen the model's response in the direction of familiar
    patterns while leaving orthogonal (novel) dimensions unchanged.

    This is more mathematically principled than uniform scaling:
    - Doesn't change activations that are already orthogonal to stored context
    - Focuses facilitation where the model "recognizes" the pattern
    - Reduces risk of perturbing unrelated dimensions

PRE-FIRE FACILITATION:
    Same as gelu249: facil pre-fires, gate = 1 + k*(facil-1).
    But applied ONLY to the stored-direction component.

PARAMS: log_k_fac (directional facilitation strength, init k=1.0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_FGATE   = 8.0


class GELU258(nn.Module):
    """Directional facilitation: amplify activation along stored-context direction."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N    = buffer_size
        self.log_k = nn.Parameter(torch.tensor(math.log(1.0)))  # k=1.0

        self._buf:  torch.Tensor = None  # normalized means (for lookup + direction)
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
            v = self._buf[nearest_idx].clone()   # normalized context direction (D,)

        # Directional facilitation
        # gate applied only along stored direction v
        gate = min(1.0 + k.item() * (facil_level - 1.0), MAX_FGATE)

        # output = y + (gate-1) * (y·v) * v
        # y: (B,T,D), v: (D,) normalized
        v_exp     = v.view(1, 1, D)                             # (1,1,D)
        proj_coef = (y.detach() * v_exp).sum(-1, keepdim=True)  # (B,T,1) = y·v
        # Amplify stored-direction component: output = y + (gate-1)*(y·v)*v
        # NOTE: proj_coef is detached for stability; gradient flows through y
        delta = (gate - 1.0) * proj_coef * v_exp                # (B,T,D)
        return y + delta
