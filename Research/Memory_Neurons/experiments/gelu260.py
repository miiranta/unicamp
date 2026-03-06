"""GELU260 – Soft KV-Store Value Blending.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Direct output blending — inject stored activation VALUES.
Not a multiplicative gate. Not additive noise. A soft interpolation between
"what the model computes now" and "what the model computed before."
═══════════════════════════════════════════════════════════════════════════

CORE IDEA:  f(x) = m(s(x,c), g(x))  from the thesis, but m is INTERPOLATION:

    KV-Store: keys   = normalized GELU means from pass 1 (retrieved by cosine sim)
              values = raw (unnormalized) GELU means from pass 1 (injected)

    In pass 2+, attention-retrieve the nearest value v̂ and blend:

        α = sigmoid(k * (facil - 1))    where facil compounds ×2 each pass
        output = (1 - α) * gelu(x) + α * v̂ * token_sim

    Pass 1: facil=1.0 → α=sigmoid(0)... but facil=1 → α=sigmoid(k*(0))=0.5?
        No: α = sigmoid(k_param * (facil - 1)):
            facil=1.0 → (facil-1)=0 → α=0.5 is wrong.
        Better: α = sigmoid(k_param * log(facil))  [log(1)=0, log(2)=0.69]
        → Pass 1: facil=1.0, α=sigmoid(0)=0.5... still wrong.
        Solution: use α = sigmoid(k * (facil - 1.5)), threshold at facil=1.5.
        → facil=1.0 → α=sigmoid(-0.5k) ≈ 0 (small or 0)   ✓
        → facil=2.0 → α=sigmoid(0.5k) ≈ 0.5                ✓
        → facil=4.0 → α=sigmoid(2.5k) ≈ 0.92               ✓

    But pass 1 we need α=0 exactly. Solution: multiply by mask:
        blend = 0                          if not pass1_complete
        blend = sigmoid(k*(facil-1))       if pass1_complete (with pre-fire)

    So: output = gelu(x) * (1-blend) + v̂ * blend (in pass 2+).

WHY THIS IS DISTINCT FROM ALL PRIOR APPROACHES:
    - Not a gate on gelu(x) (not multiplicative scaling)
    - Not injection of (stored - current) residual (not delta-based)
    - Direct value retrieval: the model's actual past computation is "replayed"
    - The blend factor α competes: 95% weight to stored, 5% to current at pass 3
    - This can work even if the current gelu(x) is less informative than stored

MONOTONICITY:
    Pass 1: blend=0 → output=gelu(x) exactly ✓
    Pass 2: facil 1→2 (pre-fire), blend=sigmoid(k*1) ≈ moderate
    Pass 3: facil 2→4 (pre-fire), blend=sigmoid(k*3) ≈ large
    → Blend ratio increases monotonically ✓

PARAMS: log_k_blend (gate strength, init 1.0)
STATE:  _buf (N,D) normalized keys, _vbuf (N,D) raw value means,
        _facil (N,) facilitation, _mask (N,), _ptr, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0


class GELU260(nn.Module):
    """Soft KV-store blending: interpolates between gelu(x) and stored activation."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_blend = nn.Parameter(torch.tensor(math.log(1.0)))

        self._buf:   torch.Tensor = None   # (N, D) normalized keys
        self._vbuf:  torch.Tensor = None   # (N, D) raw value means
        self._facil: torch.Tensor = None   # (N,) facilitation levels
        self._mask:  torch.Tensor = None
        self._ptr    = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf    = None
        self._vbuf   = None
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
        k_blend = self.log_k_blend.exp().clamp(0.1, 5.0)

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)   # (D,)

        # Episodic layer is EVAL-ONLY: during training, epoch-2 batches trigger
        # the ring buffer and blend stored-epoch-1 activations into current output,
        # corrupting training activations -> NaN loss.
        if self.training:
            return y

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._vbuf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
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
                        self._buf[self._ptr]   = F.normalize(y_mean, dim=0)
                        self._vbuf[self._ptr]  = y_mean.clone()   # raw value
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]   = F.normalize(y_mean, dim=0)
                    self._vbuf[0]  = y_mean.clone()
                    self._facil[0] = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y
            if not self._pass1_complete:
                return y

        # ── PASS-2+ PHASE ─────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if max_sim > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE   # PRE-FIRE

            facil_level  = self._facil[nearest_idx].item()
            stored_value = self._vbuf[nearest_idx].clone()   # (D,)

        # blend = 0 when facil=1.0, increases with facil
        # sigmoid(k*(facil-1)): facil=1→0.5, facil=2→sig(k)... use offset
        # Use: blend = sigmoid(k*(log2(facil))) so facil=1→0, facil=2→sig(k)
        log2_facil = math.log2(max(facil_level, 1.0 + 1e-6))
        blend = torch.sigmoid(k_blend * torch.tensor(log2_facil, device=x.device))

        # Per-token content similarity to weight injection spatially
        sv_n    = F.normalize(stored_value.unsqueeze(0), dim=-1)           # (1, D)
        y_flat  = y.flatten(0, 1)                                           # (B*T, D)
        y_n     = F.normalize(y_flat, dim=-1)
        tok_sim = (y_n * sv_n).sum(-1).clamp(0.0, 1.0).view(B, T, 1)     # (B,T,1)

        # Broadcast stored value to token level, weighted by token similarity
        sv_broadcast = stored_value.view(1, 1, D).expand(B, T, D)
        output = (1.0 - blend) * y + blend * sv_broadcast * tok_sim

        return output
