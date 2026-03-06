"""GELU264 – Two-Scale Memory Gate: Local (episodic) + Global (semantic).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Prior experiments use ONLY local (per-batch episodic) or
ONLY global (EMA) similarity signals. This experiment combines BOTH:
    - Local signal:  per-batch ring-buffer nearest-neighbor similarity
    - Global signal: cosine similarity to the MEAN of all pass-1 batches
Both are independent signals, weighted by separately learned parameters.
═══════════════════════════════════════════════════════════════════════════

DESIGN:
    During pass 1, simultaneously build:
        A) Ring buffer of per-batch normalized means  → local episodic memory
        B) Running mean of all pass-1 batch means     → global semantic memory

    During pass 2+:
        sim_local  = max cosine(current, buffer[nearest])   [episodic]
        sim_global = cosine(current, global_mean)           [semantic/distributional]
        gate = 1 + k_l * (facil_l-1) * sim_local
                 + k_g * (facil_g-1) * sim_global

    Both facil_l and facil_g compound independently:
        facil_l[nearest_slot] *= RATE_L  (per-slot)
        facil_g                *= RATE_G  (global)

WHY TWO SCALES?
    Local signal (per-batch): captures SPECIFIC content (e.g., this exact
        paragraph of text). High at pass 2 for the exact same batch.
    Global signal (distribution): captures GENERAL familiarity with the
        STYLE/REGISTER of the text. Always elevated if test domain is similar.

    Local: HIGH at pass 2 (exact match), VERY HIGH at pass 3.
    Global: MODERATE at pass 2 (distribution familiar), STAYS moderate.
    Combined: local dominates the episodic component, global provides
    a "floor" of adaptation for all batches.

    Using only local: misses batches that don't hit the FIRE_THRESH exactly.
    Using only global: can't distinguish "same batch" from "different batch".
    Combined: best of both.

MONOTONICITY:
    Pass 1: facil_l=1, facil_g=1 → gate=1.0 ✓
    Pass 2: both facil compound → gate > 1 ✓
    Pass 3: both facil compound more → gate larger ✓

PARAMS: log_k_local, log_k_global (independent strengths, both init 0.5)
STATE:  _buf (N,D), _facil_l (N,), _global_mean (D,), _facil_g (scalar),
        _mask, _ptr, _n_pass1, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
RATE_L       = 2.0    # per-slot facilitation rate
RATE_G       = 1.5    # global facilitation rate (gentler, avoids overflow)
MAX_GATE     = 8.0


class GELU264(nn.Module):
    """Two-scale memory gate: local episodic + global semantic facilitation."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_local  = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_k_global = nn.Parameter(torch.tensor(math.log(0.5)))

        self._buf:         torch.Tensor = None   # (N, D) normalized keys
        self._facil_l:     torch.Tensor = None   # (N,) per-slot facilitation
        self._global_mean: torch.Tensor = None   # (D,) global mean of pass-1 batches
        self._facil_g      = 1.0                  # global facilitation scalar
        self._mask:        torch.Tensor = None
        self._ptr          = 0
        self._n_pass1      = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf          = None
        self._facil_l      = None
        self._global_mean  = None
        self._facil_g      = 1.0
        self._mask         = None
        self._ptr          = 0
        self._n_pass1      = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_l = self.log_k_local.exp().clamp(0.01, 4.0)
        k_g = self.log_k_global.exp().clamp(0.01, 4.0)

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf        = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil_l    = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._global_mean = torch.zeros(D,         device=x.device, dtype=y.dtype)
                self._mask       = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr, self._n_pass1 = 0, 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        # Normalize global mean
                        self._global_mean = F.normalize(self._global_mean, dim=0)
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]    = F.normalize(y_mean, dim=0)
                        self._facil_l[self._ptr] = 1.0
                        self._mask[self._ptr]   = True
                        self._ptr = (self._ptr + 1) % self._N
                        # Online update of global mean (unnormalized accumulation)
                        n = self._n_pass1
                        self._global_mean = (self._global_mean * n + y_mean) / (n + 1)
                        self._n_pass1 += 1
                        return y
                else:
                    self._buf[0]     = F.normalize(y_mean, dim=0)
                    self._facil_l[0] = 1.0
                    self._mask[0]    = True
                    self._global_mean = y_mean.clone()
                    self._n_pass1     = 1
                    self._ptr         = 1
                    return y
            if not self._pass1_complete:
                return y

        # ── PASS-2+ PHASE ─────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims_local  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims_local.argmax()
            sim_l       = sims_local[nearest_idx].clamp(0.0, 1.0).item()

            sim_g = (m_n * self._global_mean.unsqueeze(0)).sum(-1).clamp(0.0, 1.0).item()

            if sim_l > FIRE_THRESH:
                self._facil_l[nearest_idx] *= RATE_L   # PRE-FIRE local
                self._facil_g = min(self._facil_g * RATE_G, 16.0)  # PRE-FIRE global

            facil_l = self._facil_l[nearest_idx].item()
            facil_g = self._facil_g

        gate = min(
            1.0
            + k_l.item() * (facil_l - 1.0) * sim_l
            + k_g.item() * (facil_g - 1.0) * sim_g,
            MAX_GATE
        )
        return y * gate
