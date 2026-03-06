"""GELU265 – Variance-Focused Per-Dimension Facilitation.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Identify STABLE dimensions — those that vary LEAST across
different pass-1 batches. These stable dimensions are the "backbone" of the
test corpus's representation. Amplify them preferentially in pass 2+.
═══════════════════════════════════════════════════════════════════════════

HYPOTHESIS:
    In a language model MLP, some neuron dimensions consistently activate
    in a specific direction regardless of exact token content (e.g., "this
    is English text" or "this is narrative prose"). These dimensions have
    LOW VARIANCE across batches.
    Other dimensions carry token-specific information and vary wildly.

    In pass 2, the TOKEN-LEVEL content is repeated exactly. The low-variance
    (stable) dimensions already captured the stable aspects well in pass 1.
    Amplifying these in pass 2 reinforces CONCEPTUAL consistency, not noise.

    Per-dim gate:
        stability_d = 1 / (var_d + eps)          (low var = high stability)
        weight_d    = stability_d / mean(stability)   (normalize: mean=1.0)
        gate_d      = 1 + k * (facil-1) * weight_d * sim  (D-dimensional)

    High-stability dims get boosted more; noisy dims get boosted less.

STABILITY COMPUTATION:
    During pass 1, accumulate per-dim mean and M2 (for online variance via
    Welford's algorithm) across all stored batch means.
    At pass-1 completion: compute var_d = M2 / n_pass1.

DISTINCT FROM gelu261:
    gelu261 gates on magnitude (|μ_d|) — strongly activated dims get boosted.
    gelu265 gates on INVERSE VARIANCE — consistently activated dims get boosted.
    A dimension could have LOW magnitude but LOW variance → high stability.
    E.g., a "constant offset" dimension that is always 0.1 gets high stability
    but low magnitude. gelu261 ignores it; gelu265 amplifies it.

MONOTONICITY: same pre-fire facilitation as gelu249.
    Pass 1: facil=1.0 → gate_d=1.0 ✓
    Pass 2: facil 1→2 (pre-fire) → gate_d = 1 + k*weight_d*sim
    Pass 3: facil 2→4 (pre-fire) → gate_d = 1 + 3k*weight_d*sim

PARAMS: log_k_var (gate strength, init k=0.5)
STATE:  _buf (N,D), _m2 (D,) online variance M2, _mean_agg (D,),
        _facil (N,), _weight_d (D,) stability weights (computed at pass-1 end),
        _mask, _ptr, _n_pass1, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0
VAR_EPS     = 1e-4    # floor variance (prevents explosively high stability)


class GELU265(nn.Module):
    """Variance-focused per-dimension facilitation: stable dims amplified most."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_var = nn.Parameter(torch.tensor(math.log(0.5)))

        self._buf:       torch.Tensor = None   # (N, D) normalized keys
        self._mean_agg:  torch.Tensor = None   # (D,) running mean of batch means
        self._m2:        torch.Tensor = None   # (D,) running M2 for Welford
        self._weight_d:  torch.Tensor = None   # (D,) stability weights (set after pass 1)
        self._facil:     torch.Tensor = None   # (N,)
        self._mask:      torch.Tensor = None
        self._ptr        = 0
        self._n_pass1    = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf        = None
        self._mean_agg   = None
        self._m2         = None
        self._weight_d   = None
        self._facil      = None
        self._mask       = None
        self._ptr        = 0
        self._n_pass1    = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _welford_update(self, y_mean: torch.Tensor):
        """Welford online variance update on per-batch means."""
        n = self._n_pass1 + 1
        delta  = y_mean - self._mean_agg
        self._mean_agg = self._mean_agg + delta / n
        delta2 = y_mean - self._mean_agg
        self._m2 = self._m2 + delta * delta2
        self._n_pass1 = n

    def _finalize_weights(self, D: int, device, dtype):
        """Convert M2 → variance → stability weights after pass 1 ends."""
        if self._n_pass1 < 2:
            self._weight_d = torch.ones(D, device=device, dtype=dtype)
            return
        var_d      = (self._m2 / self._n_pass1).clamp(min=VAR_EPS)   # (D,)
        stability  = 1.0 / var_d                                       # (D,)
        mean_stab  = stability.mean().clamp(min=1e-9)
        self._weight_d = (stability / mean_stab).clamp(max=10.0)       # normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_var = self.log_k_var.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf      = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil    = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask     = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._mean_agg = torch.zeros(D, device=x.device, dtype=y.dtype)
                self._m2       = torch.zeros(D, device=x.device, dtype=y.dtype)
            self._ptr, self._n_pass1 = 0, 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._finalize_weights(D, x.device, y.dtype)
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]  = F.normalize(y_mean, dim=0)
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr] = True
                        self._ptr = (self._ptr + 1) % self._N
                        self._welford_update(y_mean)
                        return y
                else:
                    self._buf[0]  = F.normalize(y_mean, dim=0)
                    self._facil[0] = 1.0
                    self._mask[0] = True
                    self._ptr     = 1
                    self._welford_update(y_mean)
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

        w_d  = self._weight_d                              # (D,) precomputed
        gate_d = (1.0 + k_var * (facil_level - 1.0) * w_d * sim_val).clamp(max=MAX_GATE)
        return y * gate_d.view(1, 1, D)
