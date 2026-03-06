"""GELU261 – Per-Dimension Activation Magnitude Gate.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Every previous buffer/gate experiment uses a SCALAR gate
                applied uniformly across all D dimensions.
                This experiment computes a D-DIMENSIONAL gate based on which
                neurons were the MOST ACTIVE during pass 1.
═══════════════════════════════════════════════════════════════════════════

CORE IDEA:
    During pass 1, for each stored slot accumulate the D-dimensional
    absolute mean activation: μ_d = E[|GELU(x)_d|] across tokens & batches.

    During pass 2+:
        importance_d = μ_d / (mean_d μ_d)   (relative per-dim importance)
        gate_d = 1 + k * (facil - 1) * importance_d * scalar_sim

    Neurons with HIGH average activation in pass 1 get the largest boost.
    Neurons that were consistently near-zero during pass 1 get gate ≈ 1.

WHY THIS SHOULD WORK:
    In a language model MLP, different neurons specialize in different
    linguistic features. The neurons that were most active during pass 1
    carry the most information about that text's content.
    By amplifying THOSE neurons preferentially in pass 2, we give the model
    more signal precisely where it matters most for that specific text.

    This is inspired by the "lottery ticket" hypothesis: a small subset of
    neurons carries most of the prediction signal. Identify them from pass 1,
    amplify them in pass 2.

    Standard scalar facilitation treats all dimensions equally.
    This experiment lets the TEXT ITSELF decide which dimensions to amplify.

DIMENSION NORMALIZATION:
    importance_d is normalized to have mean 1.0, so when averaged the gate
    has the same mean value as scalar facilitation with the same k.
    No unfair comparison.

MONOTONICITY:
    facil=1.0 (pass 1): gate_d = 1.0 for all d ✓
    facil=2.0 (pass 2, pre-fire): gate_d = 1 + k * importance_d * sim
    facil=4.0 (pass 3): gate_d = 1 + 3k * importance_d * sim  (3× stronger)
    → Per-dim, each dimension monotonically increases ✓

PARAMS: log_k_mag (gate strength scalar, init k=0.5)
STATE:  _keys (N,D) normalized gelu means (for cosine lookup),
        _mag  (N,D) per-dim |activation| means (importance map),
        _facil (N,), _mask (N,), _ptr, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0


class GELU261(nn.Module):
    """Per-dimension magnitude-weighted facilitation gate."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_mag = nn.Parameter(torch.tensor(math.log(0.5)))

        self._keys:  torch.Tensor = None   # (N, D) normalized keys
        self._mag:   torch.Tensor = None   # (N, D) per-dim |y| mean
        self._facil: torch.Tensor = None   # (N,)
        self._mask:  torch.Tensor = None
        self._ptr    = 0
        self._pass1_complete = False

    def reset_state(self):
        self._keys   = None
        self._mag    = None
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
        k_mag = self.log_k_mag.exp().clamp(0.01, 5.0)

        y       = self._gelu(x)
        y_flat  = y.detach().flatten(0, 1)            # (B*T, D)
        y_mean  = y_flat.mean(0)                       # (D,)  batch mean
        y_amean = y_flat.abs().mean(0)                 # (D,)  per-dim |y| mean

        # ── Init ──────────────────────────────────────────────────────
        if self._keys is None:
            with torch.no_grad():
                self._keys   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mag    = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil  = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask   = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n    = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    keys_n = F.normalize(self._keys, dim=-1)
                    sims   = (keys_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._keys[self._ptr]  = F.normalize(y_mean, dim=0)
                        self._mag[self._ptr]   = y_amean                  # (D,)
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._keys[0]  = F.normalize(y_mean, dim=0)
                    self._mag[0]   = y_amean
                    self._facil[0] = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y
            if not self._pass1_complete:
                return y

        # ── PASS-2+ PHASE ─────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            keys_n      = F.normalize(self._keys, dim=-1)
            sims        = (keys_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if max_sim > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE   # PRE-FIRE

            facil_level = self._facil[nearest_idx].item()
            mag_vec     = self._mag[nearest_idx].clone()   # (D,) per-dim magnitude

            # Normalize importance: mean=1.0  →  gate mean == 1+k*(facil-1) like gelu249
            mean_mag = mag_vec.mean().clamp(min=1e-6)
            importance = mag_vec / mean_mag                  # (D,) mean=1.0

        # Per-token scalar similarity
        mv_n    = F.normalize(self._keys[nearest_idx].unsqueeze(0), dim=-1)
        y_n     = F.normalize(y_flat, dim=-1)
        tok_sim = (y_n * mv_n).sum(-1).clamp(0.0, 1.0).view(B, T, 1)   # (B,T,1)

        # gate_d = 1 + k*(facil-1)*importance_d, scalar_sim weighting
        gate_d = (1.0 + k_mag * (facil_level - 1.0) * importance).clamp(max=MAX_GATE)  # (D,)
        gate   = gate_d.view(1, 1, D) * tok_sim + (1.0 - tok_sim)  # blend with 1.0 at low sim

        return y * gate
