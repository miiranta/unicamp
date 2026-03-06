"""GELU272 – Affine Statistics Alignment (Distribution Matching Memory).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Every prior experiment multiplies gelu(x) by a scalar or
vector gate. This experiment applies an AFFINE REMAPPING of the activation
DISTRIBUTION — matching pass-2 activations to the stored pass-1 statistics
for the same slot, NOT multiplying by a gate.
═══════════════════════════════════════════════════════════════════════════

CORE IDEA:
    In pass 1 for slot s, store:
        μ₁ᵢ = mean of GELU(x) over batch (D,)
        σ₁ = std of GELU(x) over all tokens and batch (scalar)

    In pass 2, when batch matches slot s with high similarity:
        Standardize current activations:
            y_std = (y - y_mean_now) / (y_std_now + ε)   [shape: B,T,D]

        Remap to pass-1 DISTRIBUTION:
            y_aligned = y_std * σ₁ + μ₁                  [shape: B,T,D]

        Blend by learned weight:
            blend = sigmoid(k_blend * log(hit_count + 1))  [0 at pass 1]
            output = (1 - blend) * y + blend * y_aligned

    Pass 1: blend=0 → output=y (no change) ✓
    Pass 2: blend>0, output moves toward pass-1 DISTRIBUTION
    Pass 3: blend larger → stronger alignment ✓

WHY DISTRIBUTION MATCHING SHOULD HELP:
    If the model processes the SAME TEXT twice, the token activations
    should ideally have the SAME distribution. Any deviation in pass 2
    is due to non-stationary EMA stats, random dropout, etc.
    By precisely re-aligning the distribution to pass-1, we reduce this
    inconsistency, and the subsequent projection layers see coherent input.

    This is analogous to "test-time batch normalization" or "distribution
    shift correction" — a known technique for domain adaptation.

    Key insight: it's not about boosting magnitudes, but about correctness.
    The right answer isn't louder — it just needs to be in the right place.

AFFINE REMAPPING MATH:
    y_mean_now = y.mean(dim=(0,1))            [B*T aggregate mean]
    y_std_now  = y.std(dim=(0,1)) + eps
    y_std_pass1 stored per slot (scalar for simplicity, median over D)

    For the scalar version (tractable):
        s₁ = stored scalar std from pass 1
        s₂ = current scalar std
        μ₁ = stored mean vector
        μ₂ = current mean vector

        y_aligned = μ₁ + (y - μ₂) * (s₁ / s₂)

PARAMS: log_k_blend (blend ramp, init k=1.0)
STATE:  _buf_mean (N,D) stored means,  _buf_std (N,) stored scalar stds,
        _buf_keys (N,D) normalized for lookup,  _hit_count (N,) int,
        _mask (N,) bool,  _ptr int,  _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
N_BUF       = 512


class GELU272(nn.Module):
    """Affine stats alignment: remap pass-2 activations to stored pass-1 distribution."""

    def __init__(self, buffer_size: int = N_BUF, eps: float = 1e-5):
        super().__init__()
        self._N  = buffer_size
        self.eps = eps

        self.log_k_blend = nn.Parameter(torch.tensor(math.log(1.0)))

        self._buf_keys:  torch.Tensor = None   # (N, D) normalized
        self._buf_mean:  torch.Tensor = None   # (N, D) raw means
        self._buf_std:   torch.Tensor = None   # (N,) scalar stds
        self._hit_count: torch.Tensor = None   # (N,) int
        self._mask:      torch.Tensor = None   # (N,) bool
        self._ptr        = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf_keys   = None
        self._buf_mean   = None
        self._buf_std    = None
        self._hit_count  = None
        self._mask       = None
        self._ptr        = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_blend = self.log_k_blend.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1)   # (B*T, D)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf_keys is None:
            with torch.no_grad():
                self._buf_keys  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_mean  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_std   = torch.zeros(self._N,    device=x.device, dtype=y.dtype)
                self._hit_count = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                mean_vec = m_curr.mean(0)          # (D,)
                std_sc   = m_curr.std().item()     # scalar std over all tokens+dims
                if self._mask.any():
                    q    = F.normalize(mean_vec.unsqueeze(0), dim=-1)
                    sims = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf_keys[self._ptr]  = F.normalize(mean_vec, dim=0)
                        self._buf_mean[self._ptr]  = mean_vec
                        self._buf_std[self._ptr]   = std_sc
                        self._hit_count[self._ptr] = 0
                        self._mask[self._ptr]      = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf_keys[0]  = F.normalize(mean_vec, dim=0)
                    self._buf_mean[0]  = mean_vec
                    self._buf_std[0]   = std_sc
                    self._hit_count[0] = 0
                    self._mask[0]      = True
                    self._ptr          = 1
                    return y

        # ── Pass-2+ affine alignment ───────────────────────────────────
        with torch.no_grad():
            mean_now = m_curr.mean(0)             # (D,)
            std_now  = m_curr.std().clamp(min=self.eps).item()

            q           = F.normalize(mean_now.unsqueeze(0), dim=-1)
            sims        = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()

            if sims[nearest_idx].item() > FIRE_THRESH:
                self._hit_count[nearest_idx] += 1

            count    = self._hit_count[nearest_idx].item()
            mu1      = self._buf_mean[nearest_idx]    # (D,)
            s1       = self._buf_std[nearest_idx].item()

        # blend ramps from 0 → 1 based on log(hit_count+1)
        blend = torch.sigmoid(torch.tensor(k_blend.item() * math.log(float(count) + 1.0) - 0.5))
        blend_val = blend.item()

        if blend_val < 1e-6:
            return y

        # y_aligned = mu1 + (y - mean_now) * (s1 / std_now)
        scale_ratio = s1 / (std_now + self.eps)
        y_aligned   = mu1.view(1, 1, D) + (y - mean_now.view(1, 1, D)) * scale_ratio

        output = (1.0 - blend_val) * y + blend_val * y_aligned
        return output
