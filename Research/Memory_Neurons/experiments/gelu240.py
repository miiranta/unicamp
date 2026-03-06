"""GELU240 – Per-Token Depletion Gate.

BUILDS ON gelu239 by applying the depletion gate AT TOKEN LEVEL instead of batch level.

gelu239 applies a uniform scalar gate across the entire batch (every token in the
batch gets the same gate value = exp(-k * (1 - depl_nearest))).

gelu240 ADDITIONALLY modulates per-token based on cosine similarity to the nearest
stored batch mean:
    Per-token sim: tok_sim = cos(y_token, nearest_batch_mean)   (B, T)
    Gate: exp(-k * (1 - depl_nearest) * tok_sim.clamp(0, 1))

When depl_nearest = 1.0 (fresh): gate = 1.0 for ALL tokens regardless of tok_sim.
When depl_nearest = 0.5 (fired): 
    tok_sim=0.9 (very familiar token): gate = exp(-k*0.5*0.9) = exp(-0.45k) → small
    tok_sim=0.1 (dissimilar token):    gate = exp(-k*0.5*0.1) = exp(-0.05k) ≈ 1.0

ADVANTAGE OVER gelu239:
    Within a batch on pass 2, some tokens are highly aligned with the stored mean
    (e.g., common function words appearing repeatedly) while others diverge.
    gelu240 selectively suppresses the familiar TOKENS within familiar BATCHES.
    
    This should give finer-grained adaptation: suppressing precisely what was seen
    before, while leaving genuinely different content unaffected.

SAME ZERO-PASS-1-PPL property as gelu239:
    When depl_nearest = 1.0 (all of pass 1 with N=512, FIRE_THRESH=0.85):
        gate = exp(-k * 0 * anything) = exp(0) = 1.0  ∀ tokens.

PARAMS: log_k (1 scalar), same as gelu239
STATE:  same as gelu239
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU240(nn.Module):
    """Per-token depletion gate: suppresses familiar tokens within familiar batches."""

    def __init__(self, buffer_size: int = 512, depletion_rate: float = 0.5):
        super().__init__()
        self._N  = buffer_size
        self._DR = depletion_rate
        self.log_k = nn.Parameter(torch.tensor(math.log(2.0)))

        self._buf:  torch.Tensor = None
        self._depl: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

    def reset_state(self):
        self._buf   = None
        self._depl  = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_gate = self.log_k.exp().clamp(0.1, 8.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._depl = torch.ones(self._N,    device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N,   device=x.device, dtype=torch.bool)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._depl[0] = 1.0
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Nearest slot lookup ────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()
            nearest_vec = self._buf[nearest_idx].clone()

        # ── Per-token similarity ───────────────────────────────────────
        y_flat  = y.flatten(0, 1)                                       # (B*T, D)
        y_n     = F.normalize(y_flat, dim=-1)
        nv_n    = F.normalize(nearest_vec.view(1, D), dim=-1)
        tok_sim = (y_n * nv_n).sum(-1).clamp(0.0, 1.0).view(B, T)     # (B, T)

        # ── Per-token depletion gate ───────────────────────────────────
        # gate = exp(-k * (1 - depl) * tok_sim)
        # depl=1.0: gate = 1.0 for ALL tokens (zero pass-1 impact)
        # depl=0.5: gate scales with tok_sim (selective suppression)
        gate = torch.exp(-k_gate * (1.0 - depl_level) * tok_sim)       # (B, T)
        output = y * gate.unsqueeze(-1)

        # ── Deplete on strong match + write new slot ───────────────────
        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= self._DR
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._depl[self._ptr] = 1.0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
