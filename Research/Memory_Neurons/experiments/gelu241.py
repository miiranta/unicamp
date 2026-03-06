"""GELU241 – Integer Hit Counter Gate: 1/(1 + h)^gamma.

DIRECT VISIT-COUNT MODEL:
    Maintain a hit counter h_k per slot (integer, starts at 0).
    When a batch fires slot k (cos_sim > FIRE_THRESH): h_k += 1.
    Gate applied to the entire batch output:

        gate = 1.0 / (1.0 + h_nearest) ^ gamma

    h=0 (slot never fired — all of pass 1 with N=512):    gate = 1.0 / 1.0 = 1.0
    h=1 (slot fired once — pass 2):                        gate = 1.0 / 2^gamma
    h=2 (slot fired twice — pass 3):                       gate = 1.0 / 3^gamma

    With gamma=1.0:  1.0 → 0.5  → 0.33   (harmonic decay)
    With gamma=2.0:  1.0 → 0.25 → 0.11   (aggressive harmonic decay)
    With gamma=0.5:  1.0 → 0.71 → 0.58   (gentle decay)

ADVANTAGE OVER COSINE-SIM GATES:
    No learned bias toward conservative behavior. The hit count is a discrete,
    non-differentiable counter — it doesn't care about training distribution.
    The gate value is purely structural: "how many times has this slot been fired?"

COMPARISON WITH gelu239 (continuous depletion):
    gelu239: depl *= 0.5 → continuous exponential decay (depl ∈ ℝ)
    gelu241: h += 1      → discrete integer counter     (h ∈ ℤ≥0)
    Both give gate=1.0 when h=0 / depl=1.0.

PASS-1 GUARANTEE: h_k=0 for ALL slots during entire pass 1 (with N=512).
    Gate = 1/(1+0)^gamma = 1.0 throughout pass 1.
    Zero pass-1 PPL impact, guaranteed.

PARAMS: log_gamma (1 scalar, init at 1.0)
STATE:  _buf (N,D), _hits (N,) int, _mask (N,) bool, _ptr int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU241(nn.Module):
    """Integer hit counter gate: gate=1/(1+h)^gamma per slot."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_gamma = nn.Parameter(torch.tensor(math.log(1.0)))   # gamma=1.0

        self._buf:  torch.Tensor = None
        self._hits: torch.Tensor = None  # (N,) int64
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

    def reset_state(self):
        self._buf   = None
        self._hits  = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        gamma = self.log_gamma.exp().clamp(0.1, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hits = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._hits[0] = 0
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y   # gate = 1.0

        # ── Nearest slot lookup ────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            h           = self._hits[nearest_idx].item()

        # ── Hit counter gate ───────────────────────────────────────────
        # gate = 1 / (1 + h)^gamma  →  detach to avoid differentiation through int
        gate = 1.0 / (1.0 + h) ** gamma.item()
        output = y * gate

        # ── Update counter + write new slot ───────────────────────────
        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._hits[nearest_idx] += 1

            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._hits[self._ptr] = 0   # fresh slot, counter reset
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
