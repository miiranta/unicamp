"""GELU245 – Per-Token Depletion + Hit Count Hybrid.

COMBINES gelu239's continuous depletion WITH gelu241's integer hit counter.

The two mechanisms address different aspects of familiarity:

    Continuous depletion (depl ∈ [0,1]):
        - Fine-grained: tracks DEGREE of depletion, not just count
        - Self-recovering if we add recovery: depl slowly grows back toward 1.0
          between passes (here: no recovery, full reset between test sessions)

    Integer hit count (h ∈ ℤ≥0):
        - Coarse: counts number of times a slot was fired
        - Monotonically increases, no recovery

Combined gate:
    gate = exp(-k_depl * (1 - depl)) * exp(-lambda * h)
         = exp(-k_depl * (1 - depl) - lambda * h)

When depl=1.0 (fresh) AND h=0 (fresh, pass 1):
    gate = exp(0) * exp(0) = 1.0   ← perfect pass-1 preservation

When depl=0.5 AND h=1 (fired once, pass 2):
    gate = exp(-k_depl * 0.5) * exp(-lambda)
    = exp(-k/2 - lambda)  ← double penalty vs using either alone

When depl=0.25 AND h=2 (fired twice, pass 3):
    gate = exp(-k * 0.75) * exp(-2*lambda)  ← triple penalty

This COMPOUNDS the suppression across passes, giving much larger Δ1→3 than either
mechanism alone.

PASS-1 GUARANTEE: depl=1.0 AND h=0 for all fresh slots → gate=1.0 throughout pass 1.

PARAMS: log_k_depl (k, init 1.5), log_lambda (lambda, init 0.5)
STATE:  _buf, _depl, _hits, _mask, _ptr
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU245(nn.Module):
    """Hybrid: continuous depletion × hit-count gate for compounding pass penalties."""

    def __init__(self, buffer_size: int = 512, depletion_rate: float = 0.5):
        super().__init__()
        self._N  = buffer_size
        self._DR = depletion_rate

        self.log_k      = nn.Parameter(torch.tensor(math.log(1.5)))  # k_depl
        self.log_lambda = nn.Parameter(torch.tensor(math.log(0.5)))  # lambda

        self._buf:  torch.Tensor = None
        self._depl: torch.Tensor = None
        self._hits: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

    def reset_state(self):
        self._buf   = None
        self._depl  = None
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
        k_depl = self.log_k.exp().clamp(0.1, 5.0)
        lam    = self.log_lambda.exp().clamp(0.1, 3.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._depl = torch.ones(self._N,    device=x.device, dtype=y.dtype)
                self._hits = torch.zeros(self._N,   device=x.device, dtype=torch.long)
                self._mask = torch.zeros(self._N,   device=x.device, dtype=torch.bool)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._depl[0] = 1.0
                self._hits[0] = 0
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Nearest slot ──────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()
            h           = self._hits[nearest_idx].item()

        # Compound gate = exp(-k*(1-depl)) * exp(-lambda*h)
        gate = math.exp(-k_depl.item() * (1.0 - depl_level) - lam.item() * h)
        output = y * gate

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= self._DR
                self._hits[nearest_idx] += 1
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._depl[self._ptr] = 1.0
            self._hits[self._ptr] = 0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
