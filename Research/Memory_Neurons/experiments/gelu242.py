"""GELU242 – Exponential Hit Count Gate: exp(-lambda * h).

ALTERNATIVE TO gelu241's harmonic decay. Uses exponential decay with hit count:

    gate = exp(-lambda * h)

    h=0 (fresh, pass 1):     gate = 1.0
    h=1 (fired once, pass 2): gate = exp(-lambda)
    h=2 (fired twice, pass 3): gate = exp(-2*lambda)

With lambda=0.7:  1.0 → 0.50 → 0.25   (halving each pass)
With lambda=1.0:  1.0 → 0.37 → 0.14   (stronger suppression)
With lambda=0.4:  1.0 → 0.67 → 0.45   (gentler suppression)

COMPARISON:
    gelu241 (harmonic): 1.0 → 0.50 → 0.33 (eventually levels off at 1/h)
    gelu242 (exponential): 1.0 → 0.50 → 0.25 (keeps halving)

The exponential model gives STRONGER per-pass suppression since it compounds
multiplicatively, similar to biological synaptic depletion curves.

SAME ZERO-PASS-1-PPL guarantee (h=0 → gate=1.0 throughout pass 1 with N=512).

PARAMS: log_lambda (1 scalar, init so lambda=0.7)
STATE:  _buf (N,D), _hits (N,) int, _mask (N,) bool, _ptr int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU242(nn.Module):
    """Exponential hit count gate: gate = exp(-lambda * h)."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_lambda = nn.Parameter(torch.tensor(math.log(0.7)))  # lambda=0.7

        self._buf:  torch.Tensor = None
        self._hits: torch.Tensor = None
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
        lam = self.log_lambda.exp().clamp(0.1, 3.0)

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
            return y

        # ── Nearest slot ──────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            h           = self._hits[nearest_idx].item()

        # gate = exp(-lambda * h)
        gate = math.exp(-lam.item() * h)
        output = y * gate

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._hits[nearest_idx] += 1
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._hits[self._ptr] = 0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
