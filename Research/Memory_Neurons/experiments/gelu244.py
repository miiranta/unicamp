"""GELU244 – Depletion with Learnable Rate and Floor.

EXTENDS gelu239 by making BOTH the depletion rate AND the gate floor learnable,
allowing the model to discover the optimal depletion curve during training.

In gelu239, depletion_rate=0.5 and gate_floor=0.0 are fixed hyperparameters.
Here they are learned — with the crucial constraint that training still gives
conservative values (gate≈1.0 during training since FIRE_THRESH=0.85 rarely
triggers on training data).

GATE FORMULA:
    gate = floor + (1 - floor) * exp(-k * (1 - depl_level))
    
    depl_level=1.0 (fresh):  gate = floor + (1-floor) * 1.0 = 1.0   ← always
    depl_level=0.5 (fired):  gate = floor + (1-floor) * exp(-k/2)
    depl_level → 0 (depleted): gate → floor

DEPLETION UPDATE:
    On each firing: depl *= depl_rate  (depl_rate ∈ [0.1, 0.9])

PARAMETERS:
    log_k:          gate strength (init k=2.0)
    logit_depl_rate: depletion rate (init 0.5, constrained ∈ [0.1, 0.9] via sigmoid scaling)
    logit_floor:    minimum gate value (init 0.1, constrained ∈ [0.0, 0.5])

During training (FIRE_THRESH rarely triggers), these params get near-zero gradient
and stay near their initialization. At test time they act as fixed hyperparameters.

The floor parameter is important: if floor=0.0 (gelu239), very depleted slots give
gate≈0 which may be too aggressive. A learned floor gives stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU244(nn.Module):
    """Depletion gate with learnable rate and floor for stable convergence."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size

        self.log_k          = nn.Parameter(torch.tensor(math.log(2.0)))
        # depletion_rate = 0.1 + 0.8 * sigmoid(logit) ∈ [0.1, 0.9]
        self.logit_depl_rate = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 → rate=0.5
        # floor = 0.5 * sigmoid(logit_floor) ∈ [0, 0.5]
        self.logit_floor     = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))  # ≈0.1

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
        k_gate    = self.log_k.exp().clamp(0.1, 8.0)
        depl_rate = 0.1 + 0.8 * torch.sigmoid(self.logit_depl_rate)
        floor_val = 0.5 * torch.sigmoid(self.logit_floor)           # ∈ [0, 0.5]

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

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

        # ── Nearest slot ──────────────────────────────────────────────
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()

        # gate = floor + (1 - floor) * exp(-k * (1 - depl))
        raw_gate = math.exp(-k_gate.item() * (1.0 - depl_level))
        gate = floor_val + (1.0 - floor_val) * raw_gate
        output = y * gate

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= depl_rate.item()
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._depl[self._ptr] = 1.0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
