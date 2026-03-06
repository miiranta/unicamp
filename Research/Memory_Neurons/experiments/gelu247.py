"""GELU247 – Aggressive Depletion (depletion_rate=0.2, faster collapse).

VARIANT OF gelu239 with much faster depletion:
    gelu239: depletion_rate=0.5 → after 1 firing: depl=0.50 → gate≈0.37
    gelu247: depletion_rate=0.2 → after 1 firing: depl=0.20 → gate≈0.13

This tests whether MORE AGGRESSIVE depletion produces larger Δ without
catastrophically degrading pass-2/3 PPL.

MOTIVATION:
    With depl_rate=0.5, pass-2 gate ≈ 0.37 (k=2.0). This should already be
    very strong (63% suppression). But maybe the model can still predict well
    even with 87% suppression (depl_rate=0.2).

    If the model is robust to near-zero gating (as gelu229 with gate_min=0.2
    suggests — it still had PPL 164.3 with hard suppression), then aggressive
    depletion could give Δ1→3 of several PPL points.

KEY DIFFERENCE FROM gelu239: depletion_rate HYPERPARAMETER only (not learned).
    gelu239: depletion_rate=0.5 (moderate)
    gelu247: depletion_rate=0.2 (aggressive, 80% depletion per firing)

If aggressive depletion hurts PPL too much but gives large Δ, we can tune
depletion_rate to balance. This experiment gives us a second data point.

GATE FORMULA (same as gelu239):
    gate = exp(-k * (1 - depl))
    Pass-1 (depl=1.0): gate = 1.0
    Pass-2 (depl=0.20): gate = exp(-k*0.8) ≈ 0.13 with k=2.0
    Pass-3 (depl=0.04): gate = exp(-k*0.96) ≈ 0.06

PARAMS: log_k (1 scalar)
STATE:  same as gelu239
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU247(nn.Module):
    """Aggressive depletion (rate=0.2): near-zero gate after first firing."""

    def __init__(self, buffer_size: int = 512, depletion_rate: float = 0.2):
        super().__init__()
        self._N  = buffer_size
        self._DR = depletion_rate   # 80% depletion per firing

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
        m_curr = y.detach().flatten(0, 1).mean(0)

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

        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()

        gate = math.exp(-k_gate.item() * (1.0 - depl_level))
        output = y * gate

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= self._DR
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._depl[self._ptr] = 1.0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
