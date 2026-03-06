"""GELU232 – Hard Sigmoid Gate + Large Buffer (N=512).

COMBINES gelu229 (hard sigmoid for sharp familiarity boundary) with
the large-buffer fix from gelu231 (N=512 covers full test pass).

PROBLEM WITH gelu229 (delta1to3 = +0.023, only slightly below gelu54):
    gelu229 uses the same N=32 buffer as gelu54. Its hard sigmoid gate
    is sharper (near-binary switch at θ), but only 27% of pass-2 batches
    have high-cos_sim matches. The other 73% produce gate ≈ 1.0 regardless.
    Net adaptation still tiny.

FIX: N=32 → N=512 (same reasoning as gelu231).

    At N=512, 100% of pass-2 batches get a high-quality match (cos_sim ≈ 0.9–1.0).
    The hard sigmoid then fires for ALL batches, not just 27%.

    Estimated Δ1→3: ~100% coverage × (gate1 − gate2) per token × PPL effect.
    With N=512, gate fires on all batches:
        Pass 1 (cos~0.4 to training-era buffer):  gate ≈ gate_max          = 1.0
        Pass 2 (cos~0.9 to full test-pass buffer): gate ≈ gate_min          = 0.1
    The difference is ~0.9 → large suppression → large Δ.

PARAMS: log_sharpness, logit_threshold, logit_gate_min (same 3 as gelu229)
STATE:  ring buffer (N=512, D), mask (N,), pointer (int)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU232(nn.Module):
    """Hard sigmoid suppression + large ring buffer for full pass-1 coverage."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

        self.log_sharpness   = nn.Parameter(torch.tensor(math.log(5.0)))
        self.logit_threshold = nn.Parameter(torch.tensor(math.log(0.6 / 0.4)))  # theta ≈ 0.6
        self.logit_gate_min  = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))  # gate_min ≈ 0.2

    def reset_state(self):
        self._buf   = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        sharpness = self.log_sharpness.exp().clamp(1.0, 20.0)
        threshold = torch.sigmoid(self.logit_threshold)                   # in (0, 1)
        gate_min  = 0.05 + 0.45 * torch.sigmoid(self.logit_gate_min)     # in (0.05, 0.5)

        y      = self._gelu(x)                                            # (B, T, D)
        m_curr = y.detach().flatten(0, 1).mean(0)                         # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Nearest-episode lookup ─────────────────────────────────────
        with torch.no_grad():
            m_norm      = F.normalize(m_curr.unsqueeze(0), dim=-1)        # (1, D)
            buf_n       = F.normalize(self._buf, dim=-1)                  # (N, D)
            sims        = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            nearest_vec = self._buf[nearest_idx].clone()                  # (D,)

        # ── Per-token hard sigmoid gate ────────────────────────────────
        y_flat  = y.flatten(0, 1)                                         # (B*T, D)
        y_n     = F.normalize(y_flat, dim=-1)                             # (B*T, D)
        nv_n    = F.normalize(nearest_vec.view(1, D), dim=-1)            # (1, D)
        tok_sim = (y_n * nv_n).sum(-1).view(B, T)                        # (B, T)

        # Hard sigmoid: gate_t = sigmoid(-sharpness * (cos_sim - threshold))
        gate_t      = torch.sigmoid(-sharpness * (tok_sim - threshold))   # (B, T) in (0,1)
        gate_scalar = gate_min + (1.0 - gate_min) * gate_t               # (B, T)
        output      = y * gate_scalar.unsqueeze(-1)

        # ── Update ring buffer ─────────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
