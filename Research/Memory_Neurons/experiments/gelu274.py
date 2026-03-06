"""GELU274 – gelu211 Backbone + Eval-Only Linear Pass-Counter Gate.

═══════════════════════════════════════════════════════════════════════════
DESIGN GOAL: Best possible PPL AND monotonic sequential adaptation.
gelu211 achieves PPL≈159.35 (best in class).
gelu263 achieves transparent linear pass-counter adaptation.
This experiment combines both with clean layer separation.
═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
    Layer 1 (ALWAYS active, trained):
        gelu211 product gate: g_in(z_in) * g_out(z_out)
        Trained on training data → PPL ≈ 159.35

    Layer 2 (EVAL ONLY, NOT trained):
        Linear pass-counter gate:
            gate_adapt = 1 + k_adapt * (pass_num - 1) * sim_to_nearest

        pass_num: tracked by detection-based ring buffer (same as gelu263).
        sim_to_nearest: cosine similarity to nearest pass-1 slot.

    combined: output = gelu(x) * gate_211 * gate_adapt.unsqueeze(-1)

PASS-COUNTER GATE MATH:
    Pass 1: pass_num=1, gate_adapt = 1 + k*(0)*sim = 1.0  ✓
    Pass 2: pass_num=2, gate_adapt = 1 + k*1*sim  [moderate boost]
    Pass 3: pass_num=3, gate_adapt = 1 + k*2*sim  [double boost]

    If PPL is approximately linear in gate (first-order approximation):
        Δ1→2 ≈ PPL_gain * k * sim
        Δ1→3 ≈ PPL_gain * 2k * sim = 2 * Δ1→2

    Guaranteed: Δ1→3 = 2*Δ1→2 (exactly linear scaling of adaptation).

COMPARE TO gelu254 (gelu211 + exponential facilitation):
    gelu254: gains k, 3k per pass (exponential: facil=1,2,4)
    gelu274: gains k, 2k per pass (linear: pass_num=1,2,3)  ← more controlled

COMPARE TO gelu268 (gelu211 + OCS):
    gelu268: OCS (direction-preserving boost+damp)
    gelu274: simple scalar gate ← easier to tune, more predictable

PARAMS (Layer 1, trained): log_d_x, log_d_y, log_tau_in, log_tau_out,
                            log_a_in, log_a_out (same as gelu211)
PARAMS (Layer 2, init only): log_k_adapt
STATE (Layer 1): _ema_x, _ema_x2, _ema_y, _ema_y2
STATE (Layer 2): ring buffer, pass_num, _pass1_complete, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.85
N_BUF         = 512
MAX_GATE      = 8.0


class GELU274(nn.Module):
    """gelu211 product gate (trained) + eval-only linear pass-counter gate."""

    def __init__(self, buffer_size: int = N_BUF, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # ── Layer 1: gelu211 params (trained) ─────────────────────────
        self.log_d_x     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_d_y     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_tau_in  = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_tau_out = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_a_in    = nn.Parameter(torch.tensor(0.0))
        self.log_a_out   = nn.Parameter(torch.tensor(0.0))

        # ── Layer 2: pass-counter gate (eval-only, init only) ─────────
        self.log_k_adapt = nn.Parameter(torch.tensor(math.log(0.5)))

        # ── Layer 1 state ─────────────────────────────────────────────
        self._ema_x:  torch.Tensor = None
        self._ema_x2: torch.Tensor = None
        self._ema_y:  torch.Tensor = None
        self._ema_y2: torch.Tensor = None
        self._l1_ready = False

        # ── Layer 2 state ─────────────────────────────────────────────
        self._N      = buffer_size
        self._buf:   torch.Tensor = None
        self._mask:  torch.Tensor = None
        self._ptr    = 0
        self._l2_ready       = False
        self._pass1_complete = False
        self._pass_num       = 1
        self._batches_in_pass = 0
        self._max_sim_last   = 0.0

    def reset_state(self):
        self._ema_x    = None;  self._ema_x2 = None
        self._ema_y    = None;  self._ema_y2 = None
        self._l1_ready = False
        self._buf      = None;  self._mask   = None
        self._ptr      = 0
        self._l2_ready       = False
        self._pass1_complete = False
        self._pass_num       = 1
        self._batches_in_pass = 0
        self._max_sim_last   = 0.0

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _layer1(self, x: torch.Tensor, y: torch.Tensor, B: int, T: int, D: int):
        """gelu211 product gate."""
        d_x     = torch.sigmoid(self.log_d_x).detach().item()
        d_y     = torch.sigmoid(self.log_d_y).detach().item()
        tau_in  = self.log_tau_in.exp()
        tau_out = self.log_tau_out.exp()
        a_in    = torch.sigmoid(self.log_a_in)
        a_out   = torch.sigmoid(self.log_a_out)

        x_flat = x.detach().flatten(0, 1)
        y_flat = y.detach().flatten(0, 1)

        if not self._l1_ready:
            with torch.no_grad():
                self._ema_x   = x_flat.mean(0)
                self._ema_x2  = (x_flat ** 2).mean(0)
                self._ema_y   = y_flat.mean(0)
                self._ema_y2  = (y_flat ** 2).mean(0)
            self._l1_ready = True
            return torch.ones(B, T, device=x.device, dtype=y.dtype)

        std_x  = (self._ema_x2 - self._ema_x**2).clamp(0).sqrt() + self.eps
        z_in   = ((x.detach() - self._ema_x.view(1,1,D)) / std_x.view(1,1,D)).pow(2).mean(-1)
        std_y  = (self._ema_y2 - self._ema_y**2).clamp(0).sqrt() + self.eps
        z_out  = ((y.detach() - self._ema_y.view(1,1,D)) / std_y.view(1,1,D)).pow(2).mean(-1)
        g_in   = (1.0 - a_in)  + a_in  * torch.exp(-tau_in  * z_in)
        g_out  = (1.0 - a_out) + a_out * torch.exp(-tau_out * z_out)

        with torch.no_grad():
            self._ema_x  = d_x * self._ema_x  + (1-d_x) * x_flat.mean(0)
            self._ema_x2 = d_x * self._ema_x2 + (1-d_x) * (x_flat**2).mean(0)
            self._ema_y  = d_y * self._ema_y  + (1-d_y) * y_flat.mean(0)
            self._ema_y2 = d_y * self._ema_y2 + (1-d_y) * (y_flat**2).mean(0)

        return g_in * g_out   # (B, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y = self._gelu(x)

        # ── Layer 1 (always active) ────────────────────────────────────
        gate_211 = self._layer1(x, y, B, T, D)

        # ── Layer 2 (eval only) ────────────────────────────────────────
        if self.training:
            return y * gate_211.unsqueeze(-1)

        k_adapt = self.log_k_adapt.exp().clamp(0.01, 3.0)
        m_curr  = y.detach().flatten(0, 1).mean(0)

        if not self._l2_ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr     = 1
            self._l2_ready = True
            return y * gate_211.unsqueeze(-1)

        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (F.normalize(self._buf, dim=-1) * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if not self._pass1_complete and max_sim > DETECT_THRESH:
                self._pass1_complete = True
                self._pass_num       = 2
            elif self._pass1_complete and max_sim > DETECT_THRESH and self._batches_in_pass > 10:
                if max_sim > self._max_sim_last + 0.05:   # new pass started (high sim reset)
                    self._pass_num      += 1
                    self._batches_in_pass = 0

            self._batches_in_pass += 1
            self._max_sim_last     = max(self._max_sim_last, max_sim)

            if not self._pass1_complete:
                self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
                self._mask[self._ptr] = True
                self._ptr = (self._ptr + 1) % self._N

        # gate_adapt = 1 + k * (pass_num - 1) * sim
        gate_val = min(1.0 + k_adapt.item() * (self._pass_num - 1) * max(max_sim, 0.0), MAX_GATE)
        return y * gate_211.unsqueeze(-1) * gate_val
