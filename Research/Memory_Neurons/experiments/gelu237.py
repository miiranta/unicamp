"""GELU237 – N=512 Soft Training Gate + Hard Eval Pass-2 Gate.

MOTIVATION: Bridge the training-eval gap for gate sharpness.

    gelu231/236 use a soft exp gate both during training AND eval. The model
    learns to interpret moderate gate suppression (gate ∈ [0.5, 1.0]).

    THE PROBLEM: For maximum Δ, we want pass-2 gate to be AS sharp as possible
    while still being interpretable to the model. A gate the model never saw
    during training (e.g. gate=0.05) would be out of distribution → hurt PPL.

    SOLUTION: During training: soft gate (model learns to use it, gate ≈ [0.5–1.0]).
              During eval pass 1: same soft gate (gate ≈ [0.5–1.0], consistent).
              During eval pass 2: HARD SIGMOID GATE fires on detection.

    The hard sigmoid gate is STRICTLY STRONGER than the soft gate for familiar
    tokens (same boundary but sharper transition). Since the model has seen
    "some suppression on familiar tokens" during training, the hard gate at
    pass 2 is an intensification, not a completely alien signal. The model
    can propagate the stronger suppression productively through subsequent layers.

TRAINING BEHAVIOR:
    Soft gate with log_tau=log(4.0), log_blend=0 (alpha=0.5).
    Same as gelu236. During training, buffer fills with training batches.
    No detection runs (self.training=True → detection guarded off).

EVAL PASS 1 AFTER RESET:
    Buffer starts clean. Soft gate fires using cross-batch nearest (lower sim).
    Buffer fills with 98 pass-1 batch means.

EVAL PASS 2 DETECTION AND HARD GATE:
    When cos_sim > 0.88 (first batch of pass 2 = exact match from pass 1):
        _pass2 = True, _frozen = True.
    Hard gate: gate = gate_min + (1 – gate_min) × sigmoid(–sharpness × (tok_sim – θ))
        sharpness=8.0, θ=0.55, gate_min=0.15 (conservative to avoid over-suppression).
    Familiar tokens (tok_sim > 0.55): gate ≈ 0.15 (strong suppression).
    Novel tokens    (tok_sim < 0.55): gate ≈ 1.0  (no suppression).

    In pass 1, familiar tokens get gate ≈ 0.62 (soft).
    In pass 2, familiar tokens get gate ≈ 0.15 (hard).
    CONTRAST: 0.62 → 0.15 = 4× stronger suppression in pass 2.
    This contrast produces a large Δppl.

PARAMS: log_tau, log_blend (soft gate, learned), log_sharpness, logit_threshold,
        logit_gate_min (hard gate, initialize at conservative values, receive
        near-zero grad since hard gate is eval-only and these are never in the
        backward path during training).
STATE:  ring buffer (N=512, D), _pass2 (bool), _frozen (bool)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.88


class GELU237(nn.Module):
    """Soft training gate (tau=4, alpha=0.5) → hard eval pass-2 gate (sharpness=8)."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready  = False
        self._frozen = False
        self._pass2  = False

        # Soft gate params (trained)
        self.log_tau   = nn.Parameter(torch.tensor(math.log(4.0)))
        self.log_blend = nn.Parameter(torch.tensor(0.0))              # alpha ≈ 0.5

        # Hard gate params (conservative init; receive minimal gradient)
        self.log_sharpness   = nn.Parameter(torch.tensor(math.log(8.0)))
        self.logit_threshold = nn.Parameter(torch.tensor(math.log(0.55 / 0.45)))  # θ = 0.55
        self.logit_gate_min  = nn.Parameter(torch.tensor(math.log(0.15 / 0.85)))  # gate_min = 0.15

    def reset_state(self):
        self._buf    = None
        self._mask   = None
        self._ptr    = 0
        self._ready  = False
        self._frozen = False
        self._pass2  = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        out    = self._gelu(x)
        m_curr = out.detach().flatten(0, 1).mean(0)                       # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=out.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return out

        # ── Nearest-episode lookup (always needed) ─────────────────────
        with torch.no_grad():
            m_norm      = F.normalize(m_curr.unsqueeze(0), dim=-1)       # (1, D)
            buf_n       = F.normalize(self._buf, dim=-1)                 # (N, D)
            sims        = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            nearest_vec = self._buf[nearest_idx].clone()                 # (D,)

        # ── Pass-2 detection (eval only) ───────────────────────────────
        if not self._pass2 and not self.training and max_sim > DETECT_THRESH:
            self._pass2  = True
            self._frozen = True

        # ── Compute per-token similarity ───────────────────────────────
        out_n   = F.normalize(out, dim=-1)                               # (B, T, D)
        nv_n    = F.normalize(nearest_vec.view(1, 1, D), dim=-1)        # (1, 1, D)
        tok_sim = (out_n * nv_n).sum(-1)                                 # (B, T)

        # ── Select gate formula based on pass ─────────────────────────
        if self._pass2:
            # Hard sigmoid gate (eval pass 2+)
            sharpness = self.log_sharpness.exp().clamp(1.0, 20.0)
            threshold = torch.sigmoid(self.logit_threshold)
            gate_min  = 0.05 + 0.45 * torch.sigmoid(self.logit_gate_min)  # ∈ (0.05, 0.5)
            gate_t    = torch.sigmoid(-sharpness * (tok_sim - threshold))
            gate      = gate_min + (1.0 - gate_min) * gate_t              # (B, T)
        else:
            # Soft exp gate (training + eval pass 1)
            tau   = self.log_tau.exp()
            alpha = torch.sigmoid(self.log_blend)
            gate  = (1.0 - alpha) + alpha * torch.exp(-tau * tok_sim)     # (B, T)

        output = out * gate.unsqueeze(-1)

        # ── Update ring buffer (not frozen) ───────────────────────────
        if not self._frozen:
            with torch.no_grad():
                self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
                self._mask[self._ptr] = True
                self._ptr = (self._ptr + 1) % self._N

        return output
