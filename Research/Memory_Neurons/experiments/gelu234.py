"""GELU234 – Self-Detecting Pass-2 Gate (Soft Exp Gate, Zero Pass-1 Effect).

IDENTICAL STRUCTURE TO gelu233 but uses a soft exponential gate in pass 2
instead of a hard sigmoid. This tests whether the smoother gate is better
or worse at adaptation without affecting pass-1 PPL.

PASS-1 BEHAVIOR:
    gate = 1.0 throughout. Pass-1 PPL = control baseline exactly.

PASS-2 DETECTION:
    Same mechanism as gelu233: cos_sim > DETECT_THRESH (0.88) → freeze + gate.

PASS-2 GATE FORMULA (soft exp, like gelu54 but with larger fixed tau):
    gate = (1 - alpha) + alpha * exp(-tau * tok_sim)

    With tau=5.0 and alpha=0.6 (both fixed, since they receive no training gradient):
        tok_sim = 0.0 (orthogonal):  gate ≈ 1.0
        tok_sim = 0.5 (moderate):    gate ≈ 0.4 + 0.6*exp(-2.5) = 0.4 + 0.049 = 0.45
        tok_sim = 0.9 (very familiar): gate ≈ 0.4 + 0.6*exp(-4.5) = 0.4 + 0.007 = 0.41

    The soft gate gives more graded suppression than the hard sigmoid.
    Advantage: less risk of over-suppression on borderline-familiar content.
    Trade-off: less sharp differentiation between novel and familiar.

COMPARISON WITH gelu233:
    gelu233 (hard sigmoid): binary-like switch — very similar batches get gate_min,
                            others mostly unaffected.
    gelu234 (soft exp):     continuous — all batches with some familiarity are
                            partially suppressed proportionally.

Both should give zero pass-1 PPL change and large Δ1→3.

PARAMS: log_tau, log_blend (2 scalars, same as gelu54 but different inits)
STATE:  ring buffer (N=512, D), mask (N,), pointer (int),
        _pass2 (bool), _frozen (bool)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.88   # cos_sim threshold for pass-2 detection


class GELU234(nn.Module):
    """Self-detecting pass-2 soft gate: zero effect on pass-1 PPL."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready  = False
        self._frozen = False
        self._pass2  = False

        # tau and alpha are "frozen at init" params (no gradient during training
        # since gate=1.0 always → they act as fixed hyperparameters at eval)
        # tau = 5.0 (large → aggressive suppression for familiar tokens)
        # alpha = 0.6 (high → gate range is [0.4, 1.0], meaningful suppression)
        self.log_tau   = nn.Parameter(torch.tensor(math.log(5.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.6 / 0.4)))  # alpha ≈ 0.6

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
            return y   # gate = 1.0

        # ── Compute nearest episode (always needed for detection) ──────
        with torch.no_grad():
            m_norm      = F.normalize(m_curr.unsqueeze(0), dim=-1)        # (1, D)
            buf_n       = F.normalize(self._buf, dim=-1)                  # (N, D)
            sims        = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            nearest_vec = self._buf[nearest_idx].clone()                  # (D,)

        # Detect pass 2 — only during eval (not training) ─────────────
        if not self._pass2 and not self.training and max_sim > DETECT_THRESH:
            self._pass2  = True
            self._frozen = True

        # ── Pass 1: gate = 1.0, update buffer ─────────────────────────
        if not self._pass2:
            with torch.no_grad():
                self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
                self._mask[self._ptr] = True
                self._ptr = (self._ptr + 1) % self._N
            return y

        # ── Pass 2+: soft exp gate (buffer frozen) ─────────────────────
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)

        out_n   = F.normalize(y, dim=-1)                                  # (B, T, D)
        nv_n    = F.normalize(nearest_vec.view(1, 1, D), dim=-1)         # (1, 1, D)
        tok_sim = (out_n * nv_n).sum(-1)                                  # (B, T)

        novelty = torch.exp(-tau * tok_sim)                               # (B, T)
        gate    = (1.0 - alpha) + alpha * novelty                         # (B, T)

        return y * gate.unsqueeze(-1)
        # NOTE: buffer not updated in pass 2+
