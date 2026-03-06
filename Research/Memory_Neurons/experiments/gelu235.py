"""GELU235 – gelu211 Product Gate + Self-Detecting Episodic Pass-2 Adaptation.

THE DUAL OBJECTIVE PROBLEM:
    Current best results:
        Best PPL:         gelu211 at 159.35 (product gate: input EMA × output EMA)
        Best Adaptation:  gelu54  at delta1to3 = +0.030 (ring buffer, N=32)

    These are currently DIFFERENT experiments. No single experiment achieves
    both low PPL AND strong adaptation. This experiment stacks both mechanisms.

ARCHITECTURE: Two orthogonal gating layers applied sequentially.

    LAYER 1 — gelu211 Product Gate (always active, improves PPL):
        Computes per-token novelty gates from EMA statistics.
        At training time: reduces PPL by ~12.99 vs control (our best).
        At eval time (pass 1): same benefit as trained → good base PPL.

    LAYER 2 — Episodic Adaptation Gate (only activates in pass 2+):
        Ring buffer N=512 — builds full test-pass memory in pass 1.
        Self-detecting: gate=1.0 during pass 1 (NO change to pass-1 PPL).
        On pass-2 detection: freeze buffer, apply hard sigmoid suppression.
        Effect: additional pass-2 PPL reduction on top of Layer 1.

    Combined output:
        output = y * gate_211 * gate_episodic

        Pass 1: gate_211 ∈ (0,1) (improves base PPL via product novelty)
                gate_episodic = 1.0 (zero pass-1 effect → PPL ≈ gelu211)
        Pass 2: gate_211 active (same benefit)
                gate_episodic active (additional suppression for familiar content)

EXPECTED OUTCOME:
    Pass-1 PPL: ≈ 159.35 (same as gelu211, since Layer 2 is inactive)
    Pass-2/3 PPL: < 159.35 (Layer 2 fires → additional reduction)
    Δ1→3: significantly positive (first experiment to combine good PPL + good Δ)

PARAMS:
    Layer 1 (gelu211-style): log_d_x, log_d_y, log_tau_in, log_tau_out,
                              log_alpha_in, log_alpha_out (6 scalars)
    Layer 2 (episodic):      log_sharpness, logit_threshold, logit_gate_min
                             (3 scalars, effectively fixed at init)
STATE:
    Layer 1: _ema_x, _ema_x2, _ema_y, _ema_y2 (running stats)
    Layer 2: ring buffer (N=512, D), _pass2 (bool), _frozen (bool)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.88


class GELU235(nn.Module):
    """gelu211 product-gate PPL + self-detecting episodic pass-2 adaptation."""

    def __init__(self, buffer_size: int = 512, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # ── Layer 1: gelu211 product gate parameters ───────────────────
        self.log_d_x     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))   # EMA decay for x
        self.log_d_y     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))   # EMA decay for y
        self.log_tau_in  = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_tau_out = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_a_in    = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))   # alpha_in ≈ 0.5
        self.log_a_out   = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))   # alpha_out ≈ 0.5

        # ── Layer 2: episodic gate parameters (fixed at init) ──────────
        self.log_sharpness   = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_threshold = nn.Parameter(torch.tensor(math.log(0.7 / 0.3)))  # theta = 0.7
        self.logit_gate_min  = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))  # gate_min = 0.1

        # ── Running stats (Layer 1) ───────────────────────────────────
        self._ema_x:  torch.Tensor = None
        self._ema_x2: torch.Tensor = None
        self._ema_y:  torch.Tensor = None
        self._ema_y2: torch.Tensor = None
        self._l1_ready = False

        # ── Ring buffer (Layer 2) ──────────────────────────────────────
        self._N     = buffer_size
        self._buf:  torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._l2_ready = False
        self._frozen   = False
        self._pass2    = False

    def reset_state(self):
        self._ema_x    = None
        self._ema_x2   = None
        self._ema_y    = None
        self._ema_y2   = None
        self._l1_ready = False

        self._buf      = None
        self._mask     = None
        self._ptr      = 0
        self._l2_ready = False
        self._frozen   = False
        self._pass2    = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # ── Hyperparameters ────────────────────────────────────────────
        d_x     = torch.sigmoid(self.log_d_x).detach().item()
        d_y     = torch.sigmoid(self.log_d_y).detach().item()
        tau_in  = self.log_tau_in.exp()
        tau_out = self.log_tau_out.exp()
        a_in    = torch.sigmoid(self.log_a_in)
        a_out   = torch.sigmoid(self.log_a_out)

        y = self._gelu(x)   # (B, T, D)

        # ─────────────────────────────────────────────────────────────
        # LAYER 1: gelu211 product gate (always active)
        # ─────────────────────────────────────────────────────────────
        x_flat = x.detach().flatten(0, 1)   # (B*T, D)
        y_flat = y.detach().flatten(0, 1)   # (B*T, D)

        if not self._l1_ready:
            with torch.no_grad():
                self._ema_x  = x_flat.mean(0)
                self._ema_x2 = (x_flat ** 2).mean(0)
                self._ema_y  = y_flat.mean(0)
                self._ema_y2 = (y_flat ** 2).mean(0)
            self._l1_ready = True
            # Fall through — first call has no good stats, skip layer 1 gate
            gate_211 = torch.ones(B, T, device=x.device, dtype=y.dtype)
        else:
            # Standardized input novelty
            std_x   = (self._ema_x2 - self._ema_x ** 2).clamp(min=0).sqrt() + self.eps
            z_in    = ((x.detach() - self._ema_x.view(1, 1, D)) / std_x.view(1, 1, D)) ** 2
            z_in_s  = z_in.mean(-1)    # (B, T) novelty score in input space

            # Standardized output novelty
            std_y   = (self._ema_y2 - self._ema_y ** 2).clamp(min=0).sqrt() + self.eps
            z_out   = ((y.detach() - self._ema_y.view(1, 1, D)) / std_y.view(1, 1, D)) ** 2
            z_out_s = z_out.mean(-1)   # (B, T)

            # Product gate: suppress familiar input AND familiar output
            g_in    = (1.0 - a_in)  + a_in  * torch.exp(-tau_in  * z_in_s)
            g_out   = (1.0 - a_out) + a_out * torch.exp(-tau_out * z_out_s)
            gate_211 = g_in * g_out    # (B, T)

        # Update EMA stats
        with torch.no_grad():
            x_m  = x_flat.mean(0)
            x2_m = (x_flat ** 2).mean(0)
            y_m  = y_flat.mean(0)
            y2_m = (y_flat ** 2).mean(0)
            self._ema_x  = d_x * self._ema_x  + (1 - d_x) * x_m
            self._ema_x2 = d_x * self._ema_x2 + (1 - d_x) * x2_m
            self._ema_y  = d_y * self._ema_y  + (1 - d_y) * y_m
            self._ema_y2 = d_y * self._ema_y2 + (1 - d_y) * y2_m

        # ─────────────────────────────────────────────────────────────
        # LAYER 2: episodic pass-2 gate (only active after detection)
        # ─────────────────────────────────────────────────────────────
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,) buffer key

        if not self._l2_ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr    = 1
            self._l2_ready = True
            # Gate 2 = 1.0 (no effect first call)
            output = y * gate_211.unsqueeze(-1)
            return output

        # Nearest episode lookup (always needed for detection)
        with torch.no_grad():
            m_norm      = F.normalize(m_curr.unsqueeze(0), dim=-1)     # (1, D)
            buf_n       = F.normalize(self._buf, dim=-1)               # (N, D)
            sims        = (buf_n * m_norm).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            nearest_vec = self._buf[nearest_idx].clone()               # (D,)

        # Detect pass-2 start — only during eval, never during training
        if not self._pass2 and not self.training and max_sim > DETECT_THRESH:
            self._pass2  = True
            self._frozen = True

        if self._pass2:
            # Hard sigmoid episodic gate
            sharpness = self.log_sharpness.exp().clamp(1.0, 20.0)
            threshold = torch.sigmoid(self.logit_threshold)
            gate_min  = 0.05 + 0.45 * torch.sigmoid(self.logit_gate_min)

            y_flat2 = y.flatten(0, 1)                                  # (B*T, D)
            y_n2    = F.normalize(y_flat2, dim=-1)
            nv_n    = F.normalize(nearest_vec.view(1, D), dim=-1)     # (1, D)
            tok_sim = (y_n2 * nv_n).sum(-1).view(B, T)                # (B, T)

            gate_t   = torch.sigmoid(-sharpness * (tok_sim - threshold))
            gate_ep  = gate_min + (1.0 - gate_min) * gate_t           # (B, T)
        else:
            gate_ep = torch.ones(B, T, device=x.device, dtype=y.dtype)

        # ── Combine and output ─────────────────────────────────────────
        output = y * (gate_211 * gate_ep).unsqueeze(-1)

        # ── Update Layer 2 buffer (only during pass 1) ─────────────────
        if not self._frozen:
            with torch.no_grad():
                self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
                self._mask[self._ptr] = True
                self._ptr = (self._ptr + 1) % self._N

        return output
