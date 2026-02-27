"""GELU87 – Dual-Sided Per-Channel Sparse Gate (Top-k Novel, Bottom-k Suppressed).

THE LIMITATION OF ALL PRIOR SCALAR GATES:
    Every gate (gelu80, gelu78, gelu85...) applies the SAME scalar multiplier to ALL D channels:
        out[:, :, d] = GELU(x[:, :, d]) × gate_scalar(token)

    This is a UNIFORM multiplicative rescaling of the entire token. The model must
    express novel information using the SAME channel mix as before — just louder or softer.

    In sparse coding theory (Olshausen & Field, 1996), efficient representations use:
    - A SMALL number of active channels (sparse).
    - Each channel carries independent information.
    - Familiar patterns → suppress those specific channels, keep others.

THE FIX — PER-CHANNEL DUAL GATE:
    Compute a per-channel z-score: z_d = (x_d - μ_d) / (σ_d + eps).

    For each channel d in a token (b, t):
    - If |z_d| is HIGH: channel d is "novel" for this token → keep or amplify
    - If |z_d| is LOW : channel d is "familiar" for this token → suppress

    This creates a TOKEN-LEVEL CHANNEL SELECTION:
        gate_d = softplus(|z_d|) / (softplus(mean|z|) + eps)   ← relative z-score gate
               = per-channel "novelty ratio"

    Then blend with baseline:
        final_gate_d = β + (1 - β) × normalize(gate_d)

    Where normalize ensures mean(final_gate_d) = 1 (energy preserving).

    Result: novel channels are amplified, familiar channels suppressed, per-token.
    The CHANNEL MIX changes depending on what's unusual in each token.

ENERGY PRESERVATION:
    We want mean_d(final_gate_d) ≈ 1 so the scale of representations is stable.
    Normalize: gate_d → gate_d / mean(gate_d)
    Then blend with 1: (1-β)×(gate_d/mean(gate_d)) + β×1

    This ensures that, on average across channels, no gain or loss of signal.
    Novel channels borrow "energy" from familiar channels.

WHY THIS CAN WORK:
    In the transformer FFN, D=1024 channels. Learning assigns specific channels
    to specific features (syntactic roles, named entities, part-of-speech, etc.).
    When a token is a named entity (unusual), the "name" channels fire strongly
    → high z-score in those channels → they get amplified by our gate.
    The common word channels (low z-score) get suppressed.
    Result: OUTPUT is SPARSER, more DISCRIMINATIVE, and represents what's NOVEL
    about this token better than a scalar gate ever could.

Params: logit_decay, log_sigma_raw, log_beta_raw = 3 scalars.
State: _ema_mean (D,), _ema_sq (D,) — same as gelu80.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU87(nn.Module):
    """Per-channel dual-sided gate: novel channels amplified, familiar channels suppressed."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # β: blend of raw gate vs. identity (β=1 → all identity = no effect)
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_raw  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # β≈0.5

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        sigma = F.softplus(self.log_sigma_raw)              # kept in grad graph
        alpha = torch.sigmoid(self.log_beta_raw)            # kept in grad graph

        out = self._gelu(x)   # (B, T, D)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score: detach stats + x, sigma in graph ────────
        mu_  = self._ema_mean.detach().view(1, 1, D)
        var_ = (self._ema_sq.detach() - self._ema_mean.detach().pow(2)).clamp(min=self.eps_var)
        std_ = var_.sqrt().view(1, 1, D)
        z    = (x.detach() - mu_) / (std_ + self.eps)       # (B, T, D) – x detached

        # ── Per-channel novelty gate (sigma + alpha in grad graph) ────────
        gate_raw   = F.softplus(sigma * z.abs())             # (B, T, D) – sigma gets grad
        gate_norm  = gate_raw / (gate_raw.mean(dim=-1, keepdim=True) + self.eps)
        final_gate = alpha * gate_norm + (1.0 - alpha)       # alpha gets grad
        final_gate = final_gate / (final_gate.mean(dim=-1, keepdim=True) + self.eps)

        result = out * final_gate

        # ── EMA updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1.0 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1.0 - d_val) * xf.pow(2).mean(0)

        return result
