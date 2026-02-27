"""GELU86 – Within-Sequence Causal Z-Score (No Global State).

CORE IDEA:
    All prior experiments track statistics via cross-batch EMA — the "average" token
    across many batches becomes the familiarity reference.

    But natural language has LOCAL structure: within a passage, a word is "familiar"
    if it's being used in the SAME WAY as other words in the same context.
    Cross-batch EMA mixes very different contexts (sports articles + poetry + code).

    Within-sequence causal z-score: for each token at position t, its deviation is
    measured relative to positions 0..t-1 IN THE SAME SEQUENCE.

    mu_t  = (1/t) × sum_{s<t} x_s       (causal cumulative mean)
    var_t = (1/t) × sum_{s<t} (x_s - mu_t)²  (causal cumulative variance)
    z_t   = (x_t - mu_{t-1}) / (std_{t-1} + eps)    per channel

    surp_t = tanh(σ × mean_d(|z_t,d|))
    gate_t = exp(-τ × cos(out_t, mean_out_{t-1})) × (1 + w × surp_t)

PROPERTIES:
    - Zero cross-batch state (self._ready not needed across sequences)
    - Purely local: novelty relative to the CURRENT context, not the global average
    - Causal: only uses positions strictly before t (safe for language modeling)
    - Differentiable: all ops are differentiable through the gate

IMPLEMENTATION:
    Cumulative mean along sequence dim using torch.cumsum:
        cum_x   = cumsum(x, dim=1)                          (B, T, D)
        counts  = arange(1, T+1).view(1, T, 1)              (1, T, 1)
        mu      = cum_x / counts                             (B, T, D) — but mean at t includes t
    
    For CAUSAL mean at step t (mean of x[0..t-1]):
        padded  = cat([zeros(B,1,D), cum_x[:, :-1, :]], dim=1)  ← shift right by 1
        mu_causal  = padded / count_shifted                       (B, T, D), pos 0 gets zeros
    
    Second moment: same trick for cum_sq.
    Position 0 has no prior context → treated as z=0 (no suppression), fully novel.

WHY THIS CAN BEAT GELU80:
    gelu80's EMA reference is the global average across all training batches.
    In a long training run, this average becomes very stable and represents the
    "typical" token in WikiText-2 overall.

    Causal within-seq z-score: the reference changes with EVERY sequence.
    If a sequence is about biology, the reference adapts to biology vocabulary.
    Repeated use of "cell" in the same paragraph → low causal z (familiar in context).
    A sudden appearance of "mitosis" → high causal z (novel in context).
    This is SEMANTICALLY RICHER novelty than global average deviation.

Params: log_tau, log_sigma_raw, log_w_raw = 3 scalars. State: none!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU86(nn.Module):
    """Within-sequence causal z-score gate. No persistent cross-batch state."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

    def reset_state(self):
        pass  # stateless — nothing to reset

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)   # (B, T, D)

        with torch.no_grad():
            # ── Causal cumulative mean (shifted) ────────────────────────────
            cum_x      = torch.cumsum(x.detach(), dim=1)              # (B, T, D)
            cum_xsq    = torch.cumsum(x.detach().pow(2), dim=1)       # (B, T, D)

            # Causal: mean of x[0..t-1], so shift by 1 position
            zeros_1    = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            mu_causal  = torch.cat([zeros_1, cum_x[:, :-1, :]], dim=1)   # (B, T, D)
            sq_causal  = torch.cat([zeros_1, cum_xsq[:, :-1, :]], dim=1) # (B, T, D)

            counts     = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
            # At position 0, count=0 causes divide-by-zero → clamp to 1 (which gives 0/1=0)
            counts_safe = counts.clamp(min=1.0)

            mu    = mu_causal  / counts_safe    # (B, T, D)  causal mean
            sq_m  = sq_causal  / counts_safe    # (B, T, D)  causal mean of x²
            var   = (sq_m - mu.pow(2)).clamp(min=self.eps_var)
            std   = var.sqrt()                  # (B, T, D)

            # ── Per-channel causal z-score ────────────────────────────────
            z          = (x.detach() - mu) / (std + self.eps)         # (B, T, D)
            mean_abs_z = z.abs().mean(dim=-1)                          # (B, T)
            # Position 0 has count=1 (count_safe=1) but mu=0, std≈sqrt(eps) → z can be large
            # Mask out position 0 (no prior context, no suppression)
            mask_1d           = (counts.squeeze(-1) > 1).float()       # (1, T)
            mean_abs_z_masked = mean_abs_z * mask_1d                   # (B, T)

            # ── Causal cosine: out[t] vs mean(out[0..t-1]) ────────────────
            cum_out  = torch.cumsum(out.detach(), dim=1)               # (B, T, D)
            zeros_o  = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            mu_out   = torch.cat([zeros_o, cum_out[:, :-1, :]], dim=1) # (B, T, D)
            mu_out   = mu_out / counts_safe

            out_n    = F.normalize(out.detach(), dim=-1)               # (B, T, D)
            mu_out_n = F.normalize(mu_out, dim=-1)                     # (B, T, D)
            cos      = (out_n * mu_out_n).sum(dim=-1) * mask_1d        # (B, T)

        # ── Gate: sigma and w remain in grad graph ────────────────────────
        surp = torch.tanh(sigma * mean_abs_z_masked)                   # sigma gets grad
        gate = torch.exp(-tau * cos) * (1.0 + w * surp)               # (B, T)
        return out * gate.unsqueeze(-1)
