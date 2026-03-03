"""GELU172 – Sequence-Local vs Cross-Batch EMA Contrast Gate.

THE KEY INSIGHT:
    gelu80 computes surprise as deviation from the CROSS-BATCH EMA mean (long history).
    Within any single sequence, there's also a LOCAL trend: the sequence-level mean.

    These two reference points reveal different types of novelty:
    1. EMA-surprise: token differs from what this FF layer has historically seen
       (macro-scale novelty: rare word, unusual construct)
    2. Seq-surprise: token differs from the current sequence's own mean
       (micro-scale novelty: locally unusual word within this passage)

    The KEY INSIGHT is in their INTERACTION:
    - High EMA-surprise, Low Seq-surprise: globally unusual but fits local context
      → the word is rare globally but belongs to the current topic
    - Low EMA-surprise, High Seq-surprise: globally common but weird in this passage
      → a sudden context break (common word in wrong context)
    - High both: globally rare AND unusual in this passage → MOST novel

    This dual-reference system gives finer discrimination than either alone.

    Gate = 1 + alpha * tanh(sigma_global * ema_surprise + sigma_local * seq_surprise)

    This is an ADDITIVE combination of both signals, with separate learned sensitivities.

IMPLEMENTATION:
    EMA cross-batch statistics: _ema_mean (D,), _ema_sq (D,)
    Sequence statistics (computed per forward, no state needed):
        seq_mean_b = x.mean(1)          (B, D) sequence mean per batch item
        seq_sq_b   = x.pow(2).mean(1)   (B, D)
        seq_std_b  = sqrt(seq_sq - seq_mean²)

    EMA z-score: z_ema[b,t,d] = (x - ema_mean) / ema_std
    Seq z-score: z_seq[b,t,d] = (x - seq_mean_b) / seq_std_b

    surp_ema = mean_d(|z_ema|).mean()   → scalar
    surp_seq = mean_d(|z_seq|).mean()   → scalar

    gate = 1 + alpha * tanh(sigma_g * surp_ema + sigma_l * surp_seq)

CAUSALITY: For seq_mean, we use the FULL sequence mean (not causal) — same as gelu138/152.
EMA updated after forward. ✓

Params: log_alpha, log_sigma_g, log_sigma_l (3 scalars).
State: _ema_mean (D,), _ema_sq (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
EPS_VAR   = 1e-4


class GELU172(nn.Module):
    """Dual-reference contrast: global EMA surprise + local sequence surprise."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_g = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # global
        self.log_sigma_l = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # local

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out     = self._gelu(x)
        alpha   = self.log_alpha.exp()
        sigma_g = F.softplus(self.log_sigma_g)
        sigma_l = F.softplus(self.log_sigma_l)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # ── Global EMA z-score ──────────────────────────────────────────
            var_g  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=EPS_VAR)
            std_g  = var_g.sqrt()                         # (D,)
            z_g    = (x.detach() - self._ema_mean.view(1,1,D)) / (std_g.view(1,1,D) + 1e-5)
            surp_g = z_g.abs().mean(dim=-1).mean()        # scalar

            # ── Sequence-local z-score ──────────────────────────────────────
            x_d     = x.detach()
            seq_mu  = x_d.mean(1, keepdim=True)          # (B, 1, D)
            seq_sq  = x_d.pow(2).mean(1, keepdim=True)   # (B, 1, D)
            seq_var = (seq_sq - seq_mu.pow(2)).clamp(min=EPS_VAR)
            seq_std = seq_var.sqrt()                       # (B, 1, D)
            z_l     = (x_d - seq_mu) / (seq_std + 1e-5)  # (B, T, D)
            surp_l  = z_l.abs().mean(dim=-1).mean()       # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma_g * surp_g + sigma_l * surp_l)
        output = out * gate

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = EMA_DECAY * self._ema_mean + (1 - EMA_DECAY) * xf.mean(0)
            self._ema_sq   = EMA_DECAY * self._ema_sq   + (1 - EMA_DECAY) * xf.pow(2).mean(0)

        return output
