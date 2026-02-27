"""GELU93 – Dual-Timescale Variance Burst Detector.

CORE IDEA: "Surprise Burst" Detection via Variance Ratio
    In neuroscience, a "change detector" neuron fires when the current stimulus
    changes faster than its history. This is encoded by comparing short-term and
    long-term statistics.

    Similarly: a batch where activations have HIGHER VARIANCE than usual is a
    "novelty burst" — new types of information are present.

THE MECHANISM:
    Track TWO EMAs:
        fast_var[d] = EMA(x_d², fast_decay=0.7)  — recent variance (last ~3 steps)
        slow_var[d] = EMA(x_d², slow_decay=0.95) — long-term variance (last ~20 steps)

    Variance burst score per channel:
        burst_ratio_d = fast_var[d] / (slow_var[d] + eps)

    If fast_var >> slow_var: recent batches have HIGHER variance than usual
    → model is seeing "unusual" inputs → surprise burst → AMPLIFY
    If fast_var << slow_var: recent batches have lower variance than usual
    → model is seeing familiar, low-variability inputs → SUPPRESS

    Aggregate to per-token scalar:
        burst_score = mean_d(burst_ratio_d)  ← average across channels
        surp = tanh(σ × (burst_score - 1.0))  ← centered at 0

    Output gate (same structure as gelu80):
        gate = exp(-τ × cos_out) × (1 + w × surp)
        result = GELU(x) × gate

WHY TWO TIMESCALES?
    Single-EMA approaches (gelu80) track ONE timescale.
    But "surprise" is RELATIVE: a token surprising to the 10-step EMA might not be
    surprising to the 100-step EMA, or vice versa.

    Fast/slow ratio captures TEMPORAL CONTRAST:
    - Habitual slow patterns → fast ≈ slow → ratio ≈ 1 → neutral gate
    - Sudden novel episode → fast >> slow → ratio >> 1 → amplify
    - Unusually quiet period → fast << slow → ratio < 1 → suppress

    This is a DERIVATIVE signal: not just "what is surprising" but "is SURPRISE CHANGING?"

    Similar to: predictive coding (Rao & Ballard), surprise signal adaptation,
    EEG alpha suppression followed by gamma oscillation bursts.

FAST AND SLOW DECAY RATES:
    fast_decay = 0.7: window of ~3.3 steps (recent history)
    slow_decay = 0.97: window of ~33 steps (longer history)
    These span a 10x timescale ratio — meaningful temporal contrast.

    But learning may adjust these: logit_fast and logit_slow are both trainable.
    Constraint: fast_decay < slow_decay (enforced via sigmoid ordering).

Params: logit_fast, logit_slow, log_tau, log_sigma_raw, log_w_raw = 5 scalars.
State: _fast_mean (D,), _fast_sq (D,), _slow_mean (D,), _slow_sq (D,), _ema_out (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU93(nn.Module):
    """Dual-timescale variance burst detector: amplify when recent variance exceeds long-term."""

    def __init__(self, fast_decay: float = 0.7, slow_decay: float = 0.97, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_fast  = nn.Parameter(torch.tensor(math.log(fast_decay / (1.0 - fast_decay))))
        self.logit_slow  = nn.Parameter(torch.tensor(math.log(slow_decay / (1.0 - slow_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._fast_mean: torch.Tensor = None   # (D,)
        self._fast_sq:   torch.Tensor = None   # (D,)
        self._slow_mean: torch.Tensor = None   # (D,)
        self._slow_sq:   torch.Tensor = None   # (D,)
        self._ema_out:   torch.Tensor = None   # (D,) unit
        self._ready = False

    def reset_state(self):
        self._fast_mean = None; self._fast_sq = None
        self._slow_mean = None; self._slow_sq = None
        self._ema_out   = None; self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_fast = torch.sigmoid(self.logit_fast).detach().item()
        d_slow = torch.sigmoid(self.logit_slow).detach().item()
        # Enforce fast < slow: remap so that fast is always smaller
        d_fast = min(d_fast, d_slow * 0.9)   # fast can't exceed slow

        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                xm  = xf.mean(0);  xsq = xf.pow(2).mean(0)
                self._fast_mean = xm.clone(); self._fast_sq = xsq.clone()
                self._slow_mean = xm.clone(); self._slow_sq = xsq.clone()
                self._ema_out = F.normalize(of.mean(0), dim=0).clone()
                self._ready = True
            return out

        # ── Variance burst: fast vs slow variance per channel ───────────────
        fast_var = (self._fast_sq.detach() - self._fast_mean.detach().pow(2)).clamp(min=self.eps_var)
        slow_var = (self._slow_sq.detach() - self._slow_mean.detach().pow(2)).clamp(min=self.eps_var)

        # Per-channel burst ratio
        burst_ratio   = fast_var / (slow_var + self.eps)              # (D,)
        burst_score   = burst_ratio.mean()                            # scalar: avg ratio
        # gate on (burst_score - 1): > 0 → amplify, < 0 → suppress
        surp = torch.tanh(sigma * (burst_score - 1.0))                # scalar, sigma in graph

        # ── Output cosine ─────────────────────────────────────────────────
        out_norm = F.normalize(out.detach(), dim=-1)                  # (B, T, D)
        ema_norm = self._ema_out.detach().view(1, 1, D)
        cos_out  = (out_norm * ema_norm).sum(dim=-1)                  # (B, T)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)        # (B, T)
        result = out * gate.unsqueeze(-1)

        # ── EMA updates (two timescales) ───────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            of  = out.detach().flatten(0, 1)
            xm  = xf.mean(0);  xsq = xf.pow(2).mean(0)
            self._fast_mean = d_fast * self._fast_mean + (1 - d_fast) * xm
            self._fast_sq   = d_fast * self._fast_sq   + (1 - d_fast) * xsq
            self._slow_mean = d_slow * self._slow_mean + (1 - d_slow) * xm
            self._slow_sq   = d_slow * self._slow_sq   + (1 - d_slow) * xsq
            self._ema_out   = F.normalize(d_slow * self._ema_out + (1 - d_slow) * of.mean(0), dim=0)

        return result
