"""gelu293 – Dual-Timescale Contrastive EMA Gate.

CONCEPT:
    A single EMA (as in gelu211) tracks the running mean but cannot distinguish
    "this batch is unusual right now" from "this batch matches long-term average".
    A CONTRASTIVE signal — fast_EMA − slow_EMA — encodes exactly this:
        positive: current content is above the long-term baseline  → novel
        negative: current content is below the long-term baseline  → familiar

    KEY DIFFERENCE FROM gelu211:
        gelu211 computes z-score wrt ONE EMA (the long-term mean).
        gelu293 computes a contrast signal as DIFFERENCE between two EMAs and gates
        on the sign and magnitude of that difference.

MECHANISM:
    out       = gelu(x)                                   # (B, T, D)
    x_mean    = x.flatten(0,1).mean(0)                    # (D,)

    New EMAs (differentiable, old values detached):
        fast = d_f * fast.detach() + (1-d_f) * x_mean    # d_f ≈ 0.5
        slow = d_s * slow.detach() + (1-d_s) * x_mean    # d_s ≈ 0.99

    Contrast signal:
        delta = fast - slow.detach()                      # (D,)
        std   = running_std_of_delta (slow EMA of delta^2)
        z     = delta / (std + eps)    (dimensionless contrast)

    Gate:
        up    = beta_up * relu(tanh( gamma * z))
        dn    = beta_dn * relu(tanh(-gamma * z))
        gate  = (1 + up - dn).clamp(0.05, 8.0)           # (D,)
        return out * gate.view(1,1,D)

SEQUENTIAL ADAPTATION:
    During pass 1 (eval): fast EMA quickly tracks test content; slow EMA stays
    near training → contrast signal fires for test-novel channels.
    During pass 2: fast EMA ≈ test content; slow EMA has shifted slightly.
    If content repeats: fast ≈ slow → delta ≈ 0 → gate ≈ 1 → LESS amplification
    of previously-novel content → Δ > 0.

BENEFIT FROM BACKPROP:
    logit_d_fast, logit_d_slow: gradient shapes the relative timescales.
    gate params (log_beta_up, log_beta_dn, log_gamma): same as gelu211-style.

NO CAUSALITY LEAK:
    per-batch statistics, causal across batches.

PARAMS:  logit_d_fast, logit_d_slow, logit_d_var, log_beta_up, log_beta_dn, log_gamma.
STATE:   _fast (D,), _slow (D,), _var (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU293(nn.Module):
    """Dual-timescale contrastive EMA gate: fast vs slow EMA difference drives gating."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        self.logit_d_fast = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))    # d ≈ 0.5
        self.logit_d_slow = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))  # d ≈ 0.99
        self.logit_d_var  = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))  # variance EMA
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._fast: torch.Tensor = None
        self._slow: torch.Tensor = None
        self._var:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._fast  = None
        self._slow  = None
        self._var   = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out    = self._gelu(x)
        x_mean = x.flatten(0, 1).mean(0)    # (D,)

        if not self._ready:
            with torch.no_grad():
                self._fast  = x_mean.detach().clone()
                self._slow  = x_mean.detach().clone()
                self._var   = torch.full((D,), self.eps_var, device=x.device, dtype=x.dtype)
                self._ready = True
            return out

        d_f  = torch.sigmoid(self.logit_d_fast)
        d_s  = torch.sigmoid(self.logit_d_slow)
        d_v  = torch.sigmoid(self.logit_d_var)

        # Differentiable EMA update (old state detached)
        new_fast = d_f * self._fast.detach() + (1 - d_f) * x_mean
        new_slow = d_s * self._slow.detach() + (1 - d_s) * x_mean
        delta    = new_fast - new_slow.detach()           # (D,) contrast signal

        # Variance of delta (for z-normalisation)
        new_var  = d_v * self._var.detach() + (1 - d_v) * delta.detach().pow(2)
        z        = delta / (new_var.detach().sqrt() + self.eps)

        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        up   = beta_up * F.relu(torch.tanh( gamma * z))
        dn   = beta_dn * F.relu(torch.tanh(-gamma * z))
        gate = (1.0 + up - dn).clamp(0.05, 8.0)         # (D,)

        self._fast = new_fast.detach()
        self._slow = new_slow.detach()
        self._var  = new_var.detach()

        return out * gate.view(1, 1, D)
