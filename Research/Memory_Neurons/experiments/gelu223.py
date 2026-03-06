"""GELU223 – gelu211 with Slow-Decay Output EMA.

MOTIVATION — TWO-SPEED EMA FOR INPUT AND OUTPUT:
    gelu211 (PPL 159.35) uses a SINGLE learned decay for both input and output EMAs.
    The learned decay ends up close to 0.9 (fast, 10-step halflife).

    But the input x and output GELU(x) may benefit from different timescales:
        - Input x: fast EMA (d≈0.9) captures recent training distribution
          → z_in is sensitive to recent context shifts, good for novelty detection
        - Output y=GELU(x): slow EMA (d≈0.99, 100-step halflife) gives a
          STABLE long-run baseline for output novelty
          → z_out measures departure from the long-run average output magnitude

    INTUITION:
        The slow output EMA acts like a "career baseline" for each channel.
        A channel that suddenly activates much higher than its long-run mean
        is genuinely unusual, even if it was also high in recent batches.
        This captures a different notion of novelty than gelu211.

    IMPLEMENTATION:
        d_in  = sigmoid(logit_decay_in)   ← fast, learnable (init ≈ 0.9)
        d_out = sigmoid(logit_decay_out)  ← slow, learnable (init ≈ 0.99)

        Input gate: gelu190-style asymmetric, using d_in EMA stats
        Output gate: gelu211-style symmetric β_out×tanh(γ_out×z_out), using d_out EMA stats
        gate_final = clamp(gate_in × gate_out, 0.05, 10.0)

PARAMS: logit_decay_in, logit_decay_out, log_beta_up, log_beta_dn, log_gamma,
        log_beta_out, log_gamma_out  (7 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean_slow (D,), _ema_out_sq_slow (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU223(nn.Module):
    """gelu211 with separate fast (input) and slow (output) EMA decay rates."""

    def __init__(self, ema_decay_in: float = 0.9, ema_decay_out: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # Separate decays for input and output statistics
        self.logit_decay_in  = nn.Parameter(
            torch.tensor(math.log(ema_decay_in  / (1.0 - ema_decay_in))))
        self.logit_decay_out = nn.Parameter(
            torch.tensor(math.log(ema_decay_out / (1.0 - ema_decay_out))))
        # Input asymmetric arms
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Output symmetric correction (symmetric is sufficient with stable slow EMA)
        self.log_beta_out = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out= nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:          torch.Tensor = None
        self._ema_sq:            torch.Tensor = None
        self._ema_out_mean_slow: torch.Tensor = None
        self._ema_out_sq_slow:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean          = None
        self._ema_sq            = None
        self._ema_out_mean_slow = None
        self._ema_out_sq_slow   = None
        self._ready             = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_in      = torch.sigmoid(self.logit_decay_in).detach().item()
        d_out     = torch.sigmoid(self.logit_decay_out).detach().item()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        y = self._gelu(x)   # (B, T, D)

        # ── Init EMA ─────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                yf = y.detach().flatten(0, 1)
                self._ema_mean          = xf.mean(0).clone()
                self._ema_sq            = xf.pow(2).mean(0).clone()
                self._ema_out_mean_slow = yf.mean(0).clone()
                self._ema_out_sq_slow   = yf.pow(2).mean(0).clone()
            self._ready = True
            return y

        # ── Input z-scores (fast EMA) ────────────────────────────────
        var_in = (self._ema_sq - self._ema_mean.pow(2)).clamp(self.eps_var)
        std_in = var_in.sqrt()
        z_in   = (x - self._ema_mean) / (std_in + self.eps)   # (B, T, D)

        # ── Output z-scores (slow EMA) ───────────────────────────────
        var_out_s = (self._ema_out_sq_slow - self._ema_out_mean_slow.pow(2)).clamp(self.eps_var)
        std_out_s = var_out_s.sqrt()
        z_out     = (y - self._ema_out_mean_slow) / (std_out_s + self.eps)   # (B, T, D)

        # ── Input asymmetric gate ────────────────────────────────────
        up_in  = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_in  = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in = (1.0 + up_in - dn_in).clamp(0.05, 8.0)

        # ── Output symmetric gate (slow baseline) ────────────────────
        gate_out = 1.0 + beta_out * torch.tanh(gamma_out * z_out)   # (B, T, D)
        gate_out = gate_out.clamp(0.05, 8.0)

        # ── Product gate ─────────────────────────────────────────────
        gate = (gate_in * gate_out).clamp(0.05, 10.0)

        output = y * gate

        # ── Update EMAs ──────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            yf = y.detach().flatten(0, 1)
            # Fast input EMA
            self._ema_mean  = d_in  * self._ema_mean  + (1-d_in)  * xf.mean(0)
            self._ema_sq    = d_in  * self._ema_sq    + (1-d_in)  * xf.pow(2).mean(0)
            # Slow output EMA
            self._ema_out_mean_slow = d_out * self._ema_out_mean_slow + (1-d_out) * yf.mean(0)
            self._ema_out_sq_slow   = d_out * self._ema_out_sq_slow   + (1-d_out) * yf.pow(2).mean(0)

        return output
