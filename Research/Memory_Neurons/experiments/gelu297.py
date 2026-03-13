"""gelu297 – Causal Positional Decay Gate.

CONCEPT:
    Within a sequence of T tokens, later positions have accumulated MORE
    context from preceding tokens.  A token at position t=60 has "seen"
    59 preceding tokens; a token at t=0 has seen none.

    HYPOTHESIS: tokens at later sequence positions are inherently more
    predictable (more context → lower surprise) and therefore need LESS
    amplification from the novelty gate.

    This experiment multiplies the gelu211 gate by a learned positional
    discount:
        discount_t = exp(-lambda * t / T)   λ is a learnable positive scalar
        output     = out * gate_211 * discount_t.view(B, T, 1)

    lambda = 0  → no discount  (reduces to gelu211)
    lambda > 0  → later positions are more suppressed (habituated by context)
    lambda < 0  → later positions are amplified (rewarded for context use)

    The sign and magnitude of lambda is learned from data.

STRICT CAUSALITY:
    discount_t depends only on position index t, which is a static function —
    it does NOT depend on future activations at t' > t.

CHAIN WITH gelu211:
    gelu211's gate (gate_in × gate_out × gate_cos) captures cross-batch novelty.
    The positional discount captures within-sequence familiarity.
    The two are orthogonal and complement each other.

BENEFIT FROM BACKPROP:
    log_lambda: gradient shapes the positional discount profile.
    All gelu211 params also receive their standard gradients.

SEQUENTIAL ADAPTATION:
    Stateless across passes (discount is position-only, no buffer) → Δ ≈ 0.
    Benefit is in base PPL via within-sequence positional habituation.

PARAMS:  gelu211 params (7) + log_lambda.
STATE:   gelu211 state (5 EMA buffers).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU297(nn.Module):
    """gelu211 + learned positional discount: gate_t *= exp(-lambda * t/T)."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Positional discount ─────────────────────────────────────────
        # Unconstrained: exp(log_lambda) can be positive or negative
        # Init near 0 so discount starts as identity
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

        # ── gelu211 state (initialised as None) ────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def _init_state(self, x, out):
        with torch.no_grad():
            xf = x.flatten(0,1); of = out.flatten(0,1)
            self._ema_mean     = xf.mean(0).clone()
            self._ema_sq       = xf.pow(2).mean(0).clone()
            self._ema_out_mean = of.mean(0).clone()
            self._ema_out_sq   = of.pow(2).mean(0).clone()
            self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
            self._ready        = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        if not self._ready:
            self._init_state(x, out)
            return out

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        with torch.no_grad():
            xf = x.detach().flatten(0,1); of = out.detach().flatten(0,1)
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1,1)
            gate_cos= torch.exp(-tau.detach() * cos_sim).unsqueeze(-1)

        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # ── Positional discount ─────────────────────────────────────────
        lam      = self.log_lambda.exp()
        t_idx    = torch.arange(T, device=x.device, dtype=x.dtype) / T   # (T,) ∈ [0,1)
        discount = torch.exp(-lam * t_idx).view(1, T, 1)                  # (1, T, 1)

        output = out * gate_in * gate_out * gate_cos * discount

        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
