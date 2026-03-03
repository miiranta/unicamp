"""GELU176 – Multiplicative Dual-Reference Per-Channel Z-Score.

CORE IDEA:
    gelu80 (best so far) uses ONE reference: cross-batch EMA per-channel mean/var.
    gelu86 uses ONE reference: within-sequence causal cumulative mean/var.

    Both have independent failure modes:
    - gelu80: a globally unusual token in a LOCALLY predictable context scores high
      (false positive: normal in context, just rare globally)
    - gelu86: a locally unusual token that matches the global training distribution
      scores high (false positive: rare in passage, but globally common)

    SOLUTION: MULTIPLY the two surprise scores.
        surp_global = tanh(σ_g × mean_d |z_global_d|)    ∈ (0, 1)
        surp_local  = tanh(σ_l × mean_d |z_local_d|)     ∈ (0, 1)
        surp_joint  = surp_global × surp_local            ∈ (0, 1)

    The product is HIGH only when BOTH references agree: the token is novel relative
    to the global training distribution AND unusual within its current context.

    This eliminates both classes of false positives simultaneously, at the cost of
    making the gate more conservative. We compensate with a learned amplification w.

ARCHITECTURE:
    Same cosine-EMA gate backbone as gelu80:
        gate = exp(-τ × cos(out, ema_out)) × (1 + w × surp_joint)

    Global reference: cross-batch EMA of mean and x² per channel (detached)
    Local reference: causal cumsum within-sequence (detached, stateless)

    Both z-score computations are no-grad; gradients only flow through GELU(x).

PARAMS: logit_decay, log_tau, log_sigma_g, log_sigma_l, log_w = 5 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,) unit vector
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU176(nn.Module):
    """Multiplicative dual-reference per-channel z-score gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_g   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # global sensitivity
        self.log_sigma_l   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # local sensitivity
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        tau    = self.log_tau.exp()
        sigma_g = F.softplus(self.log_sigma_g)
        sigma_l = F.softplus(self.log_sigma_l)
        w      = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # ── Global per-channel z-score (cross-batch EMA reference) ─────
            var_g  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_g  = var_g.sqrt()
            mu_g_  = self._ema_mean.view(1, 1, D)
            std_g_ = std_g.view(1, 1, D)
            z_g    = (x.detach() - mu_g_) / (std_g_ + self.eps)     # (B, T, D)
            surp_g = torch.tanh(sigma_g * z_g.abs().mean(dim=-1))    # (B, T)

            # ── Local per-channel z-score (within-sequence causal) ─────────
            xd     = x.detach()
            cum_x  = torch.cumsum(xd, dim=1)
            cum_sq = torch.cumsum(xd.pow(2), dim=1)
            zeros1 = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            mu_c   = torch.cat([zeros1, cum_x[:, :-1, :]], dim=1)    # (B, T, D) causal mean unnorm
            sq_c   = torch.cat([zeros1, cum_sq[:, :-1, :]], dim=1)
            cnts   = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1, T, 1).clamp(min=1)
            # position 0: count=0 prior; we use the shifted index so count at t is t
            cnt_shifted = torch.arange(0, T, device=x.device, dtype=x.dtype).view(1, T, 1).clamp(min=1)
            mu_l   = mu_c / cnt_shifted                              # (B, T, D)
            sq_l   = sq_c / cnt_shifted
            var_l  = (sq_l - mu_l.pow(2)).clamp(min=self.eps_var)
            std_l  = var_l.sqrt()
            z_l    = (xd - mu_l) / (std_l + self.eps)               # (B, T, D)
            # position 0 has no prior → set z=0 (neutral)
            z_l[:, 0, :] = 0.0
            surp_l = torch.tanh(sigma_l * z_l.abs().mean(dim=-1))   # (B, T)

            # ── Joint surprise: product of both references ─────────────────
            surp_joint = surp_g * surp_l                             # (B, T)

            # ── Cosine familiarity gate ────────────────────────────────────
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp_joint)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ──────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
