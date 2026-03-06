"""GELU217 – Within-Sequence NN Gate × Asymmetric Historical Gate (gelu184 × gelu190).

MOTIVATION — COMBINING STATELESS ADAPTATION (gelu184) WITH STRONG PPL (gelu190):
    gelu184 (PPL 163.55, adaptation Δ=+0.019): within-sequence nearest-neighbor gate
        — stateless globally; novelty = dissimilarity to nearest token in context
        — adapts positively because intra-seq distances don't change with EMA contamination

    gelu190 (PPL 160.54, adaptation Δ=−0.012): asymmetric per-channel EMA gate
        — best PPL; EMA gives historical context but slightly degrades on re-runs

    PRODUCT GATE:
        gate_hist_d = gelu190's asymmetric per-channel gate (historical)
        gate_nn     = gelu184's nearest-neighbor scalar gate (local context)
        gate_final  = gate_hist_d × gate_nn_scalar

    WHY PRODUCT WORKS:
        - A token must be BOTH historically novel (unusual channels vs EMA)
          AND locally isolated (no near-duplicate in its context window)
        - This double requirement reduces false positives:
          * Channels that are statistically unusual but locally common → gate ≈ 1 (pass-through)
          * Tokens that are locally isolated but historically typical → gate ≈ 1
        - Only truly isolated & unusual tokens get amplified

    ADAPTATION MECHANISM:
        gelu184's within-seq NN gate does NOT use cross-batch EMA.
        Even on pass 2, the gelu184 component doesn't change (test data patterns
        are locally re-evaluated fresh each time). gelu190's EMA does update during
        pass 1, but the gelu184 factor partially counteracts the degradation.
        Net Δ should be better than gelu190 alone.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_sigma_nn, log_w_nn
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU217(nn.Module):
    """Per-channel asymmetric EMA gate × within-seq nearest-neighbor gate (product)."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # Historical (gelu190) params
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Local NN (gelu184) params
        self.log_sigma_nn = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_nn     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        sigma_nn = F.softplus(self.log_sigma_nn)
        w_nn     = F.softplus(self.log_w_nn)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Component 1: Historical asymmetric per-channel gate ────────
        with torch.no_grad():
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)
            z   = (x.detach() - mu_) / (std + self.eps)      # (B, T, D)

        up_arm   = beta_up * F.relu(torch.tanh( gamma * z))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z))
        gate_hist= (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)  # (B, T, D)

        # ── Component 2: Within-seq nearest-neighbor scalar gate ───────
        with torch.no_grad():
            x_n  = F.normalize(x.detach(), dim=-1)            # (B, T, D)
            sim  = torch.bmm(x_n, x_n.transpose(1, 2))        # (B, T, T)
            eye  = torch.eye(T, device=x.device, dtype=torch.bool).unsqueeze(0)
            sim  = sim.masked_fill(eye, -2.0)
            nn_sim = sim.max(dim=-1).values                    # (B, T) — max excl. self
            novelty_nn = (1.0 - nn_sim) / 2.0                 # ∈ [0, 1]
            surp_nn = torch.tanh(sigma_nn * novelty_nn)       # (B, T)
        gate_nn = (1.0 + w_nn * surp_nn).unsqueeze(-1)        # (B, T, 1)

        # ── Cosine output EMA gate (scalar) ────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_hist * gate_nn * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
