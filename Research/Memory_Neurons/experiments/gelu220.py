"""GELU220 – Global EMA + Local Batch Dual-Pathway Asymmetric Gate.

MOTIVATION — STATELESS COMPONENT FOR POSITIVE ADAPTATION:
    gelu190 (PPL 160.54, Δ=−0.012): purely EMA-based z-scores.
    On pass 2, EMA is slightly contaminated → z_global shrinks a bit → gate ≈ 1 → small degradation.

    KEY IDEA: Add a STATELESS local z-score component that:
        1. Provides novelty signal independent of cross-batch EMA state
        2. Remains informative on pass 2 (intra-batch variation doesn't change between passes)

    LOCAL BATCH Z-SCORE:
        z_local_d = (x_d − mean_{B,T}(x_d)) / std_{B,T}(x_d)
        This is instance normalization across the batch × time dimensions.
        No state → no contamination between passes.

    GLOBAL EMA Z-SCORE (gelu190 style):
        z_global_d = (x_d − EMA_mean_d) / EMA_std_d

    WEIGHTED BLEND:
        z_d = w_g × z_global_d + w_l × z_local_d
              where w_g, w_l are learnable scalar weights (softmax-normalized)

    ASYMMETRIC GATE on blended z-scores:
        gate_d = 1 + β_up×ReLU(tanh(γ×z_d)) − β_dn×ReLU(tanh(−γ×z_d))

    ADAPTATION PROPERTY:
        On pass 2: z_global shrinks (EMA contaminated) BUT z_local still fires for
        any locally unusual tokens. The blend of local+global gives better continuity
        of the novelty signal across passes, reducing degradation.

BIOLOGICAL ANALOGY:
    Local z-score: within-context contrast (what stands out in THIS context)
    Global z-score: historical contrast (what's unusual vs ALL training)
    Together: a token must be contextually unusual OR historically unusual to be amplified.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_w_global, log_w_local
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU220(nn.Module):
    """Dual-pathway asymmetric gate: blend global EMA z-scores with local batch z-scores."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Learnable blend weights (raw, will be softmax-normalised)
        self.log_w_global = nn.Parameter(torch.tensor(0.0))   # init equal weight
        self.log_w_local  = nn.Parameter(torch.tensor(0.0))

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
        # Softmax-normalize blend weights
        w_raw    = torch.stack([self.log_w_global, self.log_w_local])
        w        = torch.softmax(w_raw, dim=0)    # (2,) sums to 1
        w_g, w_l = w[0], w[1]

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
            # Global z-score (from EMA history)
            var_g   = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_global= (x.detach() - self._ema_mean.view(1,1,D)) / (var_g.sqrt().view(1,1,D) + self.eps)

            # Local z-score (intra-batch instance norm)
            xd      = x.detach()
            mu_l    = xd.mean(dim=(0, 1), keepdim=True)          # (1, 1, D)
            var_l   = ((xd - mu_l).pow(2)).mean(dim=(0, 1), keepdim=True).clamp(min=self.eps_var)
            z_local = (xd - mu_l) / (var_l.sqrt() + self.eps)   # (B, T, D)

        # Weighted blend of z-scores
        z = w_g * z_global + w_l * z_local                       # (B, T, D)

        # ── Asymmetric per-channel gate ────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z))
        gate_vec = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)

        # ── Cosine output EMA gate ─────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
