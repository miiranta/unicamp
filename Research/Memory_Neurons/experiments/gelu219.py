"""GELU219 – Surprise-Gated EMA Anchor: Selective Memory Update.

CORE PROBLEM WITH STANDARD EMA APPROACHES:
    gelu190's EMA updates every batch regardless of novelty.
    After test pass 1: EMA has shifted toward test distribution.
    On pass 2: z-scores based on test-contaminated EMA → less signal → gate ≈ 1 → similar to pass 1.
    
    Actually this SHOULD help adaptation. Why does it degrade?
    Because the cosine output gate (gate_cos = exp(-τ × cos(out, ema_out))) ALSO contaminates:
    ema_out drifts toward test output direction → gate_cos changes → distorts pass 2.

HYPOTHESIS: The cosine output gate is the PRIMARY source of degradation.
    - During training: ema_out ≈ "average training activation direction"
    - After test pass 1: ema_out shifts toward test output direction
    - On pass 2: cos(out, ema_out) is higher than on pass 1 → gate_cos is lower → LESS amplification
    - This means the effective gate SHRINKS on pass 2 → less beneficial novelty amplification

FIX: SELECTIVE EMA UPDATE (Surprise-Gated Anchor)
    Only update EMA components when the current token is GENUINELY NOVEL:
        z_scalar = tanh(σ × mean_d |z_d|)   — scalar surprise
        if z_scalar > threshold: update EMA
        else: don't update (or update with MUCH smaller weight)

    This prevents test distribution from contaminating the anchor during FAMILIAR passages.
    For NOVEL passages (test data that is genuinely unusual), the EMA updates meaningfully.

    More specifically: update weight = d_base × (1 - α_gate × familiar_factor)
        where familiar_factor = 1 - z_scalar (more familiar → less update)

IMPLEMENTATION:
    Effective decay for this batch:
        d_eff = 1 - (1 - d_base) × z_scalar   — slow down update when familiar
    This prevents rapid EMA contamination from familiar test data.

    During training: z_scalar varies → mixed fast/slow → learns optimal.
    During test pass 1: mostly familiar test data → slow EMA update → less contamination.
    During test pass 2: ema_out ≈ training anchor → gate_cos still meaningful.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_sigma
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU219(nn.Module):
    """Surprise-gated EMA: update rate is proportional to novelty, preventing test contamination."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Surprise sensitivity for selective update
        self.log_sigma    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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

        d_base  = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)
        sigma   = F.softplus(self.log_sigma)

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
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)
            z   = (x.detach() - mu_) / (std + self.eps)        # (B, T, D)
            # Scalar surprise for selective update
            z_scalar = torch.tanh(sigma * z.abs().mean())       # scalar ∈ (0,1)

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
            # Selective update: more novel → update more; more familiar → update less
            d_eff = 1.0 - (1.0 - d_base) * z_scalar.item()    # d_eff ∈ [d_base, 1.0]
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_eff * self._ema_mean + (1-d_eff) * xf.mean(0)
            self._ema_sq   = d_eff * self._ema_sq   + (1-d_eff) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_eff * self._ema_out  + (1-d_eff) * F.normalize(om, dim=0)

        return output
