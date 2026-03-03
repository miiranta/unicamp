"""GELU197 – Cross-Standard-Deviation Surprise (Variance of Z-Scores).

WHAT MEAN |z| MISSES:
    gelu80: surp = tanh(σ × mean_d |z_d|)
    This captures the AVERAGE deviation across channels.
    A token with mean|z| = 2 could have all channels at z_d ≈ 2,
    OR it could have D/2 channels at z_d ≈ 4 and D/2 at z_d ≈ 0.

    The second case is MORE surprising: it means the activation pattern has
    SHIFTED — some channels are very high while others are at baseline.
    This pattern selectivity is what distinguishes a genuine semantic shift
    from a global scaling (all channels up or down proportionally).

THE NEW IDEA: Standard Deviation of Z-Scores
    std_z = sqrt(mean_d (z_d - mean_d z_d)²)    — (B, T) scalar

    This measures SPREAD in the deviation profile:
    - High std_z: the token activates some channels strongly and suppresses others
      (a selective, content-specific activation pattern)
    - Low std_z: all channels deviate by the same amount (a global shift)

    Combined surprise using BOTH mean and std:
        surp = tanh(σ1 × mean|z| + σ2 × std_z)

    This fires for:
    - Global surprises (high mean|z|, any std_z)
    - Selective reshaping (any mean|z|, high std_z)

PARAMS: logit_decay, log_tau, log_sigma1, log_sigma2, log_w_raw = 5 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU197(nn.Module):
    """Variance-of-z-scores surprise: captures selective channel reshaping, not just global deviation."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma1    = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_sigma2    = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
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
        sigma1 = F.softplus(self.log_sigma1)
        sigma2 = F.softplus(self.log_sigma2)
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

        # ── Per-channel z-score ────────────────────────────────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_ = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + self.eps)      # (B, T, D)

        # ── Mean and std of z-scores ───────────────────────────────────────
        mean_z    = z.mean(dim=-1, keepdim=True)               # (B, T, 1)
        mean_absz = z.abs().mean(dim=-1)                        # (B, T)
        std_z     = ((z - mean_z).pow(2).mean(dim=-1)).sqrt()   # (B, T)

        surp = torch.tanh(sigma1 * mean_absz + sigma2 * std_z)  # (B, T)

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xfl.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
