"""GELU328 – Multi-Scale Z-Score Gate (Short + Long EMA Combination).

MOTIVATION: gelu211 computes z-scores from a single EMA with one decay rate (d≈0.9).
This captures a single timescale of "what is normal."

HYPOTHESIS: Using TWO timescales provides richer novelty information:
    z_fast = (x - ema_fast_mean) / std_fast   — recent baseline (d_fast ≈ 0.5, adapts quickly)
    z_slow = (x - ema_slow_mean) / std_slow   — long-term baseline (d_slow ≈ 0.99, stable)

MULTI-SCALE COMBINATION:
    z_combined = z_fast + α * z_slow   (or) z_combined = (z_fast + z_slow) / 2

    - When z_fast > 0 AND z_slow > 0: unusual in BOTH short and long term → very novel
    - When z_fast > 0 AND z_slow ≈ 0: recently novel but normal in long term → transitional
    - When z_fast ≈ 0 AND z_slow > 0: unusual in long term but recently adapted → gradual novelty

    α is learnable: controls how much long-term context contributes.

GATE ON COMBINED Z-SCORE:
    gate_in  = asym(z_fast + α_in * z_slow_in)   [input multi-scale]
    gate_out = sym(z_fast_out + α_out * z_slow_out)   [output multi-scale]
    gate_cos = cosine gate (from slow EMA — more stable direction)
    output   = out × gate_in × gate_out × gate_cos

PARAMS: logit_d_fast, logit_d_slow, log_tau, alpha_in (scalar), alpha_out (scalar),
        log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (10 scalars)
STATE:  _fast_mean, _fast_sq, _slow_mean, _slow_sq, _fast_out_mean, _fast_out_sq,
        _slow_out_mean, _slow_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU328(nn.Module):
    """Multi-scale z-score gate: short + long EMA combined with learnable α weighting."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_d_fast  = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))      # d_fast ≈ 0.5
        self.logit_d_slow  = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))    # d_slow ≈ 0.99
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.alpha_in      = nn.Parameter(torch.tensor(0.3))   # learnable mixing weight
        self.alpha_out     = nn.Parameter(torch.tensor(0.3))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._fast_mean:     torch.Tensor = None
        self._fast_sq:       torch.Tensor = None
        self._slow_mean:     torch.Tensor = None
        self._slow_sq:       torch.Tensor = None
        self._fast_out_mean: torch.Tensor = None
        self._fast_out_sq:   torch.Tensor = None
        self._slow_out_mean: torch.Tensor = None
        self._slow_out_sq:   torch.Tensor = None
        self._ema_out_dir:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._fast_mean     = None
        self._fast_sq       = None
        self._slow_mean     = None
        self._slow_sq       = None
        self._fast_out_mean = None
        self._fast_out_sq   = None
        self._slow_out_mean = None
        self._slow_out_sq   = None
        self._ema_out_dir   = None
        self._ready         = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_fast    = torch.sigmoid(self.logit_d_fast).detach().item()
        d_slow    = torch.sigmoid(self.logit_d_slow).detach().item()
        tau       = self.log_tau.exp()
        alpha_in  = self.alpha_in      # differentiable — gradient flows through z_combined
        alpha_out = self.alpha_out
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                bm_x, bm_o = xf.mean(0).clone(), of.mean(0).clone()
                bsq_x, bsq_o = xf.pow(2).mean(0).clone(), of.pow(2).mean(0).clone()
                self._fast_mean     = bm_x;  self._fast_sq     = bsq_x
                self._slow_mean     = bm_x;  self._slow_sq     = bsq_x
                self._fast_out_mean = bm_o;  self._fast_out_sq = bsq_o
                self._slow_out_mean = bm_o;  self._slow_out_sq = bsq_o
                self._ema_out_dir   = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            xd  = x.detach().view(1, 1, D) if False else x.detach()   # keep shape (B,T,D)
            outd = out.detach()

            z_fast_in  = self._z(xd,   self._fast_mean.view(1,1,D),  self._fast_sq.view(1,1,D))
            z_slow_in  = self._z(xd,   self._slow_mean.view(1,1,D),  self._slow_sq.view(1,1,D))
            z_fast_out = self._z(outd, self._fast_out_mean.view(1,1,D), self._fast_out_sq.view(1,1,D))
            z_slow_out = self._z(outd, self._slow_out_mean.view(1,1,D), self._slow_out_sq.view(1,1,D))

        # Combine timescales — alpha_in/alpha_out are differentiable
        z_in  = z_fast_in  + alpha_in  * z_slow_in     # (B, T, D), alpha gets gradient
        z_out = z_fast_out + alpha_out * z_slow_out

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0);  bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0);  bsq_o = of.pow(2).mean(0)
            self._fast_mean     = d_fast * self._fast_mean     + (1 - d_fast) * bm_x
            self._fast_sq       = d_fast * self._fast_sq       + (1 - d_fast) * bsq_x
            self._slow_mean     = d_slow * self._slow_mean     + (1 - d_slow) * bm_x
            self._slow_sq       = d_slow * self._slow_sq       + (1 - d_slow) * bsq_x
            self._fast_out_mean = d_fast * self._fast_out_mean + (1 - d_fast) * bm_o
            self._fast_out_sq   = d_fast * self._fast_out_sq   + (1 - d_fast) * bsq_o
            self._slow_out_mean = d_slow * self._slow_out_mean + (1 - d_slow) * bm_o
            self._slow_out_sq   = d_slow * self._slow_out_sq   + (1 - d_slow) * bsq_o
            self._ema_out_dir   = d_slow * self._ema_out_dir   + (1 - d_slow) * F.normalize(bm_o, dim=0)

        return output
