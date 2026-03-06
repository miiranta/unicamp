"""GELU211 – Asymmetric Input × Output Product Gate.

MOTIVATION — COMBINING gelu190 AND gelu198:
    gelu190 (PPL 160.54, best overall) computes a per-channel asymmetric gate
    based on INPUT z-scores:
        gate_d = clamp(1 + β_up*ReLU(tanh(γ*z_in_d)) − β_dn*ReLU(tanh(−γ*z_in_d)), 0.05, 8.0)

    gelu198 (PPL 161.89) tracks OUTPUT statistics and gates based on output z-scores.

    KEY INSIGHT: Input novelty (z_in ≠ 0) does not always translate to output novelty.
    A channel can have unusual input but its GELU output is still near the baseline.
    If we require BOTH input AND output to be novel, we get a sharper, less noisy signal.

    PRODUCT GATE:
        gate_in_d   = asymmetric(z_in_d)      — gelu190 per-channel gate
        gate_out_d  = 1 + β_out × tanh(γ_out × z_out_d)  — symmetric output gate
        gate_final  = clamp(gate_in_d × gate_out_d, 0.05, 10.0)

    When z_in_d > 0 AND z_out_d > 0: channel is above mean in BOTH spaces → amplify strongly
    When z_in_d > 0 AND z_out_d < 0: unusual input but suppressed output → product ≈ 1 (pass)
    When z_in_d < 0 AND z_out_d < 0: below mean in BOTH spaces → suppress strongly

    This is a CONJUNCTIVE novelty requirement — more selective than either gate alone.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU211(nn.Module):
    """Asymmetric input-space gate × symmetric output-space gate, per-channel product."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        # Input-space asymmetric params (from gelu190)
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Output-space symmetric correction
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

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
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            # Input z-scores
            var_in = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in   = (x.detach() - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            # Output z-scores
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        # ── Input-space asymmetric gate ────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)    # (B, T, D)

        # ── Output-space symmetric gate ────────────────────────────────
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # ── Cosine output EMA gate (scalar, from gelu190) ─────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)   # (B, T, 1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
