"""GELU183 – Dual-Space Z-Score Gate (Input × Output Activation Space).

RATIONALE:
    gelu80: novelty in INPUT space (z-score of x, before GELU).
    gelu98: novelty in OUTPUT space (z-score of GELU(x), after GELU).

    These two spaces are NOT equivalent:
    - GELU is piecewise: near-linear for large positive x, near-zero for large negative x.
    - A token can have novel input but familiar output (GELU saturates below ~-2).
    - Conversely: familiar input but unusual output near the GELU inflection point.

    PRODUCT: token must be novel from BOTH perspectives simultaneously.
    This eliminates false positives present in either signal alone.

ARCHITECTURE:
    EMA statistics for BOTH input and output spaces:
        ema_mean_in (D,), ema_sq_in (D,)   — tracks x
        ema_mean_out (D,), ema_sq_out (D,) — tracks GELU(x)

    surp_in  = tanh(σ1 × mean_d |z_in_d|)
    surp_out = tanh(σ2 × mean_d |z_out_d|)
    surp_joint = surp_in × surp_out

    gate = exp(-τ × cos(GELU(x), ema_out_dir)) × (1 + w × surp_joint)

PARAMS: logit_decay, log_tau, log_sig_in, log_sig_out, log_w = 5 scalars
STATE:  _ema_mean_in (D,), _ema_sq_in (D,), _ema_mean_out (D,), _ema_sq_out (D,),
        _ema_out_dir (D,) unit vector for cosine gate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU183(nn.Module):
    """Dual-space z-score gate: input-space novelty × output-space novelty."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sig_in  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_sig_out = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean_in:  torch.Tensor = None
        self._ema_sq_in:    torch.Tensor = None
        self._ema_mean_out: torch.Tensor = None
        self._ema_sq_out:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean_in  = None
        self._ema_sq_in    = None
        self._ema_mean_out = None
        self._ema_sq_out   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        sig_in  = F.softplus(self.log_sig_in)
        sig_out = F.softplus(self.log_sig_out)
        w       = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean_in  = xf.mean(0).clone()
                self._ema_sq_in    = xf.pow(2).mean(0).clone()
                self._ema_mean_out = of.mean(0).clone()
                self._ema_sq_out   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq_in  - self._ema_mean_in.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()  - self._ema_mean_in.view(1,1,D))  / (var_in.sqrt().view(1,1,D)  + self.eps)
            surp_in = torch.tanh(sig_in * z_in.abs().mean(-1))

            var_out  = (self._ema_sq_out - self._ema_mean_out.pow(2)).clamp(min=self.eps_var)
            z_out    = (out.detach() - self._ema_mean_out.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)
            surp_out = torch.tanh(sig_out * z_out.abs().mean(-1))

            surp_joint = surp_in * surp_out

            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            gate_cos = torch.exp(-tau * (out_n * ema_n).sum(-1).clamp(-1,1))

        output = out * (gate_cos * (1.0 + w * surp_joint)).unsqueeze(-1)

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean_in  = d_val * self._ema_mean_in  + (1-d_val) * xf.mean(0)
            self._ema_sq_in    = d_val * self._ema_sq_in    + (1-d_val) * xf.pow(2).mean(0)
            self._ema_mean_out = d_val * self._ema_mean_out + (1-d_val) * of.mean(0)
            self._ema_sq_out   = d_val * self._ema_sq_out   + (1-d_val) * of.pow(2).mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(of.mean(0), dim=0)

        return output
