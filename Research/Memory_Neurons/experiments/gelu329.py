"""GELU329 – Gate Layer-Norm (Normalize Gate Vector to Mean=1).

MOTIVATION: gelu211 multiplies by [gate_in × gate_out × gate_cos]. If all gates are
consistently > 1 or < 1 for many channels simultaneously, this creates a global
scaling of the output — essentially a learnable per-token amplitude. This GLOBAL
SCALING is confounded with the layer's contribution to next-layer dynamics.

HYPOTHESIS: If we NORMALIZE the combined gate vector to have mean=1 per token,
we remove global amplitude scaling and force the gate to encode only RELATIVE
channel prioritization (which channels matter more vs others), not global scaling.
This is analogous to Layer Normalization but applied to the gate itself.

ARCHITECTURE:
    gate_in  = asym(z_in)     [same as gelu211]
    gate_out = sym(z_out)     [same as gelu211]
    gate_cos = cosine gate    [same as gelu211]
    gate_raw = gate_in × gate_out × gate_cos   [shape: (B, T, D)]
    gate_normed = gate_raw / gate_raw.mean(dim=-1, keepdim=True).clamp(min=0.1)
    output = out × gate_normed

EFFECT:
    - Global scaling removed: mean gate per token = 1.0 always
    - Relative gate values preserved: channel-to-channel comparison still meaningful
    - Prevents gate from learning a trivial "amplify everything" solution
    - Gradient: now flows through the normalization operation

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out,
        log_gamma_out  (7 scalars — identical to gelu211)
STATE: same 5 EMA buffers as gelu211
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU329(nn.Module):
    """Gate layer-norm: normalize combined gate to per-token mean=1 before applying."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))   # init d ≈ 0.9
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = None
        self._ema_out_mean = self._ema_out_sq = self._ema_out_dir = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
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
                bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
                bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
                self._ema_mean     = bm_x.clone(); self._ema_sq     = bsq_x.clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = bsq_o.clone()
                self._ema_out_dir  = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        # NORMALIZATION: divide combined gate by its per-token mean → mean stays = 1
        gate_raw    = gate_in * gate_out * gate_cos          # (B, T, D)
        gate_mean   = gate_raw.mean(dim=-1, keepdim=True).clamp(min=0.1)
        gate_normed = gate_raw / gate_mean                   # each token's gate has mean 1

        output = out * gate_normed

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0);  bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0);  bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(bm_o, dim=0)

        return output
