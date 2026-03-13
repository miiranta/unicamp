"""GELU309 – Per-Channel τ Cosine Gate (Vector τ Instead of Scalar).

gelu211 uses a SCALAR τ for the cosine gate:
    gate_cos = exp(-τ * cos(out_normalized, ema_dir))   [single τ for all channels]

HYPOTHESIS: Different channels may need different sensitivities to directional
novelty. A per-channel τ_d allows each channel to learn how much it should care
about its cosine similarity to the EMA direction.

PER-CHANNEL COSINE GATE:
    cos_per_channel_d = (out_d_normalized × ema_dir_d)^2   [per-channel scalar, (B,T,D)]

    Wait — the cosine similarity in gelu211 is a DOT PRODUCT over ALL D channels,
    resulting in one scalar per token. A per-channel version must use element-wise
    operations:

    gate_cos_d = exp(-τ_d * |hat_out_d - hat_ema_d|)   [element-wise difference in normalised space]

    OR: gate_cos_d = exp(-τ_d * (1 - hat_out_d * hat_ema_d))  [1 - cosine per element]
    where hat_out = out / |out|  and hat_ema = ema_dir / |ema_dir|

    This gives each channel its own decay sensitivity.

PARAMS: logit_decay, log_tau_vec (D,), log_beta_up, log_beta_dn, log_gamma,
        log_beta_out, log_gamma_out  (D+6 params = 1030 for D=1024)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU309(nn.Module):
    """gelu211 with per-channel τ_d cosine gate instead of scalar τ."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        D = D_FF
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau_vec   = nn.Parameter(torch.full((D,), math.log(2.0)))   # per-channel τ
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

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau_vec   = self.log_tau_vec.exp().view(1, 1, D)     # (1, 1, D) — differentiable
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
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()   - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # Per-channel cosine gate: element-wise dissimilarity from EMA direction
        with torch.no_grad():
            # Normalise out and ema_dir globally (L2 over D dimension)
            out_norm = out.detach() / (out.detach().norm(dim=-1, keepdim=True) + self.eps)
            ema_norm = self._ema_out_dir / (self._ema_out_dir.norm() + self.eps)
            # Element-wise: 1 - cos_d approximation via squared difference in normalised space
            diff_sq  = (out_norm - ema_norm.view(1, 1, D)).pow(2)   # (B, T, D)

        # tau_vec differentiable, diff_sq detached
        gate_cos_vec = torch.exp(-tau_vec * diff_sq)   # (B, T, D) — τ gets gradient

        output = out * gate_in * gate_out * gate_cos_vec

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
