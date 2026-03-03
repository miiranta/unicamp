"""GELU205 – Dual-Space Sparse Intersection Gate (K_each=64).

Gates only channels in top-64 by BOTH |z_in| AND |z_out| simultaneously.
Double filter: dead-zone channel spikes (large z_in, near-zero z_out) are excluded.
K_each=64 -> expected intersection ~4 channels (64^2/1024). More reliable novelty signal.
PARAMS: logit_decay, log_tau, log_beta, log_gamma
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir (all D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


K_EACH = 64


class GELU205(nn.Module):
    """Dual-space intersection gate: channels in top-K_each by BOTH |z_in| AND |z_out|."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k_each  = K_EACH
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
        k = min(self.k_each, D)

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

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

        # ── Dual-space intersection: top-K in INPUT space ∩ top-K in OUTPUT space ────
        with torch.no_grad():
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)

            _, idx_in  = z_in.abs().topk(k, dim=-1)    # (B, T, K)
            _, idx_out = z_out.abs().topk(k, dim=-1)   # (B, T, K)
            mask_in  = torch.zeros(B, T, D, dtype=torch.bool, device=x.device).scatter_(-1, idx_in,  True)
            mask_out = torch.zeros(B, T, D, dtype=torch.bool, device=x.device).scatter_(-1, idx_out, True)
            inter    = mask_in & mask_out               # channels unusual in BOTH spaces

        z_in_det = z_in.detach()
        inter_g  = (1.0 + beta * torch.tanh(gamma * z_in_det)).clamp(0.1, 8.0)
        gate_vec = torch.where(inter, inter_g, torch.ones_like(inter_g))    # (B, T, D)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)              # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            ofl = out.detach().flatten(0, 1)
            xfl = x.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xfl.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xfl.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * ofl.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * ofl.pow(2).mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(ofl.mean(0), dim=0)

        return output
