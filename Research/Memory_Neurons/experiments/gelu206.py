"""GELU206 – Sparse Top-K Absolute Gate (Amplify-Only, K=16).

ABLATION OF gelu189 — removes the sign from the sparse gate.

In gelu189 the gate at each top-K channel is:
    g_d = clamp(1 + β × tanh(γ × z_d), 0.1, 8.0)
which can SUPPRESS (z_d < 0) or AMPLIFY (z_d > 0).

Here we test: what if the gate is sign-blind and can only AMPLIFY?
    g_d = clamp(1 + β × tanh(γ × |z_d|), 1.0, 8.0)
    floor raised to 1.0 → gate ∈ [1, 8], never suppresses.

HYPOTHESIS: suppression of negative-z channels is what drives gelu189's gains.
If gelu206 performs WORSE than gelu189, that confirms suppression is key.
If gelu206 performs BETTER, then the sign was adding noise.

Everything else identical to gelu189 (cosine gate, EMA, K=16).

PARAMS: logit_decay, log_tau, log_beta, log_gamma (4 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 16


class GELU206(nn.Module):
    """Sparse top-K ABSOLUTE gate: amplify-only, gate ∈ [1, 8], never suppresses."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k       = K
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = min(self.k, D)

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0, 1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std  = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std + self.eps)       # (B, T, D) signed

            _, topk_idx = z.abs().topk(k, dim=-1)              # (B, T, K)

        gate_vec = torch.ones(B, T, D, device=x.device, dtype=x.dtype)
        z_topk   = torch.gather(z.detach(), -1, topk_idx)      # (B, T, K) signed
        # KEY DIFFERENCE FROM gelu189: use |z_topk| and floor=1.0 → amplify only
        g_topk   = (1.0 + beta * torch.tanh(gamma * z_topk.abs())).clamp(1.0, 8.0)
        gate_vec = gate_vec.scatter(-1, topk_idx, g_topk)       # (B, T, D)

        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)  # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1 - d_val) * F.normalize(om, dim=0)

        return output
