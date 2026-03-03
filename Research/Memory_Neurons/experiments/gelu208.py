"""GELU208 – SiLU Base + Sparse Top-K Signed Gate (K=16).

ISOLATES THE ACTIVATION FUNCTION CONTRIBUTION vs gelu189.

gelu189 = GELU(x) × sparse_gate × cosine_gate

Here we replace GELU with SiLU (Sigmoid Linear Unit):
    SiLU(x) = x × sigmoid(x)

Everything else is identical: same K=16 sparse signed gate, same cosine gate,
same EMA tracking, same parameter initialization.

GELU vs SiLU:
    GELU(x) ≈ x × Φ(x)  where Φ is the standard normal CDF
    SiLU(x) = x × σ(x)  where σ is the logistic sigmoid
    Both are smooth, non-monotonic, gated linear units.
    SiLU is slightly faster (no erf) and slightly "wider" in its negative region.

HYPOTHESIS: if gelu208 ≈ gelu189, the activation choice doesn't matter.
If gelu208 > gelu189, SiLU is a better base for this gating mechanism.

PARAMS: logit_decay, log_tau, log_beta, log_gamma (4 scalars, identical init to gelu189)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 16


class GELU208(nn.Module):
    """SiLU base + sparse top-K signed gate: identical to gelu189 but GELU → SiLU."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = min(self.k, D)

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        # SiLU instead of GELU
        out = x * torch.sigmoid(x)

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
            z    = (x.detach() - mu_) / (std + self.eps)     # (B, T, D)

            _, topk_idx = z.abs().topk(k, dim=-1)             # (B, T, K)

        gate_vec = torch.ones(B, T, D, device=x.device, dtype=x.dtype)
        z_topk   = torch.gather(z.detach(), -1, topk_idx)    # (B, T, K)
        g_topk   = (1.0 + beta * torch.tanh(gamma * z_topk)).clamp(0.1, 8.0)
        gate_vec = gate_vec.scatter(-1, topk_idx, g_topk)

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
