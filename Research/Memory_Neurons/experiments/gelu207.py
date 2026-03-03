"""GELU207 – Sparse Dual Gate: Amplify Novel + Suppress Familiar (K=16 each).

EXTENDS gelu189 with a symmetric bottom-K suppression arm.

gelu189 finds the top-K=16 channels by |z| (most surprising) and gates them
with a signed multiplier. But the bottom D-K channels are untouched (gate=1).

NEW IDEA: also attend to the bottom-K channels by |z| (MOST FAMILIAR channels,
z ≈ 0) and explicitly SUPPRESS them with a learned scalar β_fam.

    Top-K gate (same as gelu189):
        g_topk = clamp(1 + β_up × tanh(γ × z_topk), 0.1, 8.0)

    Bottom-K gate (new):
        β_fam = sigmoid(logit_beta_fam)         ∈ (0, 1)
        g_bot  = β_fam                          scalar suppression

    All other D - 2K channels: gate = 1.0

INTUITION: familiar channels are signalling things the network already knows.
Suppressing them forces the model to route information through the novel channels.
This is analogous to lateral inhibition in cortical circuits.

INIT: logit_beta_fam = log(0.2/0.8) ≈ -1.386 → β_fam ≈ 0.2 initially (moderate suppression)

PARAMS: logit_decay, log_tau, log_beta_up, log_gamma, logit_beta_fam (5 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 16


class GELU207(nn.Module):
    """Dual sparse gate: amplify top-K novel + suppress bottom-K familiar."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k       = K
        self.logit_decay    = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma      = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # init: sigmoid(-1.386) ≈ 0.2  → 20% suppression of familiar channels
        self.logit_beta_fam = nn.Parameter(torch.tensor(-1.386))

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
        k = min(self.k, D // 2)  # ensure top-K and bottom-K don't overlap

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        gamma   = F.softplus(self.log_gamma)
        beta_fam = torch.sigmoid(self.logit_beta_fam)  # ∈ (0, 1)

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
            z    = (x.detach() - mu_) / (std + self.eps)     # (B, T, D)

            abs_z = z.abs()
            _, topk_idx = abs_z.topk(k, dim=-1, largest=True)   # (B, T, K) most surprising
            _, botk_idx = abs_z.topk(k, dim=-1, largest=False)  # (B, T, K) most familiar

        # Start from all-ones
        gate_vec = torch.ones(B, T, D, device=x.device, dtype=x.dtype)

        # Top-K: signed amplify/suppress (same as gelu189)
        z_topk = torch.gather(z.detach(), -1, topk_idx)          # (B, T, K)
        g_topk = (1.0 + beta_up * torch.tanh(gamma * z_topk)).clamp(0.1, 8.0)
        gate_vec = gate_vec.scatter(-1, topk_idx, g_topk)

        # Bottom-K: soft suppression with learned scalar β_fam
        g_botk = beta_fam.expand(B, T, k)                        # (B, T, K)
        gate_vec = gate_vec.scatter(-1, botk_idx, g_botk)

        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)   # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1 - d_val) * F.normalize(om, dim=0)

        return output
