"""GELU185 – Channel-Group Z-Score Gate (G=8 Groups, G Independent Scalar Gates).

THE SCALAR vs PER-CHANNEL TRADE-OFF:
    gelu80:  1 scalar gate per token (collapses D → 1)
    gelu181: D vector gates per token (one per channel — maximum resolution)
    gelu185: G=8 group gates per token (structured middle ground)

WHY GROUPS:
    D=128 channels divided into G=8 groups of Dg=16 channels each.
    Within each group: mean |z| captures group-level surprise.
    Each group gets its own independent gate scalar.

    Groups capture structured co-variation: channels within a group that all
    deviate together drive a large group gate. Single outlier channels within a
    group are dampened (group mean is robust). This is intermediate between
    gelu80's fully-collapsed scalar and gelu181's fully-independent per-channel gate.

PER-GROUP COSINE GATE:
    Track ema_out in G groups: _ema_out (G, Dg) per-group direction vectors.
    cos_g = cosine(GELU(x)_{group g}, ema_out_g) — group-specific directional familiarity.

    gate_g = exp(-τ × cos_g) × (1 + w × surp_g)
    output_{group g} = GELU(x_{group g}) × gate_g     — applied to channel slice

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (G, Dg)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


N_GROUPS = 8


class GELU185(nn.Module):
    """Channel-group z-score gate: G=8 independent group-level gates."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5, n_groups: int = N_GROUPS):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.n_groups = n_groups
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None   # (G, Dg)
        self._D_last: int = -1
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
        G  = self.n_groups
        Dg = D // G
        D_used = G * Dg

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready or self._D_last != D:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                of_g = of[:, :D_used].view(-1, G, Dg).mean(0)      # (G, Dg)
                self._ema_out  = F.normalize(of_g, dim=-1).clone()
                self._D_last   = D
                self._ready    = True
            return out

        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std  = var.sqrt().view(1, 1, D)
            z    = (x.detach() - self._ema_mean.view(1,1,D)) / (std + self.eps)  # (B, T, D)

            # Group-level mean |z|
            z_g     = z[:, :, :D_used].view(B, T, G, Dg)           # (B, T, G, Dg)
            surp_g  = torch.tanh(sigma * z_g.abs().mean(-1))        # (B, T, G)

            # Per-group cosine gate
            out_g   = F.normalize(out.detach()[:, :, :D_used].view(B, T, G, Dg), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=-1).view(1, 1, G, Dg)
            cos_g   = (out_g * ema_n).sum(-1).clamp(-1, 1)         # (B, T, G)
            gate_g  = torch.exp(-tau * cos_g) * (1.0 + w * surp_g) # (B, T, G)

        # Apply per-group gate to corresponding channel slice
        gate_exp = gate_g.unsqueeze(-1).expand(B, T, G, Dg)        # (B, T, G, Dg)
        output   = out.clone()
        output[:, :, :D_used] = (out[:, :, :D_used].view(B, T, G, Dg) * gate_exp).view(B, T, D_used)

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            of_g = of[:, :D_used].view(-1, G, Dg).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(of_g, dim=-1)

        return output
