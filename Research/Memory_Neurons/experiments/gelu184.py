"""GELU184 – Within-Sequence Nearest-Neighbor Distance Gate.

STATELESS INTRA-SEQUENCE NOVELTY (MINIMAL GLOBAL STATE):
    A token is LOCALLY novel if it is dissimilar to ALL other tokens in the same
    forward pass — it has no near-duplicate in context.

    For each token t:
        nn_sim_t = max_{s ≠ t} cosine(x_t, x_s)    — similarity to nearest neighbor

    nn_sim → 1: has a near-duplicate in context → FAMILIAR
    nn_sim → −1: opposite to all others → different kind of novel (opponent)
    nn_sim → 0: orthogonal to all others → ISOLATED/NOVEL

        novelty_t = (1 − nn_sim_t) / 2   ∈ [0, 1]
        surp_t    = tanh(σ × novelty_t)

    Combined with cosine output EMA gate:
        gate_t = exp(−τ × cos(out_t, ema_out)) × (1 + w × surp_t)

WHY NEAREST-NEIGHBOR NOT MEAN:
    Mean similarity is dominated by common tokens. The MAX similarity test asks:
    "does this token have ANY close analogue in context?" The LOF concept applied
    to sequence tokens: outlier = has no nearby neighbors.

COST: O(B × T²) self-similarity per layer — with B=32, T=64: ~131K ops. Acceptable.

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars
STATE:  _ema_out (D,) unit vector only — minimal global state
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU184(nn.Module):
    """Within-sequence nearest-neighbor cosine distance gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps          = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out = None
        self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                self._ema_out = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready   = True
            return out

        with torch.no_grad():
            x_n  = F.normalize(x.detach(), dim=-1)               # (B, T, D)
            sim  = torch.bmm(x_n, x_n.transpose(1, 2))           # (B, T, T)
            # Mask diagonal: self-similarity = 1.0, set it to -2 so max ignores it
            eye  = torch.eye(T, device=x.device, dtype=torch.bool).unsqueeze(0)
            sim  = sim.masked_fill(eye, -2.0)
            nn_sim, _ = sim.max(dim=-1)                           # (B, T)
            # novelty: map nn_sim from [-1,1] to [1,0]
            novelty   = (1.0 - nn_sim.clamp(-1, 1)) * 0.5        # (B, T) ∈ [0,1]
            surprise  = torch.tanh(sigma * novelty)

            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            gate_cos = torch.exp(-tau * (out_n * ema_n).sum(-1).clamp(-1, 1))

        output = out * (gate_cos * (1.0 + w * surprise)).unsqueeze(-1)

        with torch.no_grad():
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out = d_val * self._ema_out + (1-d_val) * F.normalize(om, dim=0)

        return output
