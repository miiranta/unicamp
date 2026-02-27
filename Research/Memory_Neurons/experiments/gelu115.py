"""
gelu115 – Sparse code gate  (k-winners novelty)
─────────────────────────────────────────────────────────────────────────────
Biological inspiration: sparse coding. A token is novel if its ACTIVE SET
(top-K channels by |x|) differs from the running expected active set.

    active_t  = top-K indicator vector: 1 if |x_d| is in top K, else 0   (D,)
    ema_prob_d = EMA(freq of channel d being in top-K over all tokens)     (D,)
    expected_surp_t = mean_d(active_t,d × (1 – ema_prob_d))  (mean surprise of active set)
    surp_t     = tanh(σ × expected_surp_t)   (high when rare channels fire)
    gate_t     = 1 + w × surp_t
    result     = GELU(x) × gate

When rarely-activated channels fire, that token gets amplified (informative).
When only commonly-activated channels fire, that token gets normal treatment.
K = D//4 = 32 for D=128.
Parameters: logit_decay, log_sigma_raw, log_w_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_K_FRAC = 4   # D // K_FRAC channels active


class GELU115(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.K = max(1, d_model // _K_FRAC)   # =32 for D=128
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        # Running probability of each channel being in top-K
        self.register_buffer('_ema_prob', torch.full((d_model,), float(self.K) / d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── sparse active set surprise ─────────────────────────────────────
        # Get top-K indices (from detached x to avoid topk in autograd)
        _, idx      = x.detach().abs().topk(self.K, dim=-1)   # (B,T,K)
        # Build indicator via scatter (one-hot style)
        active      = torch.zeros(B, T, D, device=x.device)
        active.scatter_(-1, idx, 1.0)                          # (B,T,D) binary

        ema_prob    = self._ema_prob.detach()                  # (D,)
        rarity      = 1.0 - ema_prob                          # (D,) rarity score
        # expected surprise = mean rare-ness of active channels
        # keep sigma in grad path: scale surp with sigma via tanh
        raw_surp    = (active * rarity).sum(-1, keepdim=True) / self.K  # (B,T,1) [0,1]
        surp        = torch.tanh(sigma * raw_surp)
        gate        = 1.0 + w * surp
        result      = out * gate

        # ── EMA update on channel activation frequency ─────────────────────
        active_flat = active.detach().reshape(-1, D)
        freq_batch  = active_flat.mean(0)                      # (D,) mean freq
        decay       = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_prob.copy_(freq_batch)
                self._initialised = True
            else:
                self._ema_prob.mul_(decay).add_((1 - decay) * freq_batch)
        return result
