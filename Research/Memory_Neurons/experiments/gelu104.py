"""
gelu104 – Intra-batch novelty gate  (no cross-batch EMA state)
─────────────────────────────────────────────────────────────────────────────
All EMA-based experiments track statistics across batches (across time).
This variant is entirely STATELESS across batches: familiarity is computed
purely from the CURRENT BATCH, like BatchNorm but used for novelty gating.

    mu_b, std_b = batch_mean(x), batch_std(x)   over all BT tokens
    z_d         = (x_d – mu_b_d) / std_b_d
    surp        = tanh(σ × mean_d(|z_d|))
    gate        = 1 + w × surp
    result      = GELU(x) × gate

Noise-robust: within each batch the surprise is relative — training is never
confused by early-training shift in statistics.
No cosine gate here (no persistent direction memory).
Parameters: log_sigma_raw, log_w_raw  →  2 scalars.  Zero persistent state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU104(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── batch-level statistics (all BT tokens) ─────────────────────────
        # Detach x for statistics but keep for gate path to flow grads to sigma/w
        x_flat = x.detach().reshape(-1, x.shape[-1])       # (BT, D)
        mu_b   = x_flat.mean(0)                             # (D,)
        sq_b   = x_flat.pow(2).mean(0)
        std_b  = (sq_b - mu_b.pow(2)).clamp(min=1e-6).sqrt()

        # z has gradient through x → sigma path
        z    = (x - mu_b) / std_b                           # (B, T, D)
        surp = torch.tanh(sigma * z.abs().mean(-1, keepdim=True))  # (B,T,1)

        gate   = 1.0 + w * surp
        result = out * gate
        return result
