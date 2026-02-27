"""
gelu100 – Prototype-based novelty gate  (episodic memory style)
─────────────────────────────────────────────────────────────────────────────
Maintains a small bank of K=4 EMA "familiarity prototypes" — centroid vectors
in ℝ^D that adapt toward frequently-seen activation patterns.

  novelty_t  = 1 – max_k cosine(x_t, prototype_k)      ∈ [0, 2]
  surp       = tanh(σ × novelty_t)
  gate       = 1 + w × surp
  result     = GELU(x) × gate

Prototype update (inside no_grad, EMA):
  winner_k   = argmax_k cosine(x_t, prototype_k)
  proto_k   ←(1–δ)·proto_k + δ·mean(x_t where winner=k)

Hypothesis: familiarity is best captured by a small episodic memory of
typical patterns rather than per-channel marginal statistics.
Parameters: log_sigma_raw, log_w_raw  →  2 scalars.  (K=4 prototypes are buffers.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_K_PROTO = 4        # number of familiarity prototypes
_PROTO_DECAY = 0.99 # EMA decay for prototype update


class GELU100(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        # K prototype vectors — randomly initialised, EMA-adapted
        self.register_buffer('prototypes', torch.randn(_K_PROTO, d_model) * 0.02)
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── cosine distance to all prototypes (keep gradient via x) ────────
        # Normalise prototypes (detached — no grad through memory)
        proto_u    = F.normalize(self.prototypes.detach(), dim=-1)  # (K, D)

        # Normalise x — gradients flow through here → sigma, w get grads
        x_norm     = F.normalize(x, dim=-1)                  # (B, T, D)
        cos_sims   = x_norm @ proto_u.T                      # (B, T, K) ∈ [-1,1]
        max_cos, _ = cos_sims.max(dim=-1, keepdim=True)      # (B, T, 1)
        novelty    = 1.0 - max_cos                           # 0 = familiar, 2 = orthogonal

        surp   = torch.tanh(sigma * novelty)
        gate   = 1.0 + w * surp
        result = out * gate

        # ── EMA prototype update (no_grad) ─────────────────────────────────
        with torch.no_grad():
            x_flat = x.detach().reshape(-1, D)   # (B*T, D)
            x_fn   = F.normalize(x_flat, dim=-1) # (B*T, D)
            cos_f  = x_fn @ F.normalize(self.prototypes, dim=-1).T  # (B*T, K)
            winner = cos_f.argmax(dim=-1)         # (B*T,)

            if not self._initialised:
                # Spread initial prototypes evenly across the first batch
                chunk = x_flat.shape[0] // _K_PROTO
                for k in range(_K_PROTO):
                    start = k * chunk
                    end   = start + chunk if k < _K_PROTO - 1 else x_flat.shape[0]
                    if end > start:
                        self.prototypes[k].copy_(x_flat[start:end].mean(0))
                self._initialised = True
            else:
                for k in range(_K_PROTO):
                    mask = (winner == k)
                    if mask.any():
                        mean_k = x_flat[mask].mean(0)
                        self.prototypes[k].mul_(_PROTO_DECAY).add_((1.0 - _PROTO_DECAY) * mean_k)

        return result
