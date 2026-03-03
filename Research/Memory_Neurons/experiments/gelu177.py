"""GELU177 – Random-Projection Mahalanobis Surprise.

CORE IDEA:
    gelu80's per-channel z-score treats each dimension INDEPENDENTLY.
    It captures: "did channel d deviate from its own mean?"
    
    But real novelty often involves JOINT deviations: channels i and j always
    move together, so seeing i up + j down is novel even if each individually
    is within normal range. This is the off-diagonal covariance structure.

    Full Mahalanobis distance requires estimating the D×D covariance matrix —
    infeasible for D=1024 (1M params per layer, unstable estimation).

    SOLUTION: Random projection approximation of Mahalanobis.
    
    Johnson-Lindenstrauss: K random unit vectors {w_k} ∈ R^D (fixed, no grad)
    capture pairwise distances up to distortion with K = O(log D).
    
    For each projection k:
        proj_k(x) = w_k · x   ∈ R           scalar projection
        z_k       = (proj_k(x) - EMA_μ_k) / (EMA_σ_k + ε)   per-sample z
    
    Aggregate: mean_k |z_k| across K=64 projections.
    
    Each projection mixes channels → captures cross-channel covariance signals
    that are INVISIBLE to per-channel tests (gelu80), since w_k selectively
    amplifies correlated channel combinations.

WHY THIS COMPLEMENTS GELU80:
    - gelu80 is sensitive to MARGINAL departures (one channel at a time)
    - gelu177 is sensitive to JOINT departures (unusual channel combinations)
    - The two are provably orthogonal: joint z=0 can have non-zero marginals and vice versa
    - Together they form a richer novelty signal

IMPLEMENTATION:
    (B, T, D) × (D, K) = (B, T, K)   projected features, batched matmul
    EMA over K-dimensional statistics (much smaller than D)
    Same cosine output gate as gelu80 for directional familiarity

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars
STATE:  _ema_mu (K,), _ema_sq (K,), _ema_out (D,), _proj (D, K) fixed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


K_PROJ = 64   # number of random projections


class GELU177(nn.Module):
    """Random-projection Mahalanobis surprise gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5, n_proj: int = K_PROJ):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.n_proj   = n_proj
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # Fixed random projection matrix — registered as buffer (no grad, moves with model)
        # Initialized lazily in forward() once we know D
        self._proj:   torch.Tensor = None   # (D, K)
        self._ema_mu: torch.Tensor = None   # (K,)
        self._ema_sq: torch.Tensor = None   # (K,)
        self._ema_out:torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mu  = None
        self._ema_sq  = None
        self._ema_out = None
        self._ready   = False
        # Keep _proj — it's a fixed random matrix, resetting it would change the semantics

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

        # Lazy initialization of random projection matrix (once D is known)
        if self._proj is None or self._proj.shape[0] != D:
            with torch.no_grad():
                proj = torch.randn(D, self.n_proj, device=x.device, dtype=x.dtype)
                proj = F.normalize(proj, dim=0)   # unit columns
                self._proj = proj
                self._ready = False

        if not self._ready:
            with torch.no_grad():
                xf   = x.detach().flatten(0, 1)          # (B*T, D)
                pf   = xf @ self._proj                   # (B*T, K)
                self._ema_mu  = pf.mean(0).clone()
                self._ema_sq  = pf.pow(2).mean(0).clone()
                self._ema_out = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready   = True
            return out

        with torch.no_grad():
            # ── Random projection z-score ──────────────────────────────────
            xd   = x.detach().flatten(0, 1)              # (B*T, D)
            proj = (xd @ self._proj).view(B, T, -1)      # (B, T, K)

            var_k  = (self._ema_sq - self._ema_mu.pow(2)).clamp(min=self.eps_var)
            std_k  = var_k.sqrt()
            mu_k_  = self._ema_mu.view(1, 1, -1)         # (1, 1, K)
            std_k_ = std_k.view(1, 1, -1)
            z_k    = (proj - mu_k_) / (std_k_ + self.eps) # (B, T, K)
            mean_abs_z = z_k.abs().mean(dim=-1)            # (B, T)
            surprise   = torch.tanh(sigma * mean_abs_z)    # (B, T)

            # ── Cosine familiarity gate ────────────────────────────────────
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics (projections only) ──────────────────────
        with torch.no_grad():
            xf   = x.detach().flatten(0, 1)
            pf   = xf @ self._proj                        # (B*T, K)
            self._ema_mu  = d_val * self._ema_mu  + (1-d_val) * pf.mean(0)
            self._ema_sq  = d_val * self._ema_sq  + (1-d_val) * pf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out = d_val * self._ema_out + (1-d_val) * F.normalize(om, dim=0)

        return output
