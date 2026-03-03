"""GELU180 – Causal Rank-Normalized Surprise (Distribution-Free Per-Channel Novelty).

CORE IDEA:
    All prior methods assume the within-sequence or cross-batch distribution is
    approximately Gaussian and estimate its mean+variance (or use cosine similarity).
    
    But real neuron activations are often highly non-Gaussian:
    - Skewed by ReLU-like activations in upstream layers
    - Heavy-tailed from initialization
    - Bimodal after attention head competition
    
    Gaussian z-scores on non-Gaussian distributions produce misleading surprises:
    a 3σ deviation on a symmetric Gaussian is very rare, but on a skewed distribution
    the same |z| value might occur 10% of the time.

    SOLUTION: Replace the Gaussian assumption with distribution-free RANK statistics.
    
    For each token at position t, compute the CAUSAL RANK of x_{t,d} among
    all previous tokens {x_{1,d}, ..., x_{t-1,d}} in the same sequence:
    
        rank_{t,d} = (# positions s < t where x_{s,d} < x_{t,d}) / max(t-1, 1)
        
    This gives a value in [0, 1] regardless of the distribution shape.
    Extreme values (near 0 or 1) = novel; middle values = familiar.
    
    Convert to a symmetric surprise score via the signed probit transform:
        u_{t,d} = 2 × rank_{t,d} - 1  ∈ [-1, 1]   # centered
        rank_surp_{t,d} = |u_{t,d}|  ∈ [0, 1]      # = how extreme is rank?
    
    Aggregate: mean_d rank_surp_{t,d} → tanh(σ × mean_rank_surp)
    
    Combined with the standard cosine output gate (requires only 1 EMA buffer):
        gate = exp(-τ × cos(out, ema_out)) × (1 + w × surp)

WHY THIS IS NOVEL:
    - COMPLETELY distribution-free: works for any shaped distribution
    - No EMA mean/variance estimation needed per channel (uses ranks)
    - Causal: references only past tokens in same sequence (like gelu86, gelu176)
    - Rank percentiles are naturally calibrated: 95th percentile is always 5% rare
    - Complementary to gelu80's Gaussian-assumption: captures what z-scores miss

IMPLEMENTATION DETAIL:
    Computing exact ranks requires O(T²) comparisons for T tokens.
    With T=64, this is 64²=4096 comparisons per channel per sequence — manageable.
    
    Efficient: use broadcasting comparison:
        x_flat: (B, T, D)
        # Causal mask: for each position t, count positions s < t with x_s < x_t
        # out: (B, T, D) rank values

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars
STATE:  _ema_out (D,) only — no per-channel mean/variance needed!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU180(nn.Module):
    """Distribution-free causal rank surprise gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps          = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out: torch.Tensor = None   # (D,) unit vector
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
            xd = x.detach()   # (B, T, D)

            # ── Causal rank computation ────────────────────────────────────
            # rank_{t,d} = fraction of positions s in [0, t-1] where x_{s,d} < x_{t,d}
            # Use broadcasting: compare every (t, s) pair, apply causal mask
            
            # x_t: (B, T, D) → expand to (B, T, 1, D) [query positions]
            # x_s: (B, T, D) → expand to (B, 1, T, D) [reference positions]
            x_query = xd.unsqueeze(2)     # (B, T, 1, D)
            x_ref   = xd.unsqueeze(1)     # (B, 1, T, D)
            
            # less_than[b, t, s, d] = 1 if x_{b,s,d} < x_{b,t,d}
            less_than = (x_ref < x_query).float()   # (B, T, T, D)
            
            # Causal mask: only count positions s < t
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype),
                                     diagonal=-1)   # (T, T) — 1 for s < t, 0 otherwise
            causal_mask = causal_mask.view(1, T, T, 1)   # broadcast over B, D
            
            # Count valid positions s < t with x_s < x_t
            rank_count = (less_than * causal_mask).sum(dim=2)    # (B, T, D)
            
            # Number of valid reference positions: t (positions 0..t-1)
            n_refs = torch.arange(T, device=x.device, dtype=x.dtype)   # (T,)
            n_refs = n_refs.view(1, T, 1).clamp(min=1)
            
            rank = rank_count / n_refs   # (B, T, D) ∈ [0, 1]
            
            # Position 0 has no prior → rank = 0.5 (neutral, not surprising)
            rank[:, 0, :] = 0.5
            
            # Convert rank to extremeness score: |2*rank - 1| ∈ [0, 1]
            # 0 = median (familiar), 1 = extreme min/max (novel)
            rank_surprise = (2.0 * rank - 1.0).abs()   # (B, T, D)
            
            mean_rank_surp = rank_surprise.mean(dim=-1)  # (B, T)
            surprise = torch.tanh(sigma * mean_rank_surp) # (B, T)

            # ── Cosine familiarity gate ────────────────────────────────────
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA output direction ────────────────────────────────────
        with torch.no_grad():
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out = d_val * self._ema_out + (1-d_val) * F.normalize(om, dim=0)

        return output
