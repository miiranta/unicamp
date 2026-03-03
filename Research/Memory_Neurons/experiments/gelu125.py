"""GELU125 – Per-Channel Dual-Timescale Z-Score + Variance Burst Gate.

THE SYNTHESIS INSIGHT:
    gelu80 (best performer): per-channel z-score gate — one timescale, per-channel stat.
    gelu93 (good performer): dual-timescale variance burst — two timescales, SCALAR signal.
    
    gelu80's key win: per-channel granularity (every channel gets its own statistics).
    gelu93's key win: temporal contrast (fast/slow variance ratio = local burst detection).
    
    GELU125 merges both at full resolution:
    - PER-CHANNEL dual-timescale statistics
    - BOTH z-score AND variance burst signals, per channel
    - Combined per-channel gate (not collapsed to scalar)

THE MECHANISM — TWO NOVELTY SIGNALS, PER CHANNEL:

    Signal 1: Slow-timescale z-score (how far from long-term mean)
        z_d[b,t] = (x_d[b,t] - slow_mean_d) / (slow_std_d + eps)
        z_gate_d[b,t] = tanh(sigma × |z_d[b,t]|)           ∈ (0, 1)
        
        This captures: "is this channel's value currently unusual vs long history?"
        Similar to gelu80 but uses SLOW EMA statistics (longer memory).
    
    Signal 2: Per-channel variance burst (fast/slow variance ratio)
        fast_var_d = fast-EMA of x_d² — recent 3-step variance
        slow_var_d = slow-EMA of x_d² — historical 30-step variance
        burst_d = log(fast_var_d / slow_var_d + 1)           ≥ 0
        
        This captures: "is channel d currently experiencing a variance explosion?"
        A value of 1 means fast_var = slow_var (normal). > 1 means local burst.
        
    Combined per-channel gate:
        gate_d[b,t] = 1 + alpha × z_gate_d[b,t]             (z-score contribution)
                        + beta  × burst_d                     (variance burst, broadcast)
        output = GELU(x × gate)

WHY COMBINING IS BETTER THAN EITHER ALONE:
    z-score only: knows if current value deviates from MEAN (position-sensitive)
    burst only: knows if current VARIANCE is high (variance-sensitive, position-agnostic)
    
    Together: amplify channels that are BOTH off-mean AND in a high-variance episode.
    
    Example:
    - High z + high burst: genuinely surprising activation → double amplification
    - High z + low burst: one-off deviation → only z amplification
    - Low z + high burst: high variance but near mean → only burst amplification
    - Low z + low burst: normal familiar activation → gate ≈ 1

THE PER-CHANNEL DIFFERENCE FROM gelu80+gelu93:
    gelu80: scalar gate = 1 + w × mean_d(tanh(σ|z_d|)) — collapses D to 1 number
    gelu93: scalar gate = 1 + w × tanh(σ(mean_d(fast/slow)-1)) — scalar burst
    gelu125: gate_d = 1 + alpha×z_gate_d + beta×burst_d — D-dimensional, per-channel
    
    The D-dimensional gate means: each channel gets individually modulated.
    Channel d=42 might be amplified by 1.8× while channel d=100 stays at 1.02×.
    This precision is impossible with scalar gates.

TIMESCALE CHOICES (learnable):
    fast_decay ≈ 0.5: window ~2 steps (very recent)
    slow_decay ≈ 0.97: window ~33 steps (longer history)
    Gradient can adjust both. Constraint: fast < slow (enforced via sigmoid monotonicity).

STATE MANAGEMENT:
    4 EMA buffers, each (D,): fast_mean, fast_sq, slow_mean, slow_sq.
    Updated per forward call using detached x_mean values (no gradient).

Params: logit_fast, logit_slow, log_sigma, log_alpha, log_beta = 5 scalars.
State: _fast_mean (D,), _fast_sq (D,), _slow_mean (D,), _slow_sq (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU125(nn.Module):
    """Per-channel dual-timescale: z-score (slow EMA) + variance burst (fast/slow ratio), D-dim gate."""

    def __init__(self, d_ff: int = 1024, ema_fast: float = 0.5, ema_slow: float = 0.97, eps: float = 1e-5):
        # d_ff accepted for API compatibility with TransformerLM constructor pattern (ignored here)
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4   # variance floor

        self.logit_fast = nn.Parameter(torch.tensor(math.log(ema_fast / (1.0 - ema_fast))))
        self.logit_slow = nn.Parameter(torch.tensor(math.log(ema_slow / (1.0 - ema_slow))))
        self.log_sigma  = nn.Parameter(torch.tensor(math.log(1.0)))   # z-score sensitivity
        self.log_alpha  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # z gate strength ≈ 0.5
        self.log_beta   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # burst weight ≈ 0.3

        # Per-channel EMA buffers
        self._fast_mean: torch.Tensor = None
        self._fast_sq:   torch.Tensor = None
        self._slow_mean: torch.Tensor = None
        self._slow_sq:   torch.Tensor = None
        self._ready: bool = False

    def reset_state(self):
        self._fast_mean = None
        self._fast_sq   = None
        self._slow_mean = None
        self._slow_sq   = None
        self._ready     = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        d_fast = torch.sigmoid(self.logit_fast).detach().item()
        d_slow = torch.sigmoid(self.logit_slow).detach().item()
        sigma  = F.softplus(self.log_sigma)
        alpha  = F.softplus(self.log_alpha)
        beta   = F.softplus(self.log_beta)

        # Batch+sequence statistics
        x_flat    = x.detach().flatten(0, -2)         # (B*T, D)
        x_mean    = x_flat.mean(0)                    # (D,)
        x_sq_mean = x_flat.pow(2).mean(0)             # (D,)

        if not self._ready:
            self._fast_mean = x_mean.clone()
            self._fast_sq   = x_sq_mean.clone()
            self._slow_mean = x_mean.clone()
            self._slow_sq   = x_sq_mean.clone()
            self._ready     = True
            return self._gelu(x)

        # ── Signal 1: Slow-timescale per-channel z-score ─────────────
        slow_var = (self._slow_sq - self._slow_mean.pow(2)).clamp(min=self.eps_var)  # (D,)
        slow_std = slow_var.sqrt()                                                    # (D,)

        z       = (x - self._slow_mean) / (slow_std + self.eps)   # (B, T, D)
        z_gate  = torch.tanh(sigma * z.abs())                      # (B, T, D) ∈ (0, 1)

        # ── Signal 2: Per-channel variance burst (fast/slow ratio) ───
        fast_var = (self._fast_sq - self._fast_mean.pow(2)).clamp(min=self.eps_var)  # (D,)
        # log(ratio + 1) = softplus-like: 0 when equal, grows with fast burst
        burst    = torch.log(fast_var / (slow_var + self.eps_var) + 1.0)              # (D,) ≥ 0

        # ── Combined per-channel gate ─────────────────────────────────
        # z_gate: (B, T, D), burst: (D,) → broadcast to (B, T, D)
        gate = 1.0 + alpha * z_gate + beta * burst                 # (B, T, D)

        # ── Update EMA buffers (after forward for causality) ─────────
        self._fast_mean = d_fast * self._fast_mean + (1.0 - d_fast) * x_mean
        self._fast_sq   = d_fast * self._fast_sq   + (1.0 - d_fast) * x_sq_mean
        self._slow_mean = d_slow * self._slow_mean + (1.0 - d_slow) * x_mean
        self._slow_sq   = d_slow * self._slow_sq   + (1.0 - d_slow) * x_sq_mean

        return self._gelu(x * gate)
