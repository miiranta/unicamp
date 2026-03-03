"""GELU179 – Three-Signal Orthogonal Fusion Gate.

CORE IDEA:
    Each top experiment captures one axis of novelty:
    
    Axis 1 (gelu80):  Per-channel z-score magnitude
                      "How much did each channel deviate from its global running average?"
    
    Axis 2 (gelu93):  Fast/slow variance burst ratio
                      "Is the VARIANCE now higher than its usual level? (surprise is accelerating)"
    
    Axis 3 (gelu86):  Within-sequence causal per-channel z-score
                      "How much did this token deviate from other tokens in THIS context?"
    
    These three axes are largely orthogonal:
    - Axis 1 fires for globally unusual activations regardless of local context
    - Axis 2 fires for sudden volatility increases regardless of mean deviation
    - Axis 3 fires for contextually unusual tokens regardless of global frequency
    
    A token novel along ALL THREE axes is maximally "new" by any plausible definition.
    
    FUSION: product of tanh-scaled signals with learned weights:
        surp1  = tanh(σ1 × mean_d |z_global_d|)          ∈ (0, 1)
        burst  = tanh(σ2 × mean_d (var_fast_d/var_slow_d - 1)^+)  ∈ (0, 1)
        surp3  = tanh(σ3 × mean_d |z_local_d|)            ∈ (0, 1)
        
        # Weighted product (not simple product – each axis has a learned weight)
        joint  = surp1^a1 × burst^a2 × surp3^a3           ∈ (0, 1)
        gate   = exp(-τ × cos_out) × (1 + w × joint)
    
    Where a1, a2, a3 are learned non-negative blending exponents:
        ai = softplus(log_ai) ≥ 0
    Setting ai → 0 makes that axis contribute trivially (term → 1.0^0 = 1);
    setting ai = 1 gives equal weight; larger values sharpen that axis.

PARAMS: logit_decay, logit_decay_fast, log_tau, log_sig1, log_sig2, log_sig3,
        log_w_raw, log_a1, log_a2, log_a3   = 10 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,),
        _var_fast (D,), _var_slow (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU179(nn.Module):
    """Three-signal orthogonal fusion: global z-score × variance burst × local z-score."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        # EMA decays
        self.logit_decay      = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # Fast variance decay (≈0.7) and slow (≈0.97)
        self.logit_decay_fast = nn.Parameter(torch.tensor(math.log(0.7 / 0.3)))
        self.logit_decay_slow = nn.Parameter(torch.tensor(math.log(0.97 / 0.03)))
        # Cosine gate
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        # Per-signal sensitivities
        self.log_sig1      = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # global z-score
        self.log_sig2      = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # variance burst
        self.log_sig3      = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))  # local z-score
        # Amplification
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Blending exponents (product weights): init = 1 (equal weighting)
        self.log_a1 = nn.Parameter(torch.tensor(0.0))
        self.log_a2 = nn.Parameter(torch.tensor(0.0))
        self.log_a3 = nn.Parameter(torch.tensor(0.0))

        self._ema_mean:  torch.Tensor = None   # (D,)
        self._ema_sq:    torch.Tensor = None   # (D,)
        self._ema_out:   torch.Tensor = None   # (D,)
        self._var_fast:  torch.Tensor = None   # (D,) EMA of x²
        self._var_slow:  torch.Tensor = None   # (D,) EMA of x²
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = self._ema_out = None
        self._var_fast = self._var_slow = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val      = torch.sigmoid(self.logit_decay).detach().item()
        d_fast     = torch.sigmoid(self.logit_decay_fast).detach().item()
        d_slow     = torch.sigmoid(self.logit_decay_slow).detach().item()
        tau        = self.log_tau.exp()
        sig1, sig2, sig3 = F.softplus(self.log_sig1), F.softplus(self.log_sig2), F.softplus(self.log_sig3)
        w          = F.softplus(self.log_w_raw)
        a1, a2, a3 = F.softplus(self.log_a1), F.softplus(self.log_a2), F.softplus(self.log_a3)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf  = x.detach().flatten(0, 1)
                x2f = xf.pow(2)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = x2f.mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._var_fast = x2f.mean(0).clone()
                self._var_slow = x2f.mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            xd  = x.detach()
            xdf = xd.flatten(0, 1)

            # ── Axis 1: global per-channel z-score ────────────────────────
            var_g  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_g  = var_g.sqrt()
            z1     = (xd - self._ema_mean.view(1,1,D)) / (std_g.view(1,1,D) + self.eps)
            surp1  = torch.tanh(sig1 * z1.abs().mean(-1))   # (B, T)

            # ── Axis 2: variance burst (fast/slow ratio) ───────────────────
            burst_ratio = (self._var_fast / self._var_slow.clamp(min=self.eps_var)).clamp(max=10.0)  # (D,)
            # Mean over channels, subtract 1 (centered at "normal")
            burst_score = F.relu(burst_ratio.mean() - 1.0)   # scalar ≥ 0
            # Same score for all (B,T) locations in this batch step
            surp2_scalar = torch.tanh(sig2 * burst_score)    # scalar ∈ (0, 1)

            # ── Axis 3: within-sequence causal z-score ─────────────────────
            cum_x  = torch.cumsum(xd, dim=1)
            cum_sq = torch.cumsum(xd.pow(2), dim=1)
            zeros1 = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            mu_c   = torch.cat([zeros1, cum_x[:, :-1]], dim=1)
            sq_c   = torch.cat([zeros1, cum_sq[:, :-1]], dim=1)
            cnt    = torch.arange(0, T, device=x.device, dtype=x.dtype).view(1, T, 1).clamp(min=1)
            mu_l   = mu_c / cnt
            sq_l   = sq_c / cnt
            var_l  = (sq_l - mu_l.pow(2)).clamp(min=self.eps_var)
            std_l  = var_l.sqrt()
            z3     = (xd - mu_l) / (std_l + self.eps)
            z3[:, 0, :] = 0.0
            surp3  = torch.tanh(sig3 * z3.abs().mean(-1))   # (B, T)

            # ── Weighted product via exponents ─────────────────────────────
            # Protect against 0^a when signal is exactly 0 (via small clamping)
            s1 = surp1.clamp(min=1e-7).pow(a1.detach())
            s2 = surp2_scalar.clamp(min=1e-7).pow(a2.detach())
            s3 = surp3.clamp(min=1e-7).pow(a3.detach())
            joint = s1 * s2 * s3                             # (B, T)

            # ── Cosine familiarity gate ────────────────────────────────────
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * joint)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ──────────────────────────────────────────
        with torch.no_grad():
            x2f = xdf.pow(2)
            self._ema_mean = d_val  * self._ema_mean + (1-d_val)  * xdf.mean(0)
            self._ema_sq   = d_val  * self._ema_sq   + (1-d_val)  * x2f.mean(0)
            self._var_fast = d_fast * self._var_fast  + (1-d_fast) * x2f.mean(0)
            self._var_slow = d_slow * self._var_slow  + (1-d_slow) * x2f.mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val  * self._ema_out  + (1-d_val)  * F.normalize(om, dim=0)

        return output
