"""GELU95 – Max-Channel Z-Score Gate (Single Outlier Sensitivity).

THE KEY DIFFERENCE FROM GELU80:
    gelu80: surp = tanh(σ × MEAN_d(|z_d|))  ← average across ALL channels
    gelu95: surp = tanh(σ × MAX_d(|z_d|))   ← maximum across ALL channels

    These two aggregations have radically different sensitivity profiles:

    MEAN aggregation (gelu80): 
        - A token is novel if MANY channels deviate
        - Even if ONE channel is wildly novel, the other D-1 normal channels dilute it
        - Example: 5 channels at |z|=10, 1019 at |z|=0 → mean = 0.049 (tiny!)

    MAX aggregation (gelu95):
        - A token is novel if ANY SINGLE channel deviates extremely  
        - Example: same scenario → max = 10 (detected!)
        - More sensitive to RARE, HIGH-MAGNITUDE deviations

WHEN WOULD MAX BE BETTER?
    In transformer FFN neurons, individual neurons have SEMANTIC roles:
    - "Named entity" neuron fires strongly for proper nouns
    - "Negation" neuron fires for negation words  
    - "Number" neuron fires for digits/quantities

    A suddenly introduced named entity ("Napoleon") would cause the "entity" neuron to
    fire extremely while other neurons remain normal.
    → mean_d(|z|) ≈ low (diluted)
    → max_d(|z|) ≈ high (detected!)

    This is the SPARSE CODE hypothesis: semantic information is encoded in specific neurons,
    and truly novel information fires a SPECIFIC neuron strongly, not all neurons moderately.

STABILITY AND SATURATION:
    max_d(|z_d|) can be large (up to >10 for extreme outliers).
    tanh(σ × 10) ≈ 1 if σ > 0.3. The tanh ensures stability.
    σ should be initialized small (σ≈0.1) so that typical max_z=3 gives tanh(0.3)≈0.29.

    Compare gelu80 init: σ≈0.3, typical mean_z≈1 → tanh(0.3)≈0.29 (same initial effect).
    So equal initialization for fair comparison.

POTENTIAL DOWNSIDE:
    max_d(|z_d|) fluctuates more than mean_d(|z_d|):
    - A SINGLE noisy channel can trigger high surprise even for familiar tokens
    - High variance of the surprise signal → noisier gate → harder optimization
    Solution: use k-th order statistic instead of max: e.g., 95th percentile |z|
    But percentile is expensive for large D. Simple softmax-max (logsumexp) approximation:
        logmax(|z|) = (1/β) × logsumexp(β × |z|) ≈ max for large β ≈ mean for small β
    With learnable β: interpolates between mean and max!

    THIS IS GELU95: surp = tanh(σ × (1/β) × logsumexp(β × |z|))
    where β (sharpness) is learned → model chooses the right aggregation!

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw, log_beta_raw = 5 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_out (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU95(nn.Module):
    """Smooth max-channel z-score gate via learned LogSumExp aggregation."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # beta: sharpness of logsumexp aggregation
        # Small β → mean aggregation; large β → max aggregation
        # Init β=1 → intermediate, let model learn
        self.log_beta_raw  = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)
        beta  = F.softplus(self.log_beta_raw)   # aggregation sharpness

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(of.mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score ────────────────────────────────────────────
        mu_b  = self._ema_mean.detach().view(1, 1, D)
        var_b = (self._ema_sq.detach() - self._ema_mean.detach().pow(2)).clamp(min=self.eps_var)
        std_b = var_b.sqrt().view(1, 1, D)
        z     = (x.detach() - mu_b) / (std_b + self.eps)              # (B, T, D)
        abs_z = z.abs()                                                 # (B, T, D)

        # ── LogSumExp aggregation (interpolates mean↔max via beta) ─────────
        # logsumexp(β × |z|, dim=-1) = β × max_approx + log(effective_count)
        # Divide by β to normalize: (1/β) × logsumexp(β × |z|) ≈ max for large β
        # Subtract log(D)/β to account for the effective count and center around mean:
        # When β→0: (1/β) × log(sum(exp(β × |z|))) ≈ (1/β) × log(D + β × mean|z| × D) ≈ mean|z|
        # When β→∞: ≈ max|z|
        # The log(D)/β term makes large β behave like max (not log(D) + max)
        log_sum   = torch.logsumexp(beta * abs_z, dim=-1)              # (B, T)  beta in graph
        log_D     = math.log(D)
        smooth_max = (log_sum - log_D) / (beta + self.eps)             # (B, T) ≈ mean when β→0
        # Clamp to be non-negative (it can be negative if all |z| are very small)
        smooth_max = smooth_max.clamp(min=0.0)

        surp = torch.tanh(sigma * smooth_max)                          # (B, T) sigma in graph

        # ── Output cosine gate ────────────────────────────────────────────
        out_norm = F.normalize(out.detach(), dim=-1)
        ema_norm = self._ema_out.detach().view(1, 1, D)
        cos_out  = (out_norm * ema_norm).sum(dim=-1)                   # (B, T)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)         # (B, T)
        result = out * gate.unsqueeze(-1)

        # ── EMA updates ───────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            of  = out.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out  = F.normalize(d_val * self._ema_out + (1 - d_val) * of.mean(0), dim=0)

        return result
