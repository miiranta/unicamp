"""GELU178 – Learned Z-Score Aggregation (MLP on |z| vector).

CORE IDEA:
    gelu80 computes per-channel z-scores correctly, but aggregates them with a
    FIXED formula: surp = tanh(σ × mean_d |z_d|).
    
    The flat average treats all D channels equally. In reality:
    - Some channels are highly predictive of semantic novelty
    - Some channels are noisy / irrelevant to the task
    - Some COMBINATIONS of channels carry structured novelty information
    
    SOLUTION: Learn the aggregation from data.
    
    Pass |z_d| ∈ R^D through a tiny bottleneck MLP:
        surp_raw = MLP(|z|)    where MLP: D → H → 1 (H=32, single value out)
        surp     = sigmoid(surp_raw)    ∈ (0, 1)
    
    The MLP learns:
    - Which channels matter for detecting real novelty (vs noise)
    - Which channel COMBINATIONS signal semantic novelty
    - All of this is supervised by the language modeling loss end-to-end

ARCHITECTURE:
    MLP: Linear(D, H, bias=False) → SiLU → Linear(H, 1, bias=True) → sigmoid
    H = 32 (small enough to be fast, large enough for useful combinations)
    
    Important: z-scores are computed WITHOUT gradient (from EMA stats).
    The MLP receives |z|.detach() — this way gradients only flow through:
        1. The MLP weights (learning the aggregation)
        2. GELU(x) (the activation function)
    
    This avoids unstable gradients through the EMA division.

WHY H=32 AND NOT LARGER:
    D=128 in this setup (cfg.D_MODEL). H=32 gives:
    - D×H = 128×32 = 4096 weights, H×1 = 32 — ~4K extra params per layer
    - With 4 transformer layers, 4×4K = 16K extra params total (~1.5% overhead)
    - Large enough for non-trivial grouping / selection of channels

INITIALIZATION:
    First Linear init to uniform(0, 1/D) so initial output ≈ mean|z|/D × H
    → behaves like gelu80 at initialization, then learns to deviate.

PARAMS: logit_decay, log_tau, log_w_raw + MLP params ≈ ~16K per 4-layer model
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MLP_HIDDEN = 32


class GELU178(nn.Module):
    """Learned MLP aggregation of per-channel z-scores."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5, mlp_hidden: int = MLP_HIDDEN):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.mlp_hidden = mlp_hidden

        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_w_raw   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # MLP: lazily initialized once D is known
        self._mlp: nn.Sequential = None
        self._D_last: int = -1

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def _build_mlp(self, D: int, device, dtype):
        """Build tiny D → H → 1 MLP once we know D."""
        lin1 = nn.Linear(D, self.mlp_hidden, bias=False)
        lin2 = nn.Linear(self.mlp_hidden, 1, bias=True)
        # Init lin1 small so initial surp ≈ mean|z| * const
        nn.init.normal_(lin1.weight, std=1.0 / math.sqrt(D))
        nn.init.zeros_(lin2.weight)
        nn.init.zeros_(lin2.bias)
        mlp = nn.Sequential(lin1, nn.SiLU(), lin2)
        mlp = mlp.to(device=device, dtype=dtype)
        # Register parameters so optimizer picks them up
        self.add_module('_mlp_module', mlp)
        return mlp

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

        # Lazily build MLP (once D is known, after first forward)
        if self._mlp is None or self._D_last != D:
            self._mlp   = self._build_mlp(D, x.device, x.dtype)
            self._D_last = D
            self._ready  = False

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score (no grad through stats) ───────────────────
        with torch.no_grad():
            var   = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std   = var.sqrt()
            mu_   = self._ema_mean.view(1, 1, D)
            std_  = std.view(1, 1, D)
            z     = (x.detach() - mu_) / (std_ + self.eps)   # (B, T, D)
            abs_z = z.abs()                                    # (B, T, D) — detached

        # ── MLP aggregation: receives |z| detached → gradients flow into MLP ──
        abs_z_flat  = abs_z.reshape(B * T, D)                 # (B*T, D)
        surp_raw    = self._mlp(abs_z_flat).view(B, T)         # (B, T)
        surprise    = torch.sigmoid(surp_raw)                  # (B, T) ∈ (0,1)

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ──────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
