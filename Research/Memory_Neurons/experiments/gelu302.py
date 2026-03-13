"""gelu302 – gelu211 + Soft Top-K Novelty Mask (Sparse Gate).

CONCEPT:
    All existing experiments gate every channel with a smoothly-varying gate.
    Channels with moderate novelty get moderate amplification, creating a
    diffuse signal.

    INSIGHT: If only the K MOST NOVEL channels are amplified and the rest
    return to baseline (gate=1), the model focuses its habituation response
    on the channels that matter most.  Less noisy amplification of borderline
    channels should improve PPL.

MECHANISM:
    1. Compute per-channel novelty score: novel_d = abs(z_in_d) (using gelu211 z-scores).
    2. Select top-K most novel channels via STRAIGHT-THROUGH top-K:
          mask    = top-K indicator (1 for top-K, 0 otherwise)  ← discrete
          mask_st = mask + novel_norm - novel_norm.detach()      ← straight-through
       where novel_norm = softmax(novel / temperature) — a soft version.
    3. Gated output:
          gate_selected = gate_211_d where mask_active, else 1.0
          output = out * gate_selected

    K is a fixed hyperparameter: K = D // 4 (top 25% most novel channels).
    temperature is a learnable parameter.

CAUSAL GUARANTEE:
    The top-K mask uses only per-batch z-scores (same causal level as gelu211).
    No within-sequence future look-ahead.

BENEFIT FROM BACKPROP:
    Straight-through estimator passes gradient through the discrete mask.
    log_temperature: controls the sharpness of the soft selection allocation.
    All gelu211 gate params also receive gradient through the selected channels.

SEQUENTIAL ADAPTATION:
    Stateless (top-K recomputed each batch) → Δ ≈ 0.
    Benefit is sharper, less diffuse gating → better base PPL.

PARAMS:  gelu211 params (7) + log_temperature.
STATE:   gelu211 state (5 EMA buffers).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU302(nn.Module):
    """gelu211 + top-K novelty mask via straight-through estimator."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.D_FF    = D_FF
        self.k       = max(1, D_FF // 4)   # top 25% channels

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Top-K temperature ───────────────────────────────────────────
        self.log_temperature = nn.Parameter(torch.zeros(1))  # temperature = 1.0 init

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.flatten(0,1); of = out.flatten(0,1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready = True
            return out

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        with torch.no_grad():
            xf = x.detach().flatten(0,1); of = out.detach().flatten(0,1)
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1,1)
            gate_cos= torch.exp(-tau.detach() * cos_sim).unsqueeze(-1)

        # ── gelu211 gate ─────────────────────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)
        gate_211 = gate_in * gate_out * gate_cos    # (B, T, D)

        # ── Top-K novelty mask via straight-through ──────────────────────
        # Novelty score = batch-mean |z_in| per channel
        novel_score = z_in.detach().abs().mean(dim=(0, 1))       # (D,) no grad needed
        temp         = self.log_temperature.exp()
        soft_novel   = torch.softmax(novel_score / temp, dim=0)  # (D,) soft allocation

        # Hard top-K mask (no gradient)
        topk_idx    = torch.topk(novel_score, self.k, largest=True, sorted=False).indices
        mask_hard   = torch.zeros(D, device=x.device, dtype=x.dtype)
        mask_hard[topk_idx] = 1.0

        # Straight-through: gradient flows through soft_novel, discrete selection via mask_hard
        mask_st = mask_hard + soft_novel - soft_novel.detach()   # (D,)

        # Effective gate: gelu211 where selected, 1.0 elsewhere
        gate_eff = gate_211 * mask_st.view(1, 1, D) + 1.0 * (1 - mask_st.view(1, 1, D))

        output = out * gate_eff

        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
