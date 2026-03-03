"""GELU189 – Sparse Top-K Signed Vector Gate (K=16).

THE PROBLEM WITH DENSE VECTOR GATES (gelu181, gelu186):
    In gelu181: ALL D=1024 channels compute z_d and get individually gated.
    gate_vec (B, T, D) modulates every channel independently.

    But most channels will have z_d ≈ 0 (near historical mean) → gate_d ≈ 1 (no-op).
    The dense gate is doing 1024 multiplications, but only ~K carry real information.
    The rest is noise that may DILUTE the signal from the truly surprising channels.

THE NEW IDEA: Sparse Top-K Gate
    Find the K=16 channels with the LARGEST |z_d|:
        top_k_indices = argtopk(|z_d|, K)              — (B, T, K) indices

    For the top-K channels, apply a SIGNED gate (like gelu181):
        gate_d = clamp(1 + β × tanh(γ × z_d), 0.1, 8.0)

    For all other D-K channels: gate_d = 1.0 (no-op)

    Combined with the scalar cosine gate as before.

WHY K=16:
    D=1024, K=16 → K/D = 1.56% sparsity
    In GELU activations, a small fraction of channels are truly activated (≠ 0).
    The k-NN sparse structure is analogous to MoE routers and k-sparse autoencoders
    where keeping top-k activations produces sparse, disentangled representations.

GEOMETRIC INTERPRETATION:
    We don't rotate the GELU(x) vector along all dimensions (noisy).
    We only rotate it along the K most surprising dimensions (signal).
    This is like a sparse rotation in the D-dimensional activation space.

NOTE ON GRADIENT:
    argtopk is not differentiable, but the gate values at selected indices ARE.
    Gradients flow through gate_d at the selected positions (straight-through style).
    The selection mask is computed without grad; only gate values have grad.

PARAMS: logit_decay, log_tau, log_beta, log_gamma = 4 scalars (same as gelu181)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 16   # number of most-surprising channels to gate


class GELU202(nn.Module):
    """Sparse top-K=16 asymmetric gate: separate beta_up/beta_dn on K most surprising channels."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k       = K
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
        k = min(self.k, D)

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var   = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std   = var.sqrt().view(1, 1, D)
            mu_   = self._ema_mean.view(1, 1, D)
            z     = (x.detach() - mu_) / (std + self.eps)  # (B, T, D) signed

            # ── Sparse top-K selection ─────────────────────────────────────
            _, topk_idx  = z.abs().topk(k, dim=-1)          # (B, T, K)

        # Build gate_vec starting from all-ones, then scatter top-K asymmetric values
        gate_vec = torch.ones(B, T, D, device=x.device, dtype=x.dtype)  # default: no-op
        z_topk   = torch.gather(z.detach(), -1, topk_idx)               # (B, T, K) signed z at top-K
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_topk))        # excitatory
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_topk))        # inhibitory
        g_topk   = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_vec = gate_vec.scatter(-1, topk_idx, g_topk)               # (B, T, D)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)          # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
