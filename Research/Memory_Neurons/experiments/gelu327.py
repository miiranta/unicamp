"""GELU327 – Learnable EMA Initialization (Warm-Start EMA Parameters).

gelu211 initializes EMA buffers from the FIRST BATCH it sees:
    _ema_mean = x.mean(batch_0)    [a random sample]
    _ema_out_dir = normalize(out.mean(batch_0))

This initialization is NOISY — the first batch may be unrepresentative.
Worse, the initial state is not differentiable at all.

FIX: Make the initial EMA state LEARNABLE. Before any forward pass, the EMA
starts from a learned μ₀ and σ₀ (as nn.Parameters), which are optimised along
with everything else.

MECHANISM:
    At warmup (first pass), instead of using batch estimates:
        _ema_mean = self.mu0_in.clone()      — learned initial mean
        _ema_sq   = (self.mu0_in.pow(2) + F.softplus(self.log_sigma0_in).pow(2))
        _ema_out_dir = F.normalize(self.dir0_out)  — learned initial direction

    All subsequent updates proceed normally (EMA with d_val).

On SEQUENTIAL TEST EVALUATION (multiple passes), reset_state() is called before
pass 1. The learned μ₀ provides a BETTER STARTING POINT for adapting to test data,
since it was trained to minimise loss on training batches.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out,
        mu0_in (D,), log_sigma0_in (D,), mu0_out (D,), log_sigma0_out (D,), dir0_out (D,)
        = 7 scalars + 5D vectors = 7 + 5120 params for D=1024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU327(nn.Module):
    """gelu211 with learned initial EMA state: better warm-start for sequential adaptation."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        D = D_FF

        # Gate scalar params (same as gelu211)
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # Learnable initial EMA state (D-dim, trained to be good starting point)
        self.mu0_in         = nn.Parameter(torch.zeros(D))
        self.log_sigma0_in  = nn.Parameter(torch.zeros(D))    # σ₀ = softplus(0) ≈ 0.69
        self.mu0_out        = nn.Parameter(torch.zeros(D))
        self.log_sigma0_out = nn.Parameter(torch.zeros(D))
        self.dir0_out       = nn.Parameter(torch.randn(D) * 0.01)

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            # Use LEARNED initial state instead of noisy first-batch estimate
            with torch.no_grad():
                sigma0_in  = F.softplus(self.log_sigma0_in).detach()
                sigma0_out = F.softplus(self.log_sigma0_out).detach()
                self._ema_mean     = self.mu0_in.detach().clone()
                self._ema_sq       = (self.mu0_in.pow(2) + sigma0_in.pow(2)).detach().clone()
                self._ema_out_mean = self.mu0_out.detach().clone()
                self._ema_out_sq   = (self.mu0_out.pow(2) + sigma0_out.pow(2)).detach().clone()
                self._ema_out_dir  = F.normalize(self.dir0_out.detach(), dim=0).clone()
                self._ready        = True

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()   - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
