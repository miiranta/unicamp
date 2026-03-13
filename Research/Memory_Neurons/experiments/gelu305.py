"""GELU305 – gelu211 with Differentiable EMA Decay (No detach on d_val).

gelu211's key weakness: `d_val = sigmoid(logit_decay).detach().item()`
The decay parameter receives NO gradient at all — it's completely detached.

FIX: compute the EMA update IN the computational graph using d as a tensor,
so that backprop can adjust how fast the EMA tracks vs. reacts to novelty.

The EMA update itself must stay in no_grad (we don't want to unroll the full
history), but d_val now gets gradient from gate computations through the
z-score path:

    z = (x - ema_mean) / std    ← ema_mean is a buffer, but d shapes how it was built
    
Actually, since ema_mean is only updated in no_grad, the main gradient path is:
    gate = f(z)  where z is computed with detached x
    output = out * gate → loss → backward → grad to beta, gamma params

The real fix: use self.logit_decay (not detached) in a SOFT STOP-GRADIENT:
    d_soft = torch.sigmoid(self.logit_decay - self.logit_decay.detach() + self.logit_decay.detach())
This is just sigmoid(x) with a straight-through-style gradient.

BETTER: make the gate directly depend on learnable decay by computing a
one-step lookahead gate that's differentiable in d:

    mu_next = d * ema_mean + (1-d) * batch_mean
    z_pred  = (x - mu_next) / std   ← now d gets gradient from gate shape

PARAMS: logit_decay (differentiable), log_tau, log_beta_up, log_beta_dn,
        log_gamma, log_beta_out, log_gamma_out  (7 params, same as gelu211)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU305(nn.Module):
    """gelu211 with differentiable EMA decay: d gets gradient via one-step lookahead z-score."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # d_val is DIFFERENTIABLE — sigmoid(logit_decay) keeps gradient
        d_val     = torch.sigmoid(self.logit_decay)        # scalar tensor, grad enabled
        d_scalar  = d_val.detach().item()                  # for buffer update only

        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        # One-step lookahead z-score: mu_pred = d*ema + (1-d)*batch_mean
        # This makes logit_decay differentiable through the z-score normalization
        xf_batch = x.detach().flatten(0, 1)
        of_batch = out.detach().flatten(0, 1)
        batch_mean_x   = xf_batch.mean(0)          # (D,)
        batch_mean_out = of_batch.mean(0)            # (D,)

        # Differentiable predicted mean using d_val (gradient flows to logit_decay)
        mu_pred_in  = d_val * self._ema_mean.detach()     + (1 - d_val) * batch_mean_x
        mu_pred_out = d_val * self._ema_out_mean.detach() + (1 - d_val) * batch_mean_out

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_in  = var_in.sqrt().view(1, 1, D)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            std_out = var_out.sqrt().view(1, 1, D)

        # z-scores with differentiable mean prediction
        z_in  = (x.detach()   - mu_pred_in.view(1, 1, D))  / (std_in  + self.eps)
        z_out = (out.detach() - mu_pred_out.view(1, 1, D)) / (std_out + self.eps)

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
            self._ema_mean     = d_scalar * self._ema_mean     + (1 - d_scalar) * batch_mean_x
            self._ema_sq       = d_scalar * self._ema_sq       + (1 - d_scalar) * xf_batch.pow(2).mean(0)
            self._ema_out_mean = d_scalar * self._ema_out_mean + (1 - d_scalar) * batch_mean_out
            self._ema_out_sq   = d_scalar * self._ema_out_sq   + (1 - d_scalar) * of_batch.pow(2).mean(0)
            om = of_batch.mean(0)
            self._ema_out_dir  = d_scalar * self._ema_out_dir  + (1 - d_scalar) * F.normalize(om, dim=0)

        return output
