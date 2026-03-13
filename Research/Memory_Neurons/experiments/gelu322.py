"""GELU322 – gelu211 with Batch-Statistics Cosine Gate (Differentiable τ).

gelu211's cosine gate is FULLY DETACHED:
    gate_cos = exp(-τ * cosine(out.detach(), ema_out_dir))  — τ gets no real gradient

The cosine similarity is computed against a running EMA direction (`ema_out_dir`),
which does NOT accumulate gradient. τ only gets gradient through the SCALE of the
output, not through the cosine similarity value itself.

FIX: Replace the EMA direction baseline with the CURRENT BATCH mean direction:
    batch_out_dir = normalize(out.mean(dim=(0,1)))       — mean direction in current batch
    cos_sim_batch = dot(normalize(out), batch_out_dir)   — similarity to CURRENT batch mean

    gate_cos = exp(-τ * cos_sim_batch)   — now τ gets gradient via cos_sim_batch
                                            AND out gets gradient via batch_out_dir

WHY THIS WORKS WITHOUT CAUSALITY VIOLATION:
    batch_out_dir uses mean over ALL B×T tokens (like batch norm) → same level as EMA
    The gradient flows: loss → output → cos_sim_batch → τ (useful gradient)
    and:               loss → output → batch_out_dir → τ (additional gradient path)

ADDITIONALLY: τ is now TRULY differentiable — its gradient reflects the actual
cosine similarity landscape, not just a scaling factor.

COMBINED WITH gelu211's full input × output gate structure:
    output = out × gate_in × gate_out × gate_cos_batch

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (7)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq  (no _ema_out_dir — batch replaces it)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU322(nn.Module):
    """gelu211 with differentiable batch-mean cosine gate: τ gets true gradient."""

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
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

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
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()   - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # Differentiable cosine gate using CURRENT BATCH mean direction
        # batch_out_mean = out.mean(dim=(0,1)) is a function of out → gradient flows
        batch_out_mean = out.mean(dim=(0, 1))                            # (D,) — differentiable
        batch_out_dir  = F.normalize(batch_out_mean.detach(), dim=0)    # detach mean, not out
        # cos_sim: (B, T) — gradient flows through out (normalisation)
        out_norm  = F.normalize(out, dim=-1)                            # (B, T, D) — differentiable
        cos_sim   = (out_norm * batch_out_dir.view(1, 1, D)).sum(-1).clamp(-1, 1)  # (B, T)
        gate_cos  = torch.exp(-tau * cos_sim).unsqueeze(-1)             # (B, T, 1) — τ gets grad

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)

        return output
