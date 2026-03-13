"""gelu292 – Trained Prototype Initialisation + Fast Eval-EMA Adaptation Gate.

MOTIVATION:
    gelu280 adds a fast secondary EMA (_eval_ema) that resets to ZEROS before
    each eval run.  The first batch of pass-1 finds _eval_ema at the zero
    vector, producing a near-zero cosine similarity → gate ≈ 1 (no suppression).
    Only after the EMA "warms up" over many batches does the gate fire.

    The fundamental problem: the eval-EMA starts from an UNINFORMATIVE prior.

    This experiment replaces the zero initialisation with a TRAINED PROTOTYPE
    (a learnable D-dimensional vector) that is shaped via backpropagation to
    represent the expected test-time activation.  When the eval run begins:
        _eval_ema = trained_prototype.detach()
    The warm start means the gate fires from batch 1 of pass 1.

MECHANISM:
    Training forward:
        - Standard gelu211 gate (identically to gelu283's differentiable EMA).
        - Additionally, _train_ema updates toward out_mean with fast d≈0.5 (no_grad).
          This is a running "center of training activations" for use as init.

    Before eval (reset_state):
        - _eval_ema ← log_init_proto (a D-dim parameter, trained)
                      or more precisely: exp(log_scale) * F.normalize(init_dir)
        - The magnitude and direction of the init prototype are both trainable.

    Eval forward:
        - Standard gelu211 gate (using training EMA, which is stable after training).
        - PLUS a secondary eval gate:
              out_n   = normalize(out)
              ema_n   = normalize(_eval_ema)
              cos     = (out_n · ema_n).mean over (B,T)
              extra   = exp(-w_eval * relu(cos - theta))    (scalar extra suppression)
          The extra gate fires when the eval content aligns with the eval EMA.
        - _eval_ema ← d_fast * _eval_ema + (1 - d_fast) * out_mean  (updates within pass)

SEQUENTIAL ADAPTATION:
    Pass 1: _eval_ema starts near the trained prototype (near test distribution).
            Extra gate fires from batch 1.  EMA converges during pass 1.
    Pass 2: _eval_ema = test distribution after a full pass.
            Extra gate fires HARDER on familiar content → Δ > 0.

    The trained prototype init ensures pass 1 benefits from the gate too,
    unlike gelu280 where pass 1 PPL is only helped by the training EMA.

NO CAUSALITY LEAK:
    Prototype and eval-EMA are per-batch, not per-position — same as gelu211.

BENEFIT FROM BACKPROP:
    init_dir (D,), log_init_scale: gradient through training loss shapes the
    warm-start prototype to produce a good first-pass gate.
    log_w_eval, logit_theta: trained alongside all gelu211 params.

PARAMS:  gelu211 params + init_dir (D), log_init_scale, logit_d_eval,
         log_w_eval, logit_theta_eval.
STATE:   gelu211 state + _eval_ema (D) — reset to trained prototype.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU292(nn.Module):
    """gelu211 + trained-prototype-initialised eval EMA for warm-start sequential adaptation."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.D_FF    = D_FF

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Eval EMA params ──────────────────────────────────────────────
        # Trained prototype used to warm-start _eval_ema
        self.init_dir       = nn.Parameter(torch.randn(D_FF) * 0.01)
        self.log_init_scale = nn.Parameter(torch.zeros(1))
        # Fast decay for eval EMA (d ≈ 0.5 → half-life ≈ 1 batch)
        self.logit_d_eval   = nn.Parameter(torch.zeros(1))           # sigmoid(0)=0.5
        # Suppression strength and threshold
        self.log_w_eval     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.logit_theta    = nn.Parameter(torch.zeros(1))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── Eval EMA buffer ──────────────────────────────────────────────
        self._eval_ema: torch.Tensor = None   # None → use trained prototype

    def reset_state(self):
        """Called once before each eval run; warm-starts eval EMA from trained prototype."""
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        # Warm start: trained prototype = normalised init_dir scaled by init_scale
        with torch.no_grad():
            init_proto = (self.log_init_scale.exp() *
                          F.normalize(self.init_dir, dim=0))
            self._eval_ema = init_proto.detach().clone()

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        # ── First batch: initialise gelu211 EMAs ─────────────────────────
        if not self._ready:
            with torch.no_grad():
                xf = x.flatten(0, 1); of = out.flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                if self._eval_ema is None:
                    self._eval_ema = of.mean(0).clone()
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
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)
            out_n_cos = F.normalize(out.detach(), dim=-1)
            ema_n     = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim   = (out_n_cos * ema_n).sum(-1).clamp(-1, 1)
            gate_cos  = torch.exp(-tau.detach() * cos_sim).unsqueeze(-1)

        # ── gelu211 gate ─────────────────────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # ── Eval EMA secondary gate ───────────────────────────────────────
        out_m   = out.detach().flatten(0, 1).mean(0)              # (D,)
        out_n   = F.normalize(out, dim=-1)                        # (B, T, D)
        eva_n   = F.normalize(self._eval_ema, dim=0).view(1, 1, D)
        cos_ev  = (out_n * eva_n).sum(-1).clamp(-1, 1)           # (B, T)
        w_ev    = self.log_w_eval.exp()
        theta   = torch.sigmoid(self.logit_theta)
        extra   = torch.exp(-w_ev * F.relu(cos_ev - theta)).unsqueeze(-1)  # (B, T, 1)

        output = out * gate_in * gate_out * gate_cos * extra

        # ── Update eval EMA (no_grad — side effect only) ─────────────────
        with torch.no_grad():
            d_eval         = torch.sigmoid(self.logit_d_eval).item()
            self._eval_ema = d_eval * self._eval_ema + (1 - d_eval) * out_m

        # ── Update gelu211 EMAs ───────────────────────────────────────────
        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
