"""GELU221 – Output-Space-Only Asymmetric Per-Channel Gate.

MOTIVATION — COMPLEMENT TO gelu190:
    gelu190 (PPL 160.54) operates on INPUT z-scores:
        z_in_d = (x_d - EMA_mean(x_d)) / EMA_std(x_d)
        gate_d = 1 + β_up*ReLU(tanh(γ*z_in_d)) − β_dn*ReLU(tanh(−γ*z_in_d))

    gelu211 (PPL 159.35, new best) uses INPUT gate × OUTPUT gate (product).
    That product form is very strong but complex.

    KEY QUESTION: Is it the OUTPUT statistics that drive gelu211's improvement,
    or is it the product structure?

    gelu221 isolates the OUTPUT-only hypothesis:
        y_d   = GELU(x_d)                              (standard activation)
        z_out_d = (y_d - EMA_mean(y_d)) / EMA_std(y_d)  (output z-score)
        gate_d   = clamp(1 + β_up*ReLU(tanh(γ*z_out_d)) − β_dn*ReLU(tanh(−γ*z_out_d)), 0.05, 8)
        output   = y * gate

    This is gelu190's architecture but tracking OUTPUT rather than INPUT statistics.

WHY OUTPUT MIGHT BE BETTER:
    - Output of GELU has a specific distribution (mostly non-negative, heavy tail for large x)
    - Novelty in output space directly measures deviation from the model's typical output pattern
    - Input-space novelty can be masked by the nonlinearity (unusual input → normal output)
    - Output-space novelty: unusual output always matters, regardless of input

PARAMS: logit_decay, log_beta_up, log_beta_dn, log_gamma = 4 scalars
STATE:  _ema_out_mean (D,), _ema_out_sq (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU221(nn.Module):
    """Output-space asymmetric per-channel gate: EMA on GELU(x) statistics."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_beta_up = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        y = self._gelu(x)   # (B, T, D)

        # ── Init EMA on first call ────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                yf = y.detach().flatten(0, 1)           # (B*T, D)
                self._ema_out_mean = yf.mean(0).clone()
                self._ema_out_sq   = yf.pow(2).mean(0).clone()
            self._ready = True
            return y

        # ── Compute output z-scores ───────────────────────────────────
        mu_out  = self._ema_out_mean                    # (D,)
        sq_out  = self._ema_out_sq                      # (D,)
        var_out = (sq_out - mu_out.pow(2)).clamp(self.eps_var)
        std_out = var_out.sqrt()                        # (D,)

        z_out = (y - mu_out) / (std_out + self.eps)    # (B, T, D)

        # ── Asymmetric gate on output z-scores ────────────────────────
        up_arm  = beta_up * F.relu(torch.tanh( gamma * z_out))   # (B, T, D)
        dn_arm  = beta_dn * F.relu(torch.tanh(-gamma * z_out))
        gate    = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)

        output = y * gate

        # ── Update EMA (detached, stateful) ───────────────────────────
        with torch.no_grad():
            yf  = y.detach().flatten(0, 1)              # (B*T, D)
            m_y = yf.mean(0)
            s_y = yf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1.0 - d_val) * m_y
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1.0 - d_val) * s_y

        return output
