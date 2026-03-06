"""GELU216 – Ring Buffer Episodic Recall + Per-Channel Asymmetric Gate.

MOTIVATION — COMBINING BEST PPL (gelu190) AND BEST SEQUENTIAL ADAPTATION (gelu54):
    gelu54  (PPL 168.45, adaptation Δ=+0.030): ring buffer episodic recall gate
        — after pass 1 through test, buffer contains test episode means
        — on pass 2, tokens match buffer → low novelty → gate→1 → cleaner output

    gelu190 (PPL 160.54, adaptation Δ=−0.012): asymmetric per-channel vector gate
        — best PPL, but EMA contaminated by test data → slightly degrades on pass 2

    KEY INSIGHT: Replace gelu190's EMA-based z-scores with RING BUFFER z-scores.
    Instead of z_d = (x_d − EMA_mean_d) / EMA_std_d (contaminated by test distribution),
    use z_d = (x_d − nearest_episode_d) / EMA_std_d (episode-relative deviation).

    After pass 1:
        - Ring buffer contains test episode means
        - x_d on pass 2 is close to buffer content → z_d ≈ 0 → gate_d ≈ 1 → no distortion
        - This gives POSITIVE ADAPTATION: pass 2 ≈ GELU(x), which is ideal for familiar text

    During training:
        - Buffer tracks training episodes → z-scores measure deviation from recent context
        - Asymmetric gate amplifies novel channels, suppresses overly familiar ones
        - Better than gelu54's scalar gate because per-channel is more informative

MECHANISM:
    ring buffer: (N, D) activation means, ring-FIFO
    for each batch:
        m_curr = mean_{BT}(x)                      current batch mean (D,)
        nearest_episode = argmax cosine(m_curr, buffer[i])
        ema_std_d = sqrt(EMA_variance)              long-term std for scaling
        z_d = (x_d − buffer[nearest_d]) / ema_std_d  (using nearest episode as reference)
        gate_d = 1 + β_up×ReLU(tanh(γ×z_d)) − β_dn×ReLU(tanh(−γ×z_d))

    update buffer: write m_curr into next ring slot (FIFO)
    update EMA_std: update with current batch (for variance estimation only)

PARAMS: logit_decay, log_beta_up, log_beta_dn, log_gamma, log_tau
STATE:  ring buffer (N, D), _ema_sq (D,), _ema_mean (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU216(nn.Module):
    """Ring buffer episodic reference + per-channel asymmetric gate."""

    def __init__(self, buffer_size: int = 32, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self._N   = buffer_size
        self.eps  = eps
        self.eps_var = 1e-4

        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._buf:  torch.Tensor = None   # (N, D)
        self._mask: torch.Tensor = None   # (N,) bool — slot filled?
        self._ptr   = 0
        # EMA stats for variance estimation
        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._buf   = None
        self._mask  = None
        self._ptr   = 0
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                mu = xf.mean(0)
                self._buf      = mu.unsqueeze(0).expand(self._N, -1).clone()
                self._mask     = torch.zeros(self._N, device=x.device, dtype=torch.bool)
                self._ema_mean = mu.clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                # Seed first slot
                self._buf[0]   = mu
                self._mask[0]  = True
                self._ptr      = 1
                self._ready    = True
            return out

        with torch.no_grad():
            m_curr = x.detach().flatten(0, 1).mean(0)               # (D,)
            # Find nearest episode
            valid_slots = self._mask.nonzero(as_tuple=True)[0]
            if len(valid_slots) > 0:
                buf_valid = self._buf[valid_slots]                   # (M, D)
                sims = F.normalize(buf_valid, dim=-1) @ F.normalize(m_curr.unsqueeze(-1), dim=0)
                best = valid_slots[sims.argmax()]
                ref  = self._buf[best].view(1, 1, D)                # (1, 1, D)
            else:
                ref = self._ema_mean.view(1, 1, D)

            # EMA std for scaling
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            z   = (x.detach() - ref) / (std + self.eps)             # (B, T, D)

        # ── Asymmetric per-channel gate based on episode-relative z ───
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z))
        gate_vec = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)

        # ── Cosine output EMA gate (scalar) ────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

        # Update ring buffer and EMA stats
        with torch.no_grad():
            self._buf[self._ptr]  = m_curr
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
