"""
gelu110 – Entropy-based novelty gate
─────────────────────────────────────────────────────────────────────────────
Instead of measuring how far a token is from the EMA mean (z-score),
this variant measures the SOFTMAX ENTROPY of the absolute activation values
within a single token, then compares it to the running average entropy.

    p_d     = softmax(|x_d| / τ_e)           per-token prob distribution over D
    H       = –Σ p_d log(p_d)                token entropy (high: uniform; low: peaked)
    ema_H   = EMA(batch mean H)              running average entropy
    novelty = H / (ema_H + ε)               relative entropy (1 = average)
    surp    = tanh(σ × (novelty – 1))
    gate    = 1 + w × surp                   amplify peaked (low-entropy) tokens
    result  = GELU(x) × gate

Low-entropy token = most activation concentrated in few channels = specific/novel.
High-entropy token = spread activation = less specific = more familiar.
Parameters: logit_decay, log_tau_e_raw, log_sigma_raw, log_w_raw  →  4 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU110(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        import math
        _H_MAX = math.log(d_model)
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_tau_e_raw = nn.Parameter(torch.tensor(0.0))   # entropy temp
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_H',   torch.tensor(_H_MAX * 0.8))  # init near max
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        tau_e  = F.softplus(self.log_tau_e_raw) + 0.1
        sigma  = F.softplus(self.log_sigma_raw) + 0.01
        w      = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── entropy of per-token activation distribution ──────────────────
        # p_d via softmax with temperature – gradients flow through tau_e, sigma, w
        logits  = x.abs() / tau_e                              # (B,T,D)
        p       = F.softmax(logits, dim=-1)                    # (B,T,D)
        H       = -(p * (p + 1e-10).log()).sum(-1, keepdim=True)  # (B,T,1)

        ema_H   = self._ema_H.detach().clamp(min=1e-3)
        novelty = H / ema_H                                    # (B,T,1)
        surp    = torch.tanh(sigma * (novelty - 1.0))
        gate    = 1.0 + w * surp
        result  = out * gate

        decay = torch.sigmoid(self.logit_decay).detach().item()
        H_batch_mean = H.detach().mean().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_H.fill_(H_batch_mean)
                self._initialised = True
            else:
                self._ema_H.mul_(decay).add_((1 - decay) * H_batch_mean)
        return result
