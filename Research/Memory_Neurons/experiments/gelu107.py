"""
gelu107 – Fractal z-score: multi-scale EMA bank
─────────────────────────────────────────────────────────────────────────────
gelu80 uses a single EMA timescale.  This variant maintains THREE timescales:
    fast   decay ≈ sigmoid(-2.0) ≈ 0.12   (very recent)
    medium decay ≈ sigmoid(0.0)  ≈ 0.50
    slow   decay ≈ sigmoid(2.0)  ≈ 0.88   (long-term)

Surprise is computed at each scale and the THREE scores are combined:
    z_s = (x – ema_mean_s) / std_s     for s in {fast, medium, slow}
    surp_s = tanh(σ × mean|z_s|)
    gate = exp(–τ × cos_out) × (1 + w_f·surp_fast + w_m·surp_med + w_s·surp_slow)

This captures novelty at multiple temporal resolutions simultaneously.
Parameters: logit_fast, logit_med, logit_slow (EMA decay logits),
            log_tau, log_sigma_raw, log_wf_raw, log_wm_raw, log_ws_raw  →  8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU107(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        # EMA decay logits (learnable but .detach() for state update = no grad)
        self.logit_fast  = nn.Parameter(torch.tensor(-2.0))   # ≈ 0.12
        self.logit_med   = nn.Parameter(torch.tensor(0.0))    # ≈ 0.50
        self.logit_slow  = nn.Parameter(torch.tensor(2.0))    # ≈ 0.88
        self.log_tau     = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_wf_raw  = nn.Parameter(torch.tensor(0.0))
        self.log_wm_raw  = nn.Parameter(torch.tensor(0.0))
        self.log_ws_raw  = nn.Parameter(torch.tensor(0.0))
        for tag in ('fast', 'med', 'slow'):
            self.register_buffer(f'_ema_mean_{tag}', torch.zeros(d_model))
            self.register_buffer(f'_ema_sq_{tag}',   torch.ones(d_model))
        self.register_buffer('_ema_out', torch.zeros(d_model))
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def _z_surp(self, x, mean_buf, sq_buf, sigma):
        mean = mean_buf.detach()
        sq   = sq_buf.detach()
        std  = (sq - mean.pow(2)).clamp(min=1e-6).sqrt()
        z    = (x - mean) / std
        return torch.tanh(sigma * z.abs().mean(-1, keepdim=True))

    def forward(self, x):
        B, T, D = x.shape
        tau   = torch.exp(self.log_tau)
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        wf    = F.softplus(self.log_wf_raw)
        wm    = F.softplus(self.log_wm_raw)
        ws    = F.softplus(self.log_ws_raw)

        out = self._gelu(x)

        surp_f = self._z_surp(x, self._ema_mean_fast, self._ema_sq_fast, sigma)
        surp_m = self._z_surp(x, self._ema_mean_med,  self._ema_sq_med,  sigma)
        surp_s = self._z_surp(x, self._ema_mean_slow, self._ema_sq_slow, sigma)

        ema_out_u = self._ema_out.detach()
        ema_out_u = ema_out_u / (ema_out_u.norm() + 1e-8)
        cos_out   = (F.normalize(out, dim=-1) * ema_out_u).sum(-1, keepdim=True)

        gate   = torch.exp(-tau * cos_out) * (1.0 + wf * surp_f + wm * surp_m + ws * surp_s)
        result = out * gate

        x_flat   = x.detach().reshape(-1, D)
        out_flat = out.detach().reshape(-1, D)
        xb, xsq, ob = x_flat.mean(0), x_flat.pow(2).mean(0), out_flat.mean(0)

        df = torch.sigmoid(self.logit_fast).detach().item()
        dm = torch.sigmoid(self.logit_med).detach().item()
        ds = torch.sigmoid(self.logit_slow).detach().item()
        with torch.no_grad():
            if not self._initialised:
                for buf_m, buf_sq in [(self._ema_mean_fast, self._ema_sq_fast),
                                       (self._ema_mean_med,  self._ema_sq_med),
                                       (self._ema_mean_slow, self._ema_sq_slow)]:
                    buf_m.copy_(xb); buf_sq.copy_(xsq)
                self._ema_out.copy_(ob)
                self._initialised = True
            else:
                for (buf_m, buf_sq, d) in [
                    (self._ema_mean_fast, self._ema_sq_fast, df),
                    (self._ema_mean_med,  self._ema_sq_med,  dm),
                    (self._ema_mean_slow, self._ema_sq_slow, ds),
                ]:
                    buf_m.mul_(d).add_((1 - d) * xb)
                    buf_sq.mul_(d).add_((1 - d) * xsq)
                self._ema_out.mul_(ds).add_((1 - ds) * ob)
        return result
