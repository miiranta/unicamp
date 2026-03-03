"""
GELU150 — Channel-isotropy gate (decorrelated channels carry more information).

Mutual information between a channel and a global signal (the sequence mean)
roughly measures how "redundant" that channel is. Channels that always
track the global activation are predictable = familiar.

If channel d is highly correlated with mean(x) over time, it carries
REDUNDANT information (whatever the mean predicts, d follows).
If channel d is UNCORRELATED with the mean, it carries independent,
specific information — novel and non-redundant.

    global_act(t) = mean_d(x_{b,t,d})           — per-token scalar
    corr_d = EMA_corr(x_d, global_act)          — running Pearson correlation
    
    novelty_d = 1 - |corr_d|                    — uncorrelated = specific
    gate = 1 + alpha * tanh(sigma * mean_d(novelty_d))   — per-token scalar

This has no connection to the EMA of individual channels — it measures
CORRELATION STRUCTURE between channels and global activity.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_xd (D,), _ema_g (scalar),
        _ema_xd_sq (D,), _ema_g_sq (scalar), _ema_cross (D,)
"""

import torch
import torch.nn as nn


class GELU150(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_xd",    torch.zeros(d_ff))
        self.register_buffer("_ema_xd_sq", torch.ones(d_ff))
        self.register_buffer("_ema_g",     torch.tensor(0.0))
        self.register_buffer("_ema_g_sq",  torch.tensor(1.0))
        self.register_buffer("_ema_cross", torch.zeros(d_ff))  # E[x_d * global]
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_xd.zero_()
        self._ema_xd_sq.fill_(1.0)
        self._ema_g.zero_()
        self._ema_g_sq.fill_(1.0)
        self._ema_cross.zero_()
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        # global mean activation per token, averaged over batch
        glob = x.mean(dim=-1)   # (B, T) — global activity signal

        if self._warmup:
            bm   = x.detach().mean(dim=(0, 1))        # (D,)
            bsq  = (x.detach() ** 2).mean(dim=(0, 1))
            bg   = glob.detach().mean()
            bgsq = (glob.detach() ** 2).mean()
            bc   = (x.detach() * glob.detach().unsqueeze(-1)).mean(dim=(0, 1))
            self._ema_xd.copy_(bm);   self._ema_xd_sq.copy_(bsq)
            self._ema_g.copy_(bg);    self._ema_g_sq.copy_(bgsq)
            self._ema_cross.copy_(bc)
            self._warmup = False
            return base

        # Pearson correlation: E[xd*g] - E[xd]*E[g]
        #   --------------------------
        #   sqrt(Var[xd] * Var[g])
        cov_d  = self._ema_cross - self._ema_xd * self._ema_g       # (D,)
        var_xd = (self._ema_xd_sq - self._ema_xd ** 2).clamp(min=1e-8)
        var_g  = (self._ema_g_sq  - self._ema_g  ** 2).clamp(min=1e-8)
        corr_d = cov_d / (var_xd.sqrt() * var_g.sqrt() + 1e-8)     # (D,) ∈ [-1,1]

        novelty_d = 1.0 - corr_d.abs()             # (D,) — 0=correlated, 1=independent
        surp      = novelty_d.mean()               # scalar (no grad: just used as gate scale)

        # recompute with actual token-level correlation for gate with gradient
        g_norm = glob.unsqueeze(-1)                                # (B, T, 1)
        cross_token = x * g_norm.detach()                          # (B, T, D) — per-token cross
        # compute per-token deviation of cross from expected
        token_cov = cross_token - self._ema_xd * self._ema_g      # (B, T, D)
        token_nov = 1.0 - (token_cov.abs() / (var_xd.sqrt() * var_g.sqrt() + 1e-8)).clamp(max=1.0)
        surp_token = token_nov.mean(dim=-1, keepdim=True)          # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp_token)
        out   = base * gate

        with torch.no_grad():
            d  = self._decay
            bm   = x.detach().mean(dim=(0,1))
            bsq  = (x.detach()**2).mean(dim=(0,1))
            bg   = glob.detach().mean()
            bgsq = (glob.detach()**2).mean()
            bc   = (x.detach() * glob.detach().unsqueeze(-1)).mean(dim=(0,1))
            self._ema_xd.mul_(d).add_(bm  *(1-d))
            self._ema_xd_sq.mul_(d).add_(bsq*(1-d))
            self._ema_g.mul_(d).add_(bg  *(1-d))
            self._ema_g_sq.mul_(d).add_(bgsq*(1-d))
            self._ema_cross.mul_(d).add_(bc*(1-d))

        return out
