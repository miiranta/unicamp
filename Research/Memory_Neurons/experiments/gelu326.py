"""GELU326 – Cosine-Only Gate Ablation (No Z-Score Gates).

gelu211 has three gate components:
    gate_in  (asymmetric z-score on input)
    gate_out (symmetric z-score on output)
    gate_cos (scalar cosine similarity to EMA direction)

QUESTION: How much of gelu211's PPL improvement comes from the cosine gate alone?
    - gelu190 (PPL=160.54) uses: gate_in × gate_cos      (no output z-score)
    - gelu211 (PPL=159.35) uses: gate_in × gate_out × gate_cos

THIS EXPERIMENT: only gate_cos, no z-scores at all.
    output = GELU(x) × gate_cos

If this is close to gelu211's PPL → the z-score gates add little value.
If this is close to gelu190's PPL → it's the input z-score that matters most.
If this is MUCH WORSE → both z-score gates are essential.

Also tests whether the cosine gate alone (simplest possible state-based gate)
is sufficient for competitive performance.

PARAMS: logit_decay, log_tau  (2 — absolute minimum parameter count)
STATE:  _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU326(nn.Module):
    """Cosine-only gate ablation: simplest stateful gate, 2 params."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps         = eps
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))  # init d ≈ 0.9
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))

        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out_dir = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                of = out.detach().flatten(0, 1)
                self._ema_out_dir = F.normalize(of.mean(0), dim=0).clone()
                self._ready       = True
            return out

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_cos

        with torch.no_grad():
            of = out.detach().flatten(0, 1)
            self._ema_out_dir = d_val * self._ema_out_dir + (1 - d_val) * F.normalize(of.mean(0), dim=0)

        return output
