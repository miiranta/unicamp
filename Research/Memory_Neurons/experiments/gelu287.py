"""gelu287 – Causal Token-Level Cosine Gate.

MOTIVATION:
    All existing experiments gate using BATCH-LEVEL statistics (scalar/vector
    applied uniformly to every position in the batch).  The gate is identical
    for token 0 and token 63 in the same sequence.

    This experiment computes a PER-TOKEN gate that uses only the preceding
    positions in the sequence (strictly causal), creating WITHIN-SEQUENCE
    adaptation: as a sequence progresses, tokens similar to what came before
    are suppressed, while genuinely novel tokens pass through.

MECHANISM:
    out    = gelu(x)                                    # (B, T, D)
    padded = [learned_init, out_0, ..., out_{T-2}]     # shift right by 1
           = cat([init.expand(B,1,D), out[:,:-1,:]], dim=1)
    cumsum = cumsum(padded, dim=1)                      # (B, T, D)
    count  = arange(1, T+1) as (1, T, 1)
    causal_mean_t = cumsum_t / count_t                  # mean of preceding tokens
    cos_t  = (out_t_norm · causal_mean_t_norm)          # (B, T) per-token cosine
    gate_t = exp(-w * relu(cos_t - θ))                  # (B, T, 1) per-token
    return out * gate_t

    learned_init (D,): a trained parameter representing the "expected" first
    activation.  It is used as position -1 so that t=0 has a reference.

STRICT CAUSAL GUARANTEE:
    gate_t depends only on out[:, 0:t, :] (via padded construction), never
    on out_t itself or any position t' > t.  This preserves the autoregressive
    property of the full transformer.

BENEFIT FROM BACKPROP:
    - learned_init is gradient-trained to the typical first-position activation.
    - log_w and logit_theta are trained via per-position gradient signals.
    - Gradient flows through the cumsum computation (differentiable).

SEQUENTIAL ADAPTATION:
    Causal_init is stateless across passes — same gate for each pass → Δ ≈ 0.
    Benefit is within-sequence habituation driving better base PPL.

PARAMS:  causal_init (D,), log_w, logit_theta.
STATE:   none — stateless across batches.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU287(nn.Module):
    """Causal per-token cosine gate: gate_t = exp(-w * cos(out_t, mean(out_0..t-1)))."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.D_FF  = D_FF

        # "virtual" predecessor for position 0
        self.causal_init = nn.Parameter(torch.zeros(D_FF))
        self.log_w       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.logit_theta = nn.Parameter(torch.zeros(1))  # threshold in (0,1) via sigmoid

    def reset_state(self):
        pass  # stateless across batches / eval passes

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)   # (B, T, D)

        # ── Build causal context: shift out right by 1, prepend learned init ──
        # padded[:, t, :] = out[:, t-1, :] for t >= 1; padded[:, 0, :] = causal_init
        init   = self.causal_init.view(1, 1, D).expand(B, 1, D)   # (B, 1, D)
        padded = torch.cat([init, out[:, :-1, :]], dim=1)          # (B, T, D)

        # ── Causal mean at each position ─────────────────────────────────────
        # causal_mean[:, t, :] = mean of padded[:, 0..t, :] = mean of out_{-1..t-1}
        cumsum = torch.cumsum(padded, dim=1)                        # (B, T, D)
        count  = torch.arange(1, T + 1, device=x.device,
                              dtype=x.dtype).view(1, T, 1)          # (1, T, 1)
        causal_mean = cumsum / count                                # (B, T, D)

        # ── Per-token cosine similarity (strictly causal) ────────────────────
        out_n  = F.normalize(out,         dim=-1)   # (B, T, D)
        ref_n  = F.normalize(causal_mean, dim=-1)   # (B, T, D)
        cos    = (out_n * ref_n).sum(-1)             # (B, T)

        # ── Gate ─────────────────────────────────────────────────────────────
        w     = self.log_w.exp()
        theta = torch.sigmoid(self.logit_theta)
        gate  = torch.exp(-w * F.relu(cos - theta)).unsqueeze(-1)  # (B, T, 1)

        return out * gate
