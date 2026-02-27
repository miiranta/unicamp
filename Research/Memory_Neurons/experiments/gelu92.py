"""GELU92 – Per-Channel Inverse Activity Gate (Rare-is-Novel).

THE FREQUENCY-BASED NOVELTY PRINCIPLE:
    In language models, certain neurons are "chronic activators" — they fire strongly
    for common, high-frequency patterns. These are the familiar, overused neurons.

    Other neurons are "rare activators" — they fire only for specific unusual inputs.
    When a rare-activator neuron fires, it's carrying NOVEL INFORMATION.

    We want to:
    - Amplify: outputs from rarely-active channels (when they fire, it means something)
    - Suppress: outputs from chronically-active channels (firing is expected, low info)

MECHANISM:
    Track per-channel EMA of absolute activation: ema_act_d = EMA(|GELU(x)_d|)

    Per-channel gate:
        gate_d = ema_act_mean / (ema_act_d + eps)   ← inverse activity weighting
                                                       mean activity norm → mean(gate_d) ≈ 1

    Where ema_act_mean = mean_d(ema_act_d) = average channel activity level.

    BEHAVIOR:
        High ema_act_d (chronic activator): gate_d = small/mean = small ← suppressed
        Low ema_act_d (rare activator): gate_d = mean/small = large ← amplified

    Energy: sum(gate_d × ema_act_d) = sum(ema_act_mean) = D × ema_act_mean
    Mean gate ≈ 1 only if ema_act distribution is uniform. In practice, mean(gate) will
    depend on the activity distribution. We NORMALIZE the gate to ensure mean = 1.

    NORMALIZED VERSION:
        raw_gate_d = ema_act_mean / (ema_act_d + eps)   
        gate_d = raw_gate_d / mean_d(raw_gate_d)         ← normalize to mean = 1

    Then blend with identity for stability:
        final_gate_d = alpha × gate_d + (1 - alpha)

    output = GELU(x) × final_gate

NOTE ON STATIONARITY:
    ema_act_d tracks HOW MUCH channel d fires ON AVERAGE across the dataset.
    This is stable and meaningful: channels that encode frequent patterns will have
    high running activation; channels that encode rare patterns will have low running activation.

    The gate then amplifies the rare channels PROPORTIONALLY to their rarity.
    This is like INVERSE DOCUMENT FREQUENCY (TF-IDF) in information retrieval:
    common terms (familiar channels) get low weight; rare terms (novel channels) get high weight.

DIFFERENCE FROM gelu87:
    gelu87: gate based on CURRENT z-score (how unusual is THIS token, for this channel NOW)
    gelu92: gate based on HISTORICAL activity (which channels are ALWAYS underused?)
    gelu87 is token-specific; gelu92 is channel-specific (learned across all tokens).

Params: logit_decay, log_alpha_raw = 2 scalars.
State: _ema_act (D,) per-channel running absolute activation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU92(nn.Module):
    """Per-channel inverse activity gate: amplify rare-activating channels."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        self._ema_act: torch.Tensor = None   # (D,) per-channel mean |GELU(x)|
        self._ready = False

    def reset_state(self):
        self._ema_act = None
        self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        alpha = torch.sigmoid(self.log_alpha_raw)   # blend strength, in graph for gradient

        out = self._gelu(x)   # (B, T, D)

        if not self._ready:
            with torch.no_grad():
                self._ema_act = out.detach().abs().flatten(0, 1).mean(0).clone()
                self._ready   = True
            return out

        # ── Per-channel inverse activity gate ─────────────────────────────
        ema_mean  = self._ema_act.detach()                            # (D,) — detached
        act_mean  = ema_mean.mean()                                   # scalar mean activity

        # Raw inverse gate: high activity → small gate, low activity → large gate
        raw_gate  = act_mean / (ema_mean + self.eps)                  # (D,)
        # Normalize to mean = 1 (energy-preserving)
        gate_norm = raw_gate / (raw_gate.mean() + self.eps)           # (D,) mean ≈ 1
        # Blend toward identity with learned alpha (alpha in grad graph)
        gate_d    = alpha * gate_norm + (1.0 - alpha)                 # (D,)
        gate_d    = gate_d / (gate_d.mean() + self.eps)               # re-normalize

        result = out * gate_d.view(1, 1, D)                           # broadcast over B, T

        # ── EMA update ─────────────────────────────────────────────────
        with torch.no_grad():
            batch_act = out.detach().abs().flatten(0, 1).mean(0)
            self._ema_act = d_val * self._ema_act + (1 - d_val) * batch_act

        return result
