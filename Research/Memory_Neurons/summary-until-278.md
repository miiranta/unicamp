# Memory Neurons — Experiment Results

**Baseline (control): 172.34 ppl**

Groups are sorted by their best (lowest) test perplexity.
Within each group, experiments are also sorted by test perplexity (best first).

---

## 1. Per-Channel Asymmetric Vector Gate — best: 159.35

Applies a **separate multiplicative gate to each of the D channels**, with one learned coefficient for amplification (z > 0, channel above its mean) and a different one for suppression (z < 0, channel below its mean). Introduced in gelu181 as a vector gate; gelu190 added asymmetry; gelu211 combined input and output z-score spaces.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu211 | 159.35 | Input × output asymmetric product gate (combines gelu190 on input z-scores with gelu198 on output z-scores) |
| gelu190 | 160.54 | Asymmetric bidirectional gate: β_up ≠ β_dn, independent control of amplification vs suppression |
| gelu181 | 161.09 | Signed per-channel z-score vector gate: gate_d = clamp(1 + β·tanh(γ·z_d), 0.1, 5.0) |
| gelu205 | 161.10 | Dual-space sparse intersection: gates only channels in top-K by both input and output z-score |
| gelu198 | 161.89 | Output-space per-channel vector gate: z-scores computed on GELU(x) instead of x |
| gelu214 | 161.93 | Asymmetric gate with learned soft dead-zone; near-zero z_d → gate = 1.0 exactly |
| gelu202 | 162.16 | Sparse top-K=16 signed gate (variant of gelu189, K channels fully gated, rest pass-through) |
| gelu219 | 162.52 | Selective EMA: only updates memory on surprising tokens to reduce test contamination |
| gelu193 | 162.70 | Joint per-channel deviation product: gates channels unusual in BOTH input and output space |
| gelu204 | 162.83 | Sparse top-K=8 gate (ultra-sparse K ablation) |
| gelu207 | 162.91 | Sparse dual gate: amplifies top-K novel channels and suppresses bottom-K familiar channels |
| gelu201 | 163.09 | Output-space per-channel z-score gate (architectural repeat of gelu198) |
| gelu218 | 163.43 | Fast-adaptive EMA for quicker test-distribution shift detection |
| gelu189 | 163.48 | Sparse top-K=16 signed gate (foundation sparse-gate experiment) |
| gelu212 | 163.64 | Multi-scale (fast/slow EMA) asymmetric per-channel gate |
| gelu220 | 163.66 | Global EMA + local batch dual-pathway asymmetric gate |
| gelu217 | 163.75 | Within-sequence nearest-neighbor gate × asymmetric historical gate (hybrid) |
| gelu203 | 163.80 | Sparse top-K=32 signed gate (double-K ablation of gelu189) |
| gelu186 | 164.41 | Absolute z-score vector gate: gate_d = 1 + β·tanh(γ·\|z_d\|), always ≥ 1 (amplify-only) |
| gelu206 | 164.65 | Sparse top-K=16 absolute gate (sign-blind amplify-only ablation) |
| gelu213 | 164.91 | Sparse top-K=16 asymmetric gate applied to output-space z-scores |
| gelu196 | 165.09 | Pre-GELU per-channel modulation: GELU(x_d × (1 + α·tanh(γ·z_d))) |
| gelu216 | 166.50 | Ring buffer episodic recall × asymmetric historical gate (test-adaptation hybrid) |
| gelu215 | 166.55 | Dual-space sparse asymmetric intersection: asymmetric gate on top-K input ∩ top-K output channels |
| gelu208 | 169.80 | SiLU activation + sparse top-K=16 signed gate (activation function ablation) |

---

## 2. Per-Channel Z-Score → Scalar Novelty Gate — best: 162.57

Computes per-channel z-scores z_d = (x_d − μ_d) / σ_d but **collapses them to a single scalar surprise signal** (mean\|z_d\|, max\|z_d\|, weighted combinations, etc.) that then multiplies into `(1 + w·surp) × exp(−τ·cos_out)`. Gelu71 introduced scalar surprise × cosine gate; gelu80 introduced per-channel z-score aggregation; subsequent experiments explored aggregation strategies and statistical refinements.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu177 | 162.57 | Random-projection Mahalanobis: captures joint channel co-deviations via random projections |
| gelu188 | 163.06 | Relative per-channel surprise: \|z_d\| normalised by EMA of \|z_d\| (persistent noisy channels discounted) |
| gelu184 | 163.55 | Within-sequence nearest-neighbor distance: suppresses tokens with a near-duplicate in context |
| gelu81  | 163.64 | K=2 dual-prototype cosine + scalar input surprise |
| gelu182 | 163.78 | Per-position EMA: separate μ_t, σ²_t for each sequence position |
| gelu90  | 164.06 | Pre-GELU gate: GELU(x × gate) instead of GELU(x) × gate |
| gelu192 | 164.18 | Output-magnitude-weighted mean \|z\|: dead/near-zero channels de-weighted |
| gelu93  | 164.19 | Dual-timescale variance burst: fast/slow EMA variance ratio as surprise |
| gelu82  | 164.26 | Combined input + output surprise signal, no contrast normalisation |
| gelu197 | 164.29 | Variance-of-z-scores: spread of z_d distribution (not mean) as surprise |
| gelu200 | 164.41 | Surprise momentum: habituates if high surprise persists (normalise by EMA of surprise) |
| gelu83  | 164.41 | Perpendicular-deviation surprise: \|dev_⊥\|² (component orthogonal to mean direction) |
| gelu80  | 164.45 | Per-channel z-score + output cosine gate; parent experiment for this family |
| gelu185 | 164.61 | Channel-group z-score: G=8 groups, one scalar gate per group of D/8 channels |
| gelu85  | 164.62 | Per-channel z-score × asymmetric output cosine gate |
| gelu180 | 164.74 | Causal rank-normalised surprise: distribution-free (rank percentile instead of z-score) |
| gelu86  | 164.78 | Within-sequence causal z-score (no cross-batch EMA state) |
| gelu84  | 164.79 | Mahalanobis surprise × cosine gate (per-channel variance normalisation) |
| gelu95  | 164.85 | Max-channel z-score: surp = tanh(σ × max_d\|z_d\|) instead of mean |
| gelu88  | 164.86 | Relative surprise: current z-score normalised by running mean of past surprises |
| gelu71  | 165.07 | Surprise-amplified cosine gate; parent experiment introducing surprise × cosine structure |
| gelu179 | 165.14 | Three-signal fusion: per-channel z-score × variance burst × output cosine |
| gelu191 | 165.15 | Dual-speed EMA: MAX(surp_fast, surp_slow) as the scalar surprise signal |
| gelu195 | 165.20 | Log-scale surprise: log(x_d / μ_d) for multiplicative rather than additive deviations |
| gelu79  | 165.23 | Temporal velocity surprise: EMA drift speed as the surprise signal |
| gelu187 | 165.59 | Learned per-channel importance weights (softmax α_d) for z-score aggregation |
| gelu183 | 165.60 | Dual-space z-score: combines input-space and output-space surprise scores |
| gelu78  | 165.78 | Asymmetric cosine gate: anti-correlated tokens (cos < 0) amplified above 1.0 |
| gelu194 | 165.88 | Anti-trend surprise: deviation that opposes the direction EMA is currently drifting |
| gelu77  | 165.94 | Output-space surprise (\|\|GELU(x) − ema_out\|\|) × output cosine gate |
| gelu178 | 166.51 | MLP on full \|z\| vector replaces fixed mean aggregation |
| gelu176 | 167.06 | Dual-reference: min(global EMA, local seq) per-channel z-score |
| gelu109 | 167.57 | Second-order temporal novelty: surprise from EMA drift rate, not displacement |
| gelu99  | 167.62 | Signed asymmetric scalar: separate w_up/w_dn for positive vs negative z-scores |
| gelu106 | 167.72 | Exponential surprise: exp(σ·mean\|z\|) instead of tanh squashing |
| gelu140 | 168.73 | Homeostatic gate: rescales gate so its running average stays near 1.0 |
| gelu98  | 169.23 | Post-activation z-score: EMA statistics tracked on GELU(x), not x |
| gelu102 | 169.41 | Position-aware z-score: T=64 independent per-position EMA buffers |
| gelu112 | 169.51 | Variance spike: per-token internal activation spread vs EMA of spread |
| gelu116 | 169.71 | Channel-wise asymmetric EMA: different decay logits for low/high channel halves |
| gelu107 | 170.27 | Fractal multi-scale: fast/medium/slow EMA z-scores combined with cosine gate (8 params) |
| gelu74  | 170.37 | Dual-axis surprise: input + output deviation ratio × cosine gate |
| gelu97  | 171.15 | Channel-attention z-score: learned per-channel softmax weights α_d |

---

## 3. Ring Buffer / Episodic Adaptation — best: 164.36

Stores batch-level activation means in a **ring buffer**; on subsequent evaluation passes, tokens whose representations are close to stored entries are recognised as familiar and suppressed. The primary aim is test-time adaptation (improving PPL on second/third passes), though PPL on the first pass also improves.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu229 | 164.36 | Hard sigmoid suppression gate: near-binary familiar/novel switch |
| gelu230 | 164.64 | Familiarity floor: gate pushed to near-zero for very high cosine familiar content |
| gelu234 | 165.75 | Self-detecting pass-2 gate (soft exp): zero gate effect on pass 1, activates on pass 2 |
| gelu232 | 166.14 | Hard sigmoid + large buffer N=512 (covers full evaluation pass) |
| gelu236 | 166.64 | N=512 large buffer with stronger suppression (higher τ, α initialisation) |
| gelu276 | 166.66 | gelu211 backbone + frozen-EMA novelty release for improved adaptation |
| gelu238 | 167.00 | gelu211 backbone + eval-only ring buffer gate |
| gelu233 | 167.68 | Self-detecting pass-2 gate (hard sigmoid): zero gate effect on pass 1 |
| gelu54  | 168.45 | Ring buffer with hard nearest-episode cosine gate; parent experiment |
| gelu231 | 168.54 | Large buffer N=512 (original gelu54 gate, just bigger buffer) |
| gelu278 | 169.28 | Predictive coding: stored activation injects prediction to assist the model |
| gelu235 | 169.31 | gelu211 backbone + self-detecting episodic pass-2 adaptation |
| gelu240 | 169.44 | Per-token depletion gate (token-level cosine to nearest buffer entry) |
| gelu262 | 169.46 | EMA self-similarity gate: no ring buffer; pure EMA drift-toward-test-set |
| gelu227 | 170.84 | Soft-attention episodic retrieval (vs hard argmax nearest in gelu54) |
| gelu237 | 171.14 | N=512 + soft training gate / hard eval gate (bridges training-eval sharpness gap) |
| gelu253 | 182.60 | Pre-fire depletion (gate suppresses tokens even on pass 1 for next-pass benefit) — worse |
| gelu274 | 210.38 | gelu211 backbone + linear pass-counter gate — catastrophic failure |

---

## 4. Output Cosine Gate — best: 168.01

Gates the GELU output based on **cosine similarity between the current output vector and a stored EMA of past output directions**. High cosine → familiar → suppress. Gelu25 introduced post-GELU gating; gelu28 formalised the cosine-on-output structure that all subsequent variants extend.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu38 | 168.01 | Per-channel frequency habituation: gates each channel by its EMA firing rate |
| gelu28 | 168.18 | Output-cosine EMA gate: `scale = (1−α) + α·exp(−τ·cos(out, ema_out))`; parent experiment |
| gelu25 | 169.55 | Output-side EMA gate; introduced post-GELU gating structure |
| gelu36 | 170.00 | Dual-context: `sim = max(cos(out, ema_out), cos(out, seq_mean))`; global + local familiarity |
| gelu33 | 170.02 | Learned suppression curve shape: sigmoid(−w·sim − b) replaces fixed exp(−τ·sim) |
| gelu34 | 170.24 | Multi-head familiarity: H=4 independent cosine gates over D/H subspaces |

---

## 5. Double Cosine Gate — best: 169.09

Gates with **two independent cosine terms** — one against the EMA of the input direction and one against the EMA of the output direction — so familiarity in either space contributes to suppression.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu69 | 169.09 | Contrast-normalised: gate_norm = gate_raw / mean(gate_raw) so mean gate = 1.0 and novel tokens are actively amplified |
| gelu76 | 169.77 | Double cosine × per-channel z-score surprise boost: `gate × (1 + w·surp)` |

---

## 6. EMA Prototype / Memory Bank — best: 169.19

Maintains **K stored prototype vectors** (either EMA-tracked or gradient-trained) and measures familiarity as cosine or cosine² similarity against those prototypes, capturing familiarity with any of the K learned directions rather than just the mean.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu61 | 169.19 | K=16 EMA vectors; familiarity = mean cosine²(out, v_k); approximates second-moment covariance memory |
| gelu12 | 169.62 | Winner-take-all EMA prototype clustering: hard assignment to nearest prototype |
| K=1   | 172.18 | Single-prototype ablation (K=1) — near-control |
| gelu8  | 175.26 | Gradient-trained K=4 prototype bank: prototypes learned end-to-end by backprop — worse than control |

---

## 7. Standalone & Isolated Mechanisms — best: 170.53

Architecturally distinct approaches that did not evolve into multi-experiment families.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu56 | 170.53 | Tsodyks-Markram synaptic depression: biologically-derived resource-depletion model |
| gelu7  | 171.59 | Batch discriminativeness: per-channel batch variance as novelty; fully stateless |
| gelu66 | 172.34 | Shunting (divisive) inhibition by EMA conductance — ties control |
| gelu53 | 172.53 | FFT spectral whitening across D feature channels — slightly worse than control |

---

## 8. Per-Channel Input EMA Habituation — best: 170.61

Tracks per-channel EMA statistics **on the GELU input** (pre-nonlinearity) and scales each channel proportionally to its novelty relative to its own history.

| Experiment | Test PPL | Notes |
|---|---|---|
| gelu10 | 170.61 | Sign-agreement familiarity: `tanh(β·x_d)·tanh(β·μ_d) > 0` → familiar; direction-based |
| gelu5  | 171.32 | Magnitude-deviation familiarity: `exp(−τ·\|x_d − μ_d\| / ρ_d)` per channel |
| gelu11 | 172.86 | Deviation amplification: enhanced = x + α·d·(x−μ); relies on GELU nonlinearity to suppress — worse than control |

---

## Summary

| Rank | Group | Best PPL | vs baseline |
|---|---|---|---|
| 1 | Per-Channel Asymmetric Vector Gate | 159.35 | ✓ (−13.0) |
| 2 | Per-Channel Z-Score → Scalar Gate | 162.57 | ✓ (−9.8) |
| 3 | Ring Buffer / Episodic Adaptation | 164.36 | ✓ (−8.0) |
| 4 | Output Cosine Gate | 168.01 | ✓ (−4.3) |
| 5 | Double Cosine Gate | 169.09 | ✓ (−3.3) |
| 6 | EMA Prototype / Memory Bank | 169.19 | ✓ (−3.2) |
| 7 | Standalone & Isolated Mechanisms | 170.53 | ✓ (−1.8) |
| 8 | Per-Channel Input EMA Habituation | 170.61 | ✓ (−1.7) |
| — | **Control (baseline)** | **172.34** | — |
