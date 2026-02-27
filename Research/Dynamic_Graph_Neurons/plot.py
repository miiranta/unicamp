"""
Dynamic Graph Neurons  –  plot.py
══════════════════════════════════════════════════════════════════════════════
Generates training-curve, test-result, and parameter-count plots from the
metrics saved by train.py.

Usage:
    python plot.py
══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Colour per experiment (extend if more dgn variants are added)
PALETTE = {
    "dgn1": "#4C72B0",
    "dgn2": "#55A868",
    "dgn3": "#C44E52",
    "dgn4": "#8172B2",
    "dgn5": "#CCB974",
    "dgn6": "#64B5CD",
}
DEFAULT_COLOR = "#888888"


def pretty(name: str) -> str:
    """dgn1 → DGN-1, dgn12 → DGN-12, etc."""
    if name.startswith("dgn") and name[3:].isdigit():
        return f"DGN-{name[3:]}"
    return name.upper()


def color_of(name: str) -> str:
    return PALETTE.get(name, DEFAULT_COLOR)


# ── Load all experiment metrics ───────────────────────────────────────────────
def _sort_key(name: str):
    if name.startswith("dgn") and name[3:].isdigit():
        return int(name[3:])
    return 9999


exp_names = sorted(
    [e for e in os.listdir(OUTPUT_DIR)
     if os.path.isfile(os.path.join(OUTPUT_DIR, e, "metrics.csv"))],
    key=_sort_key,
)

if not exp_names:
    print("No metrics.csv files found in output/. Run train.py first.")
    raise SystemExit(0)

records = []
for exp_name in exp_names:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, exp_name, "metrics.csv"))
    df["experiment"] = exp_name
    df["label"]      = pretty(exp_name)
    records.append(df)

data = pd.concat(records, ignore_index=True)
data.to_csv(os.path.join(PLOTS_DIR, "training_curves.csv"), index=False)
print(f"Saved training data → {os.path.join(PLOTS_DIR, 'training_curves.csv')}")

# ── Training curves (2×2) ─────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
palette = {pretty(n): color_of(n) for n in exp_names}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("DGN Training Curves — WikiText-2", fontsize=14, fontweight="bold")

for ax, col, ylabel, title in [
    (axes[0, 0], "train_loss", "Loss",       "Train Loss"),
    (axes[0, 1], "val_loss",   "Loss",       "Validation Loss"),
    (axes[1, 0], "train_ppl",  "Perplexity", "Train Perplexity"),
    (axes[1, 1], "val_ppl",    "Perplexity", "Validation Perplexity"),
]:
    for exp_name in exp_names:
        sub = data[data["experiment"] == exp_name]
        lbl = pretty(exp_name)
        ax.plot(sub["epoch"], sub[col], label=lbl, color=color_of(exp_name),
                linewidth=2, marker="o", markersize=3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "training_curves.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── Test results ───────────────────────────────────────────────────────────────
test_records = []
for exp_name in exp_names:
    test_path = os.path.join(OUTPUT_DIR, exp_name, "test_metrics.csv")
    if not os.path.isfile(test_path):
        continue
    df = pd.read_csv(test_path)
    df["experiment"] = exp_name
    df["label"]      = pretty(exp_name)
    test_records.append(df)

if not test_records:
    print("No test_metrics.csv files found — skipping test plots.")
    raise SystemExit(0)

test_data = pd.concat(test_records, ignore_index=True)[
    ["experiment", "label", "test_loss", "test_ppl"]
].copy()

best_ppl  = test_data["test_ppl"].min()
worst_ppl = test_data["test_ppl"].max()

# Relative improvement vs worst performer (since there's no fixed control)
test_data["ppl_vs_worst%"] = 100 * (worst_ppl - test_data["test_ppl"]) / worst_ppl

test_csv = os.path.join(PLOTS_DIR, "test_results.csv")
test_data.to_csv(test_csv, index=False)
print(f"Saved test data  → {test_csv}")

# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST RESULTS")
print("=" * 60)
print(f"  {'Experiment':<14} {'Loss':>7} {'PPL':>9} {'vs worst%':>10}")
print("  " + "-" * 44)
for _, row in test_data.sort_values("test_ppl").iterrows():
    marker = "  ◀ best" if row["test_ppl"] == best_ppl else ""
    print(f"  {row['label']:<14} {row['test_loss']:>7.4f} {row['test_ppl']:>9.2f}"
          f" {row['ppl_vs_worst%']:>9.1f}%{marker}")
print("=" * 60 + "\n")

# ── Bar chart of test results ─────────────────────────────────────────────────
plot_df = test_data.sort_values("test_ppl").reset_index(drop=True)
n = len(plot_df)
bar_colors = [color_of(e) for e in plot_df["experiment"]]

fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * n + 2)))
fig.suptitle("DGN Test Results — WikiText-2", fontsize=13, fontweight="bold")

for ax, col, pct_col, xlabel, title in [
    (axes[0], "test_loss", "ppl_vs_worst%", "Cross-Entropy Loss",  "Test Loss"),
    (axes[1], "test_ppl",  "ppl_vs_worst%", "Perplexity",          "Test Perplexity"),
]:
    bars = ax.barh(plot_df["label"], plot_df[col], color=bar_colors,
                   edgecolor="white", height=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    span = plot_df[col].max() - plot_df[col].min() + 1e-9
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        val = row[col]
        pct = row["ppl_vs_worst%"]
        label_str = f"{val:.2f}  ({pct:+.1f}%)" if pct != 0 else f"{val:.2f}  (worst)"
        ax.text(val + 0.015 * span, bar.get_y() + bar.get_height() / 2,
                label_str, va="center", ha="left", fontsize=9)
    ax.set_xlim(plot_df[col].min() - 0.05 * span,
                plot_df[col].max() + 0.35 * span)
    # Highlight best
    best_val = plot_df[col].min()
    ax.axvline(best_val, color="green", linewidth=1.2, linestyle="--",
               alpha=0.6, label="best")
    ax.legend(fontsize=8)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "test_results.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── Parameter count ───────────────────────────────────────────────────────────
param_records = []
for exp_name in exp_names:
    info_path = os.path.join(OUTPUT_DIR, exp_name, "model_info.csv")
    if not os.path.isfile(info_path):
        continue
    df = pd.read_csv(info_path)
    param_records.append({
        "experiment": exp_name,
        "label":      pretty(exp_name),
        "n_params":   int(df["n_params"].iloc[0]),
    })

if param_records:
    param_df = pd.DataFrame(param_records)
    p_colors = [color_of(e) for e in param_df["experiment"]]
    n = len(param_df)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * n + 2)))
    fig.suptitle("Trainable Parameters per DGN Variant",
                 fontsize=13, fontweight="bold")

    bars = ax.barh(param_df["label"], param_df["n_params"],
                   color=p_colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Number of trainable parameters")
    ax.invert_yaxis()
    span = param_df["n_params"].max() - param_df["n_params"].min() + 1
    for bar, val in zip(bars, param_df["n_params"]):
        ax.text(val + 0.015 * span, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, param_df["n_params"].max() + 0.25 * span)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "param_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")

    print("\n" + "=" * 40)
    print("  PARAMETER COUNTS")
    print("=" * 40)
    for _, row in param_df.iterrows():
        print(f"  {row['label']:<12} {row['n_params']:>12,}")
    print("=" * 40)
else:
    print("No model_info.csv files found — skipping parameter count plot.")
