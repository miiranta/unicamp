"""
Dynamic Graph Neurons  –  train.py
══════════════════════════════════════════════════════════════════════════════
Trains DGN1, DGN2, DGN3 on WikiText-2 and saves per-epoch metrics.
Reuses the same dataset, hyperparameters, and evaluation as Memory_Neurons
for direct comparison.

Usage:
    python train.py                # runs all experiments sequentially
    python train.py --exp dgn1     # runs only one experiment
══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, math, csv, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import Config, DGNLanguageModel

from experiments.dgn1 import DGN1
from experiments.dgn2 import DGN2
from experiments.dgn3 import DGN3
from experiments.dgn4 import DGN4
from experiments.dgn5 import DGN5
from experiments.dgn6 import DGN6
from experiments.dgn7 import DGN7
from experiments.dgn8 import DGN8
from experiments.dgn9 import DGN9


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset  (identical to Memory_Neurons)
# ──────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<unk>": 0, "<eos>": 1}
        self.idx2word = ["<unk>", "<eos>"]

    def add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, word):
        return self.word2idx.get(word, 0)


def load_wikitext2(data_dir: str):
    vocab = Vocabulary()
    splits = {}
    for split in ("train", "valid", "test"):
        path = os.path.join(data_dir, f"wiki.{split}.tokens")
        tokens = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                for w in words:
                    vocab.add(w)
                    tokens.append(w)
                tokens.append("<eos>")
        splits[split] = tokens
    # encode
    for k in splits:
        splits[k] = [vocab.encode(w) for w in splits[k]]
    return vocab, splits


class SequenceDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        # truncate to multiple of seq_len
        n = (len(tokens) // seq_len) * seq_len
        self.data = torch.tensor(tokens[:n], dtype=torch.long)

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, i):
        start = i * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              grad_clip: float, desc: str = ""):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    phase = "train" if train else "eval "
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        bar = tqdm(enumerate(loader, 1), total=len(loader),
                   desc=f"  {desc or phase}", leave=False, unit="batch", dynamic_ncols=True)
        for step, (x, y) in bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)                                    # (B, T, V)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
            avg = total_loss / total_tokens
            bar.set_postfix(loss=f"{avg:.4f}",
                            ppl=f"{math.exp(min(avg, 20)):.2f}")
    avg_loss = total_loss / total_tokens
    elapsed  = time.time() - t0
    return avg_loss, math.exp(avg_loss), elapsed


# ──────────────────────────────────────────────────────────────────────────────
#  Experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(name: str, dgn_cls, vocab, train_loader, val_loader, test_loader,
                   device, cfg: Config):
    out_dir      = os.path.join(cfg.OUTPUT_DIR, name)
    metrics_path = os.path.join(out_dir, "metrics.csv")

    print(f"\n{'='*60}")
    print(f"  Experiment : {name}")
    print(f"  DGN class  : {dgn_cls.__name__}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    model     = DGNLanguageModel(len(vocab), cfg, dgn_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # Warm-up forward pass to catch shape errors / materialise lazy params
    with torch.no_grad():
        dummy = torch.zeros(1, cfg.SEQ_LEN, dtype=torch.long, device=device)
        model(dummy)
    for m in model.modules():
        if hasattr(m, 'reset_state'):
            m.reset_state()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}\n")

    # Save model info
    with open(os.path.join(out_dir, "model_info.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["n_params"], [n_params]])

    # Open metrics CSV and write header — append row by row during training
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl",
                                 "train_time", "val_time"])

    best_val_ppl = float('inf')

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"── Epoch {epoch}/{cfg.EPOCHS} ──")
        tr_loss, tr_ppl, tr_t  = run_epoch(model, train_loader, criterion, optimizer,
                                           device, train=True,  grad_clip=cfg.GRAD_CLIP)
        vl_loss, vl_ppl, vl_t  = run_epoch(model, val_loader,   criterion, optimizer,
                                           device, train=False, grad_clip=cfg.GRAD_CLIP)
        scheduler.step()

        print(f"  train loss {tr_loss:.4f} | ppl {tr_ppl:.2f} | {tr_t:.1f}s")
        print(f"  valid loss {vl_loss:.4f} | ppl {vl_ppl:.2f} | {vl_t:.1f}s")

        # Append row immediately (crash-safe)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_ppl, vl_loss, vl_ppl,
                                    round(tr_t, 2), round(vl_t, 2)])

        if vl_ppl < best_val_ppl:
            best_val_ppl = vl_ppl
            print(f"  → new best model (val ppl {vl_ppl:.2f})")

        print()

    # ── Test evaluation (last epoch model, same as Memory_Neurons) ────
    print("── Test ──")
    te_loss, te_ppl, te_t = run_epoch(model, test_loader, criterion, optimizer,
                                      device, train=False, grad_clip=cfg.GRAD_CLIP)
    print(f"  test loss {te_loss:.4f} | ppl {te_ppl:.2f} | {te_t:.1f}s\n")

    with open(os.path.join(out_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_loss", "test_ppl"])
        w.writerow([te_loss, te_ppl])


# ──────────────────────────────────────────────────────────────────────────────
#  All experiments
# ──────────────────────────────────────────────────────────────────────────────
ALL_EXPERIMENTS = [
    ("dgn1", DGN1),   # fixed K=8, cosine similarity top-K, binary adjacency
    ("dgn2", DGN2),   # novelty-adaptive K: novel=16, familiar=2 connections
    ("dgn3", DGN3),   # R=3 looped rounds, topology recomputed from current state
    ("dgn4", DGN4),   # dual-graph: top-K similar + bottom-K dissimilar, learned gate
    ("dgn5", DGN5),   # EdgeDrop: stochastic Bernoulli edge masking during training
    ("dgn6", DGN6),   # growing K per round: 4→8→16 progressive expansion
    ("dgn7", DGN7),   # attention-weighted messages within binary top-K neighbourhood
    ("dgn8", DGN8),   # contrastive + progressive combo (DGN4 × DGN6)
    ("dgn9", DGN9),   # adaptive per-token gate scales the graph delta
]


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, help="Run only this experiment name")
    args = parser.parse_args()

    cfg    = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ──────────────────────────────────────────────────────────
    print("Loading WikiText-2 …")
    vocab, splits = load_wikitext2(cfg.DATA_DIR)
    print(f"Vocab size: {len(vocab):,}")

    train_ds = SequenceDataset(splits["train"], cfg.SEQ_LEN)
    val_ds   = SequenceDataset(splits["valid"], cfg.SEQ_LEN)
    test_ds  = SequenceDataset(splits["test"],  cfg.SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ── run experiments ───────────────────────────────────────────────────
    for name, dgn_cls in ALL_EXPERIMENTS:
        if args.exp is not None and args.exp != name:
            continue
        run_experiment(name, dgn_cls, vocab, train_loader, val_loader, test_loader, device, cfg)

    # ── summary ──────────────────────────────────────────────────────────
    print("\n\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    print(f"  {'Experiment':<12}  {'Final val PPL':>14}  {'Best val PPL':>12}  {'Test PPL':>10}")
    print("  " + "─" * 56)
    for name, _ in ALL_EXPERIMENTS:
        mp = os.path.join(cfg.OUTPUT_DIR, name, "metrics.csv")
        tp = os.path.join(cfg.OUTPUT_DIR, name, "test_metrics.csv")
        if os.path.exists(mp):
            with open(mp) as f:
                rows = list(csv.DictReader(f))
            final_ppl = float(rows[-1]["val_ppl"])
            best_ppl  = min(float(r["val_ppl"]) for r in rows)
            test_ppl  = float(list(csv.DictReader(open(tp)))[0]["test_ppl"]) if os.path.exists(tp) else float('nan')
            print(f"  {name:<12}  {final_ppl:>14.2f}  {best_ppl:>12.2f}  {test_ppl:>10.2f}")
        else:
            print(f"  {name:<12}  {'—':>14}  {'—':>12}  {'—':>10}")
    print()


if __name__ == "__main__":
    main()
