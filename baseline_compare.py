"""
baseline_compare.py
===================
Answers the key reviewer question: is CONSTRAINED adversarial training actually
better than conventional UNCONSTRAINED adversarial training against the
realizable (constrained) threat?

Trains three detectors and evaluates all on the constrained attack at eps=0.3:
  (a) Undefended               (reuses cached victim)
  (b) Unconstrained adv-trained (adversarial training on UNCONSTRAINED PGD)
  (c) Constrained  adv-trained  (our defense; reuses cached defended model)

Outputs EXP_DIR/baseline_compare.csv with clean acc + constrained-attack acc
for each, for both binary and multiclass.

Run AFTER run_experiments.py (it reuses victim_*.pt and defended_*.pt caches).
Just Run in PyCharm.
"""
import os, csv, json
import numpy as np
import torch

from model import IDSNet, DEVICE, evaluate, make_loader
from attack import ConstraintSpec, pgd_attack

# ----------------------------------------------------------------------
BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))

MODES        = ["binary", "multiclass"]
EPS          = 0.3
ALPHA        = 0.05
ATTACK_STEPS = 20
DEF_EPOCHS   = 20
DEF_STEPS    = 7
EVAL_BATCH   = 4096
# ----------------------------------------------------------------------


def regenerate_data(mode):
    import subprocess, sys
    cmd = [sys.executable, os.path.join(SRC_DIR, "preprocess.py"),
           "--data_dir", os.path.join(BASE_DIR, "data"),
           "--out_dir", OUT_DIR, "--min_class", "10"]
    if mode == "binary":
        cmd.append("--binary")
    subprocess.run(cmd, check=True)


def load_split():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    return d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"]


def load_spec():
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        return json.load(f)


@torch.no_grad()
def clean_acc(model, X, y):
    model.eval()
    xb = torch.from_numpy(X).float().to(DEVICE)
    yb = torch.from_numpy(y).long().to(DEVICE)
    out = []
    for i in range(0, len(xb), EVAL_BATCH):
        out.append(model(xb[i:i+EVAL_BATCH]).argmax(1))
    pred = torch.cat(out)
    return (pred == yb).float().mean().item()


def constrained_atk_acc(model, X, y, spec):
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    correct = total = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(model, xb, yb, spec, eps=EPS, alpha=ALPHA,
                            steps=ATTACK_STEPS, constrained=True)
        correct += (model(adv).argmax(1) == yb).sum().item(); total += yb.size(0)
    return correct / total


def adv_train(X_tr, y_tr, X_val, y_val, spec, n_classes, in_dim, constrained):
    """Train a detector with adversarial training; constrained flag selects
    the training-time attack type."""
    model = IDSNet(in_dim, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)
    w = (w / w.sum() * n_classes).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    Xt = torch.from_numpy(X_tr).float(); yt = torch.from_numpy(y_tr).long()
    idx = np.arange(len(Xt)); best = 0.0; best_state = None
    for ep in range(1, DEF_EPOCHS + 1):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]
            xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            model.eval()
            adv, _ = pgd_attack(model, xb, yb, spec, eps=EPS, alpha=ALPHA,
                                steps=DEF_STEPS, constrained=constrained)
            model.train()
            xm = torch.cat([xb, adv], 0); ym = torch.cat([yb, yb], 0)
            opt.zero_grad(); crit(model(xm), ym).backward(); opt.step()
        va, _, _ = evaluate(model, make_loader(X_val, y_val, bs=EVAL_BATCH))
        if va > best: best = va; best_state = {k: v.cpu().clone()
                                               for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model


def run_mode(mode, writer):
    print(f"\n=== {mode} ===")
    regenerate_data(mode)
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_split()
    spec_raw = load_spec(); spec = ConstraintSpec(spec_raw, DEVICE)
    n_classes = len(spec_raw["classes"]); in_dim = X_tr.shape[1]

    # (a) undefended -- reuse cache if present
    vpath = os.path.join(EXP_DIR, f"victim_{mode}.pt")
    victim = IDSNet(in_dim, n_classes).to(DEVICE)
    victim.load_state_dict(torch.load(vpath, map_location=DEVICE, weights_only=True))
    ua = clean_acc(victim, X_te, y_te); uc = constrained_atk_acc(victim, X_te, y_te, spec)
    print(f"(a) undefended            clean={ua:.4f} con-atk={uc:.4f}")
    writer.writerow([mode, "undefended", ua, uc])

    # (b) UNCONSTRAINED adversarial training (the baseline defense)
    print("(b) training unconstrained adversarial model ...")
    unc_model = adv_train(X_tr, y_tr, X_val, y_val, spec, n_classes, in_dim,
                          constrained=False)
    ba = clean_acc(unc_model, X_te, y_te); bc = constrained_atk_acc(unc_model, X_te, y_te, spec)
    print(f"(b) unconstrained adv-tr  clean={ba:.4f} con-atk={bc:.4f}")
    writer.writerow([mode, "unconstrained_advtrain", ba, bc])
    torch.save(unc_model.state_dict(),
               os.path.join(EXP_DIR, f"unconstrained_advtrain_{mode}.pt"))

    # (c) CONSTRAINED adversarial training (our defense) -- reuse cache
    dpath = os.path.join(EXP_DIR, f"defended_{mode}.pt")
    defended = IDSNet(in_dim, n_classes).to(DEVICE)
    defended.load_state_dict(torch.load(dpath, map_location=DEVICE, weights_only=True))
    ca = clean_acc(defended, X_te, y_te); cc = constrained_atk_acc(defended, X_te, y_te, spec)
    print(f"(c) constrained adv-tr    clean={ca:.4f} con-atk={cc:.4f}")
    writer.writerow([mode, "constrained_advtrain", ca, cc])


def main():
    out = os.path.join(EXP_DIR, "baseline_compare.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "defense", "clean_acc", "constrained_atk_acc"])
        for mode in MODES:
            run_mode(mode, writer)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
