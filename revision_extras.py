"""
revision_extras.py
==================
Generates the remaining reviewer-requested numbers in one script:

  1. MACRO-F1 for the multiclass tables (clean / unconstrained / constrained,
     undefended vs defended).                            [R1, R2, R3]
  2. FOUR-VARIANT constraint ablation: bounds only; bounds+integrality;
     bounds+immutability; all.                            [R3]
  3. FEATURE SPECIFICATION dump: exact immutable + integer feature name lists,
     ready to paste into Table tab:features.              [R1, R2, R3]
  4. CLASS DISTRIBUTION after preprocessing, with test counts, for
     Table tab:classdist.                                 [R2, R3]

Run in multiclass mode (do NOT pass --binary to preprocess first, or run this
which regenerates multiclass data itself). Paths default to ./outputs.
"""
import os, json, sys, subprocess, csv
import numpy as np
import torch
from sklearn.metrics import f1_score
from model import IDSNet, DEVICE
from attack import ConstraintSpec, pgd_attack

BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
EPS, ALPHA, STEPS = 0.3, 0.05, 20
EVAL_BATCH = 4096


def regen_multiclass():
    subprocess.run([sys.executable, "preprocess.py", "--data_dir", "./data",
                    "--out_dir", OUT_DIR, "--min_class", "10"], check=True)


# ---------- 3 & 4: spec + class distribution (no model needed) ----------
def dump_spec_and_dist():
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        spec = json.load(f)
    cols = spec["columns"]
    imm = [c for c, m in zip(cols, spec["is_immutable"]) if m]
    integ = [c for c, m in zip(cols, spec["is_integer"]) if m]
    print("=== IMMUTABLE FEATURES ({}) ===".format(len(imm)))
    print(", ".join(imm))
    print("\n=== INTEGER FEATURES ({}) ===".format(len(integ)))
    print(", ".join(integ))
    with open(os.path.join(OUT_DIR, "feature_lists.json"), "w") as f:
        json.dump({"immutable": imm, "integer": integ}, f, indent=2)

    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    classes = spec["classes"]
    y_all = np.concatenate([d["y_tr"], d["y_val"], d["y_te"]])
    print("\n=== CLASS DISTRIBUTION (total / test) ===")
    rows = []
    for i, c in enumerate(classes):
        total = int((y_all == i).sum()); test = int((d["y_te"] == i).sum())
        rows.append((c, total, test))
        print(f"  {c:35s} {total:>9d} {test:>8d}")
    with open(os.path.join(OUT_DIR, "class_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "total", "test"]); w.writerows(rows)


# ---------- 1: macro-F1 under each condition ----------
@torch.no_grad()
def preds(model, X):
    model.eval(); xb = torch.from_numpy(X).float().to(DEVICE)
    out = [model(xb[i:i+EVAL_BATCH]).argmax(1) for i in range(0, len(xb), EVAL_BATCH)]
    return torch.cat(out).cpu().numpy()


def preds_attacked(model, X, y, spec, constrained):
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    out = []
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(model, xb, yb, spec, eps=EPS, alpha=ALPHA,
                            steps=STEPS, constrained=constrained)
        out.append(model(adv).argmax(1).cpu())
    return torch.cat(out).numpy()


def macro_f1_table():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    X_te, y_te = d["X_te"], d["y_te"]
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        spec_raw = json.load(f)
    spec = ConstraintSpec(spec_raw, DEVICE)
    n_classes = len(spec_raw["classes"]); in_dim = X_te.shape[1]

    def load(fn_exp, fn_out):
        p = os.path.join(EXP_DIR, fn_exp)
        if not os.path.exists(p): p = os.path.join(OUT_DIR, fn_out)
        m = IDSNet(in_dim, n_classes).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        return m

    victim   = load("victim_multiclass.pt", "victim.pt")
    defended = load("defended_multiclass.pt", "defended.pt")
    print("\n=== MULTICLASS MACRO-F1 ===")
    for name, model in [("undefended", victim), ("defended", defended)]:
        f_clean = f1_score(y_te, preds(model, X_te), average="macro", zero_division=0)
        f_unc = f1_score(y_te, preds_attacked(model, X_te, y_te, spec, False),
                         average="macro", zero_division=0)
        f_con = f1_score(y_te, preds_attacked(model, X_te, y_te, spec, True),
                         average="macro", zero_division=0)
        print(f"  {name:11s} clean={f_clean:.4f} unconstrained={f_unc:.4f} "
              f"constrained={f_con:.4f}")


# ---------- 2: four-variant ablation ----------
def four_way_ablation():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    X_te, y_te = d["X_te"], d["y_te"]
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        spec_raw = json.load(f)
    n_classes = len(spec_raw["classes"]); in_dim = X_te.shape[1]
    victim = IDSNet(in_dim, n_classes).to(DEVICE)
    p = os.path.join(EXP_DIR, "victim_multiclass.pt")
    if not os.path.exists(p): p = os.path.join(OUT_DIR, "victim.pt")
    victim.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))

    def acc_with(bounds, integ, immut):
        spec = ConstraintSpec(spec_raw, DEVICE)
        if not integ: spec.is_int = torch.zeros_like(spec.is_int)
        if not immut: spec.is_immutable = torch.zeros_like(spec.is_immutable)
        if not bounds:
            spec.fmin = torch.full_like(spec.fmin, -1e30)
            spec.fmax = torch.full_like(spec.fmax, 1e30)
        Xt = torch.from_numpy(X_te).float(); yt = torch.from_numpy(y_te).long()
        cor = tot = 0
        for i in range(0, len(Xt), EVAL_BATCH):
            xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
            adv, _ = pgd_attack(victim, xb, yb, spec, eps=EPS, alpha=ALPHA,
                                steps=STEPS, constrained=True)
            cor += (victim(adv).argmax(1) == yb).sum().item(); tot += yb.size(0)
        return cor / tot

    print("\n=== FOUR-VARIANT ABLATION (multiclass, undefended constrained acc) ===")
    print(f"  bounds only               : {acc_with(True, False, False):.4f}")
    print(f"  bounds + integrality      : {acc_with(True, True, False):.4f}")
    print(f"  bounds + immutability     : {acc_with(True, False, True):.4f}")
    print(f"  all constraints           : {acc_with(True, True, True):.4f}")


def main():
    print("Regenerating multiclass data...")
    regen_multiclass()
    dump_spec_and_dist()
    macro_f1_table()
    four_way_ablation()
    print("\n[done] See console output + outputs/feature_lists.json, "
          "class_distribution.csv")


if __name__ == "__main__":
    main()
