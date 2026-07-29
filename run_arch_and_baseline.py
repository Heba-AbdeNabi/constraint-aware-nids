
import os, csv, json, sys, subprocess
import numpy as np
import torch
import torch.nn as nn

from model import IDSNet, DEVICE, evaluate, make_loader
from model_cnn import IDSNetCNN
from attack import ConstraintSpec, pgd_attack

# ----------------------------------------------------------------------
BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))

MODES   = ["binary", "multiclass"]
ARCHES  = ["mlp", "cnn"]            # trim to ["mlp"] or ["cnn"] to go faster
EPS, ALPHA, ATTACK_STEPS = 0.3, 0.05, 20
VICTIM_EPOCHS = 30
DEF_EPOCHS, DEF_STEPS = 20, 7
EVAL_BATCH = 4096
# ----------------------------------------------------------------------
os.makedirs(EXP_DIR, exist_ok=True)


def build(arch, in_dim, n_classes):
    if arch == "mlp":
        return IDSNet(in_dim, n_classes).to(DEVICE)
    if arch == "cnn":
        return IDSNetCNN(in_dim, n_classes).to(DEVICE)
    raise ValueError(arch)


def regenerate_data(mode):
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


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)
    return (w / w.sum() * n_classes).to(DEVICE)


@torch.no_grad()
def clean_acc(model, X, y):
    model.eval()
    xb = torch.from_numpy(X).float().to(DEVICE)
    yb = torch.from_numpy(y).long().to(DEVICE)
    preds = []
    for i in range(0, len(xb), EVAL_BATCH):
        preds.append(model(xb[i:i+EVAL_BATCH]).argmax(1))
    return (torch.cat(preds) == yb).float().mean().item()


def atk_acc(model, X, y, spec, constrained):
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    correct = total = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(model, xb, yb, spec, eps=EPS, alpha=ALPHA,
                            steps=ATTACK_STEPS, constrained=constrained)
        correct += (model(adv).argmax(1) == yb).sum().item(); total += yb.size(0)
    return correct / total


def train_clean(arch, X_tr, y_tr, X_val, y_val, n_classes, in_dim):
    model = build(arch, in_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_tr, n_classes))
    Xt = torch.from_numpy(X_tr).float(); yt = torch.from_numpy(y_tr).long()
    idx = np.arange(len(Xt)); best = 0.0; best_state = None
    for ep in range(1, VICTIM_EPOCHS + 1):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]
            xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        va, _, _ = evaluate(model, make_loader(X_val, y_val, bs=EVAL_BATCH))
        if va > best: best = va; best_state = {k: v.cpu().clone()
                                               for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model


def train_adv(arch, X_tr, y_tr, X_val, y_val, spec, n_classes, in_dim, constrained):
    model = build(arch, in_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_tr, n_classes))
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


def run():
    arch_rows, base_rows = [], []
    for mode in MODES:
        print(f"\n########## MODE = {mode} ##########")
        regenerate_data(mode)
        X_tr, y_tr, X_val, y_val, X_te, y_te = load_split()
        spec_raw = load_spec(); spec = ConstraintSpec(spec_raw, DEVICE)
        n_classes = len(spec_raw["classes"]); in_dim = X_tr.shape[1]

        for arch in ARCHES:
            print(f"\n--- arch = {arch} ---")
            # undefended victim
            victim = train_clean(arch, X_tr, y_tr, X_val, y_val, n_classes, in_dim)
            v_clean = clean_acc(victim, X_te, y_te)
            v_unc   = atk_acc(victim, X_te, y_te, spec, constrained=False)
            v_con   = atk_acc(victim, X_te, y_te, spec, constrained=True)
            print(f"[victim]  clean={v_clean:.4f} unc={v_unc:.4f} con={v_con:.4f}")

            # constrained adversarial training (our defense)
            defended = train_adv(arch, X_tr, y_tr, X_val, y_val, spec,
                                 n_classes, in_dim, constrained=True)
            d_clean = clean_acc(defended, X_te, y_te)
            d_unc   = atk_acc(defended, X_te, y_te, spec, constrained=False)
            d_con   = atk_acc(defended, X_te, y_te, spec, constrained=True)
            print(f"[def-con] clean={d_clean:.4f} unc={d_unc:.4f} con={d_con:.4f}")

            # EXPERIMENT 1 rows (architecture robustness)
            arch_rows.append([mode, arch, "undefended", v_clean, v_unc, v_con])
            arch_rows.append([mode, arch, "defended",   d_clean, d_unc, d_con])

            # unconstrained adversarial training (baseline defense)
            unc_def = train_adv(arch, X_tr, y_tr, X_val, y_val, spec,
                                n_classes, in_dim, constrained=False)
            u_clean = clean_acc(unc_def, X_te, y_te)
            u_con   = atk_acc(unc_def, X_te, y_te, spec, constrained=True)
            print(f"[def-unc] clean={u_clean:.4f} con={u_con:.4f}")

            # EXPERIMENT 2 rows (training-type baseline) on constrained attack
            base_rows.append([mode, arch, "undefended",            v_clean, v_con])
            base_rows.append([mode, arch, "unconstrained_advtrain", u_clean, u_con])
            base_rows.append([mode, arch, "constrained_advtrain",   d_clean, d_con])

    with open(os.path.join(EXP_DIR, "arch_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "arch", "model", "clean_acc",
                    "unconstrained_atk_acc", "constrained_atk_acc"])
        w.writerows(arch_rows)
    with open(os.path.join(EXP_DIR, "baseline_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "arch", "defense", "clean_acc", "constrained_atk_acc"])
        w.writerows(base_rows)

    print(f"\n[done] wrote arch_compare.csv and baseline_compare.csv to\n  {EXP_DIR}")


if __name__ == "__main__":
    run()
