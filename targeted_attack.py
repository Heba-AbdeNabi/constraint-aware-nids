"""
targeted_attack.py
==================
Addresses Reviewer 2 (Major Concern 2): the threat model is evasion (malicious
-> benign), but the main evaluation uses an UNTARGETED attack. This script runs
a TARGETED constrained attack that specifically pushes malicious flows to the
BENIGN class, and reports the attack-to-benign SUCCESS RATE: the fraction of
malicious test flows classified as benign after attack.

Reports for undefended and defended (constrained-adv-trained) models, both
binary and multiclass. Fills the targeted-evasion subsection (sec:targeted).

Assumes BENIGN is class index 0 (true for our preprocessing; adjust BENIGN_IDX
if your label encoding differs -- check constraint_spec.json "classes").
"""
import os, json, sys, subprocess
import numpy as np
import torch
from model import IDSNet, DEVICE
from attack import ConstraintSpec

BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
EPS, ALPHA, STEPS = 0.3, 0.05, 20
EVAL_BATCH = 4096
MODES = ["binary", "multiclass"]


def regenerate_data(mode):
    cmd = [sys.executable, "preprocess.py", "--data_dir", "./data",
           "--out_dir", OUT_DIR, "--min_class", "10"]
    if mode == "binary":
        cmd.append("--binary")
    subprocess.run(cmd, check=True)


def targeted_pgd_to_benign(model, x, spec, benign_idx, eps, alpha, steps):
    """Constrained PGD that MINIMIZES loss toward the benign target class,
    i.e., maximizes benign logit. Projection keeps examples constraint-valid."""
    model.eval()
    x_orig = x.clone().detach()
    target = torch.full((x.size(0),), benign_idx, dtype=torch.long, device=x.device)
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = spec.project((x + delta).detach(), x_orig).detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, target)  # minimize -> toward benign
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - alpha * grad.sign()   # DESCEND toward target
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = spec.project(x_adv, x_orig).detach().requires_grad_(True)
    return x_adv.detach()


def benign_success_rate(model, X, y, spec, benign_idx):
    """Fraction of MALICIOUS flows classified benign after targeted attack."""
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    mal = yt != benign_idx
    Xt, yt = Xt[mal], yt[mal]
    succ = tot = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE)
        adv = targeted_pgd_to_benign(model, xb, spec, benign_idx, EPS, ALPHA, STEPS)
        pred = model(adv).argmax(1)
        succ += (pred == benign_idx).sum().item(); tot += xb.size(0)
    return succ / tot


def run_mode(mode, writer):
    regenerate_data(mode)
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    X_te, y_te = d["X_te"], d["y_te"]
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        spec_raw = json.load(f)
    spec = ConstraintSpec(spec_raw, DEVICE)
    classes = spec_raw["classes"]; n_classes = len(classes); in_dim = X_te.shape[1]
    benign_idx = classes.index("BENIGN") if "BENIGN" in classes else 0

    for tag, fname in [("undefended", f"victim_{mode}.pt"),
                       ("defended",  f"defended_{mode}.pt")]:
        path = os.path.join(EXP_DIR, fname)
        if not os.path.exists(path):
            path = os.path.join(OUT_DIR, "victim.pt" if tag == "undefended"
                                else "defended.pt")
        model = IDSNet(in_dim, n_classes).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        rate = benign_success_rate(model, X_te, y_te, spec, benign_idx)
        print(f"[{mode}/{tag}] attack-to-benign success rate = {rate:.4f}")
        writer.writerow([mode, tag, f"{rate:.4f}"])


def main():
    import csv
    out = os.path.join(EXP_DIR, "targeted_benign.csv")
    os.makedirs(EXP_DIR, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "model", "attack_to_benign_rate"])
        for m in MODES:
            run_mode(m, w)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
