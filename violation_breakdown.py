"""
violation_breakdown.py
======================
Addresses Reviewer 1 (comment 3) and Reviewer 3: the 0.0% validity result is
striking but should say WHICH constraint types the unconstrained adversarial
examples violate, and how often.

For the undefended victim, it crafts UNCONSTRAINED PGD adversarial examples on
the test set and, for each, checks which of the three constraint types it
violates (bounds / integrality / immutability), then reports the fraction
violating each type and the fraction violating multiple simultaneously.

Fills Table "tab:violations" in the paper.

Run after the victim is trained. Paths default to ./outputs (edit BASE_DIR).
"""
import os, json
import numpy as np
import torch
from model import IDSNet, DEVICE
from attack import ConstraintSpec, pgd_attack

BASE_DIR = "."
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EPS, ALPHA, STEPS = 0.3, 0.05, 20
EVAL_BATCH = 4096


def _to_raw(x_std, mean, scale):
    return x_std * scale + mean


def main():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    X_te, y_te = d["X_te"], d["y_te"]
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        spec_raw = json.load(f)
    spec = ConstraintSpec(spec_raw, DEVICE)
    n_classes = len(spec_raw["classes"]); in_dim = X_te.shape[1]

    victim = IDSNet(in_dim, n_classes).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(OUT_DIR, "victim.pt"),
                                      map_location=DEVICE, weights_only=True))
    victim.eval()

    mean, scale = spec.mean, spec.scale
    fmin, fmax = spec.fmin, spec.fmax
    is_int, is_imm = spec.is_int, spec.is_immutable
    tol = 1e-4

    n = 0
    v_bounds = v_int = v_imm = v_multi = v_any = 0
    Xt = torch.from_numpy(X_te).float(); yt = torch.from_numpy(y_te).long()
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(victim, xb, yb, spec, eps=EPS, alpha=ALPHA,
                            steps=STEPS, constrained=False)
        xr = _to_raw(adv, mean, scale)
        x0 = _to_raw(xb, mean, scale)
        # per-sample violation flags
        b = ((xr < fmin - tol) | (xr > fmax + tol)).any(1)
        it = (is_int & (torch.abs(xr - torch.round(xr)) > tol)).any(1)
        im = (is_imm & (torch.abs(xr - x0) > tol)).any(1)
        cnt = b.int() + it.int() + im.int()
        multi = cnt >= 2
        anyv = cnt >= 1
        bs = yb.size(0); n += bs
        v_bounds += b.sum().item(); v_int += it.sum().item()
        v_imm += im.sum().item(); v_multi += multi.sum().item()
        v_any += anyv.sum().item()

    def pct(x): return 100.0 * x / n
    print(f"n = {n} unconstrained adversarial examples")
    print(f"  violate bounds        : {pct(v_bounds):.1f}%")
    print(f"  violate integrality   : {pct(v_int):.1f}%")
    print(f"  violate immutability  : {pct(v_imm):.1f}%")
    print(f"  violate >= 2 types    : {pct(v_multi):.1f}%")
    print(f"  violate any (invalid) : {pct(v_any):.1f}%")
    print("\nPaste these into Table tab:violations.")
    with open(os.path.join(OUT_DIR, "violation_breakdown.json"), "w") as f:
        json.dump({"n": n, "bounds": pct(v_bounds), "integrality": pct(v_int),
                   "immutability": pct(v_imm), "multiple": pct(v_multi),
                   "any": pct(v_any)}, f, indent=2)


if __name__ == "__main__":
    main()
