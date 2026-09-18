"""
run_experiments.py
==================
Full experimental + ablation study for the paper, for BOTH binary and
multiclass settings. Produces every number needed for the figures/tables and
saves them as JSON + CSV. Also saves a small sample of clean vs adversarial
flows so you can make feature-perturbation plots.

WHAT IT RUNS (per label mode: binary, multiclass):
  A. Clean baseline (train victim if not already trained).
  B. Attack-strength sweep: for several eps values, measure undefended
     accuracy under unconstrained + constrained attack, plus validity rate.
  C. Immutable-masking ablation: constrained attack WITH vs WITHOUT the
     immutable-feature mask, to show the mask matters.
  D. Defense: adversarial train on constrained PGD, then re-measure the
     full attack sweep on the defended model.
  E. Per-class breakdown (multiclass only): constrained-attack accuracy per
     attack class, undefended vs defended.

OUTPUTS (in --out_dir/experiments/):
  results_binary.json, results_multiclass.json   (everything, structured)
  sweep_binary.csv, sweep_multiclass.csv         (eps sweep -> easy plotting)
  ablation_binary.csv, ablation_multiclass.csv
  perclass_multiclass.csv
  summary_table.csv                              (the headline table)
  adv_sample_<mode>.npz                          (clean/adv flow samples)

HOW TO RUN:
  Edit CONFIG below (paths + which modes/eps), then just Run in PyCharm.
  It is resumable: victims/defended models are cached and reused.

NOTE: This regenerates data.npz for each mode by importing preprocess. If you
prefer to keep your existing binary data.npz, set REUSE_EXISTING_DATA=True and
run one mode at a time.
"""
import os, json, csv, time, argparse, subprocess, sys
import numpy as np
import torch

# import your existing modules (must be in the same folder)
from model import IDSNet, DEVICE, evaluate, make_loader, train as train_victim, prf
from attack import ConstraintSpec, pgd_attack

# ----------------------------------------------------------------------
# CONFIG - edit these
# ----------------------------------------------------------------------
BASE_DIR   = "."
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))   # where preprocess.py lives
EXP_DIR    = os.path.join(OUT_DIR, "experiments")

MODES        = ["binary", "multiclass"]   # run both; trim to one if short on time
EPS_SWEEP    = [0.05, 0.1, 0.2, 0.3, 0.5] # attack-strength sweep
ATTACK_STEPS = 20
ALPHA        = 0.05
VICTIM_EPOCHS   = 30
DEFENSE_EPOCHS  = 20
DEFENSE_STEPS   = 7      # cheaper PGD during adversarial training
MIN_CLASS       = 10     # for multiclass rare-class dropping
ADV_SAMPLE_N    = 2000   # how many clean/adv flows to save for perturbation plots
EVAL_BATCH      = 4096
# ----------------------------------------------------------------------

os.makedirs(EXP_DIR, exist_ok=True)


# ---------- data handling ----------
def regenerate_data(mode):
    """Run preprocess.py for the given mode so data.npz matches it."""
    cmd = [sys.executable, os.path.join(SRC_DIR, "preprocess.py"),
           "--data_dir", DATA_DIR, "--out_dir", OUT_DIR,
           "--min_class", str(MIN_CLASS)]
    if mode == "binary":
        cmd.append("--binary")
    # NOTE: preprocess must accept --binary as a real store_true (not forced).
    print(f"[data] regenerating data.npz for mode={mode} ...")
    subprocess.run(cmd, check=True)


def load_split():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    return (d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"])


def load_spec():
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        return json.load(f)


# ---------- metrics ----------
@torch.no_grad()
def clean_metrics(model, X, y, n_classes):
    loader = make_loader(X, y, bs=EVAL_BATCH)
    acc, yp, yt = evaluate(model, loader)
    p, r, f = prf(yt, yp, n_classes)
    return {"acc": acc, "macro_p": p, "macro_r": r, "macro_f1": f}


def attack_accuracy(model, X, y, spec, constrained, eps, use_immutable=True):
    """Accuracy under attack + validity rate. If use_immutable is False, the
    immutable mask is temporarily disabled (ablation)."""
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    # optionally disable immutable masking for the ablation
    saved = spec.is_immutable
    if not use_immutable:
        spec.is_immutable = torch.zeros_like(spec.is_immutable)
    correct = total = 0; vcount = vtot = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, valid = pgd_attack(model, xb, yb, spec, eps=eps, alpha=ALPHA,
                                steps=ATTACK_STEPS, constrained=constrained)
        pred = model(adv).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
        vcount += valid.sum().item(); vtot += valid.numel()
    spec.is_immutable = saved  # restore
    return correct / total, vcount / vtot


def perclass_constrained_acc(model, X, y, spec, classes, eps):
    """Constrained-attack accuracy per class (multiclass diagnostic)."""
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    per_correct = np.zeros(len(classes)); per_total = np.zeros(len(classes))
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(model, xb, yb, spec, eps=eps, alpha=ALPHA,
                            steps=ATTACK_STEPS, constrained=True)
        pred = model(adv).argmax(1)
        for c in range(len(classes)):
            m = (yb == c)
            per_total[c] += m.sum().item()
            per_correct[c] += (pred[m] == c).sum().item()
    return {classes[c]: (per_correct[c] / per_total[c] if per_total[c] else float("nan"))
            for c in range(len(classes))}


# ---------- models ----------
def get_victim(X_tr, y_tr, X_val, y_val, n_classes, in_dim, tag):
    path = os.path.join(EXP_DIR, f"victim_{tag}.pt")
    model = IDSNet(in_dim, n_classes).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        print(f"[victim:{tag}] loaded cached model")
        return model
    print(f"[victim:{tag}] training ...")
    tr = make_loader(X_tr, y_tr, bs=512, shuffle=True)
    val = make_loader(X_val, y_val, bs=EVAL_BATCH)
    model = train_victim(model, tr, val, n_classes, epochs=VICTIM_EPOCHS)
    torch.save(model.state_dict(), path)
    return model


def get_defended(X_tr, y_tr, X_val, y_val, spec, n_classes, in_dim, tag, eps=0.3):
    path = os.path.join(EXP_DIR, f"defended_{tag}.pt")
    model = IDSNet(in_dim, n_classes).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        print(f"[defended:{tag}] loaded cached model")
        return model
    print(f"[defended:{tag}] adversarial training ...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)
    w = (w / w.sum() * n_classes).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    Xt = torch.from_numpy(X_tr).float(); yt = torch.from_numpy(y_tr).long()
    idx = np.arange(len(Xt)); best = 0.0; best_state = None
    for ep in range(1, DEFENSE_EPOCHS + 1):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]
            xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            model.eval()
            adv, _ = pgd_attack(model, xb, yb, spec, eps=eps, alpha=ALPHA,
                                steps=DEFENSE_STEPS, constrained=True)
            model.train()
            xm = torch.cat([xb, adv], 0); ym = torch.cat([yb, yb], 0)
            opt.zero_grad(); loss = crit(model(xm), ym); loss.backward(); opt.step()
        va, _, _ = evaluate(model, make_loader(X_val, y_val, bs=EVAL_BATCH))
        if va > best: best = va; best_state = {k: v.cpu().clone()
                                               for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  [adv-train:{tag}] epoch {ep:3d} | val {va:.4f} (best {best:.4f})")
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), path)
    return model


def save_adv_sample(model, X, y, spec, tag, eps=0.3):
    """Save a sample of clean vs constrained-adv flows for perturbation figures."""
    n = min(ADV_SAMPLE_N, len(X))
    xb = torch.from_numpy(X[:n]).float().to(DEVICE)
    yb = torch.from_numpy(y[:n]).long().to(DEVICE)
    adv, valid = pgd_attack(model, xb, yb, spec, eps=eps, alpha=ALPHA,
                            steps=ATTACK_STEPS, constrained=True)
    np.savez_compressed(os.path.join(EXP_DIR, f"adv_sample_{tag}.npz"),
                        X_clean=X[:n], X_adv=adv.cpu().numpy(),
                        y=y[:n], valid=valid.cpu().numpy())


# ---------- per-mode driver ----------
def run_mode(mode, reuse_data=False):
    tag = mode
    print(f"\n{'='*60}\nMODE = {mode}\n{'='*60}")
    if not reuse_data:
        regenerate_data(mode)
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_split()
    spec_raw = load_spec()
    classes = spec_raw["classes"]; n_classes = len(classes); in_dim = X_tr.shape[1]
    spec = ConstraintSpec(spec_raw, DEVICE)
    print(f"[info] classes={n_classes} in_dim={in_dim} "
          f"train={X_tr.shape} test={X_te.shape}")

    results = {"mode": mode, "n_classes": n_classes, "classes": classes}

    # --- A. clean baseline (undefended) ---
    victim = get_victim(X_tr, y_tr, X_val, y_val, n_classes, in_dim, tag)
    results["undef_clean"] = clean_metrics(victim, X_te, y_te, n_classes)
    print(f"[A] undef clean acc={results['undef_clean']['acc']:.4f} "
          f"F1={results['undef_clean']['macro_f1']:.4f}")

    # --- B. attack-strength sweep (undefended) ---
    sweep = []
    for eps in EPS_SWEEP:
        unc_acc, unc_val = attack_accuracy(victim, X_te, y_te, spec, False, eps)
        con_acc, _       = attack_accuracy(victim, X_te, y_te, spec, True, eps)
        row = {"eps": eps, "unconstrained_acc": unc_acc,
               "constrained_acc": con_acc, "unconstrained_validity": unc_val}
        sweep.append(row)
        print(f"[B] eps={eps:<4} unconstrained={unc_acc:.4f} "
              f"(valid {unc_val*100:.1f}%) constrained={con_acc:.4f}")
    results["undef_sweep"] = sweep

    # --- C. immutable-masking ablation (undefended, at eps=0.3) ---
    abl = []
    for use_imm in [True, False]:
        acc, val = attack_accuracy(victim, X_te, y_te, spec, True, 0.3, use_immutable=use_imm)
        abl.append({"immutable_mask": use_imm, "constrained_acc": acc,
                    "validity": val})
        print(f"[C] immutable_mask={use_imm} constrained_acc={acc:.4f}")
    results["ablation_immutable"] = abl

    # --- D. defense + re-sweep ---
    defended = get_defended(X_tr, y_tr, X_val, y_val, spec, n_classes, in_dim, tag)
    results["def_clean"] = clean_metrics(defended, X_te, y_te, n_classes)
    dsweep = []
    for eps in EPS_SWEEP:
        unc_acc, unc_val = attack_accuracy(defended, X_te, y_te, spec, False, eps)
        con_acc, _       = attack_accuracy(defended, X_te, y_te, spec, True, eps)
        dsweep.append({"eps": eps, "unconstrained_acc": unc_acc,
                       "constrained_acc": con_acc, "unconstrained_validity": unc_val})
        print(f"[D] eps={eps:<4} DEF unconstrained={unc_acc:.4f} "
              f"constrained={con_acc:.4f}")
    results["def_sweep"] = dsweep
    print(f"[D] def clean acc={results['def_clean']['acc']:.4f}")

    # --- E. per-class breakdown (multiclass only) ---
    if mode == "multiclass":
        results["perclass_undef"] = perclass_constrained_acc(
            victim, X_te, y_te, spec, classes, 0.3)
        results["perclass_def"] = perclass_constrained_acc(
            defended, X_te, y_te, spec, classes, 0.3)

    # --- save sample flows for perturbation plots ---
    save_adv_sample(victim, X_te, y_te, spec, tag)

    # --- write JSON + CSVs ---
    with open(os.path.join(EXP_DIR, f"results_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(EXP_DIR, f"sweep_{mode}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "model", "unconstrained_acc", "constrained_acc",
                    "unconstrained_validity"])
        for r in sweep:
            w.writerow([r["eps"], "undefended", r["unconstrained_acc"],
                        r["constrained_acc"], r["unconstrained_validity"]])
        for r in dsweep:
            w.writerow([r["eps"], "defended", r["unconstrained_acc"],
                        r["constrained_acc"], r["unconstrained_validity"]])

    with open(os.path.join(EXP_DIR, f"ablation_{mode}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["immutable_mask", "constrained_acc", "validity"])
        for r in abl:
            w.writerow([r["immutable_mask"], r["constrained_acc"], r["validity"]])

    if mode == "multiclass":
        with open(os.path.join(EXP_DIR, "perclass_multiclass.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["class", "undef_constrained_acc",
                                           "def_constrained_acc"])
            for c in classes:
                w.writerow([c, results["perclass_undef"][c], results["perclass_def"][c]])

    print(f"[save] results_{mode}.json + CSVs written to {EXP_DIR}")
    return results


def write_summary(all_results):
    """Headline table: clean / eps=0.3 attacks, undefended vs defended, per mode."""
    path = os.path.join(EXP_DIR, "summary_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "setting", "undefended", "defended"])
        for res in all_results:
            m = res["mode"]
            # pull eps=0.3 rows
            u = next(r for r in res["undef_sweep"] if abs(r["eps"]-0.3) < 1e-9)
            d = next(r for r in res["def_sweep"] if abs(r["eps"]-0.3) < 1e-9)
            w.writerow([m, "clean", res["undef_clean"]["acc"], res["def_clean"]["acc"]])
            w.writerow([m, "unconstrained_atk@0.3", u["unconstrained_acc"], d["unconstrained_acc"]])
            w.writerow([m, "constrained_atk@0.3", u["constrained_acc"], d["constrained_acc"]])
            w.writerow([m, "unconstrained_validity@0.3", u["unconstrained_validity"], ""])
    print(f"[save] summary_table.csv written")


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--reuse_data", action="store_true",
                    help="use existing data.npz instead of regenerating (single mode)")
    args = ap.parse_args()

    all_res = []
    for mode in args.modes:
        all_res.append(run_mode(mode, reuse_data=args.reuse_data))
    write_summary(all_res)
    print(f"\n[done] all experiments in {(time.time()-t0)/60:.1f} min. "
          f"Everything saved under:\n  {EXP_DIR}")


if __name__ == "__main__":
    main()
