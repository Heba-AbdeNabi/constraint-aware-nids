"""
run_adaptive_and_seeds.py
=========================
Two reviewer-requested experiments in one script:

  A. ADAPTIVE / STRONGER ATTACK (Reviewer 2, Major Concern 3)
     Evaluates the DEFENDED model against a much stronger CONSTRAINED attack:
     multiple random restarts + more PGD steps (the attacker keeps, per sample,
     the most successful restart). This tests whether the defense survives an
     attacker who tries harder, not just the same attack used in training.
     Reports defended constrained-attack accuracy under the adaptive attack,
     for MLP, both binary and multiclass.

  B. MULTI-SEED VARIANCE (Reviewer 3.5)
     Retrains, for N seeds, the MLP victim and BOTH defenses (constrained- and
     unconstrained-adversarial-training) at eps=0.3, and reports mean +/- std
     for the Table I cells and the baseline comparison. Only the cells where
     variance matters are recomputed (MLP, eps=0.3), to keep compute bounded.

Reuses preprocess.py / model.py / attack.py. Config at top.

OUTPUTS (./outputs/experiments/):
  adaptive_results.json          + printed summary
  seed_results.json              + printed summary (mean/std per cell)
  summary_adaptive_seeds.txt     (paste-ready)

Budget: multi-seed retraining dominates. 3 seeds x 2 modes x 3 trainings
(victim + 2 defenses) plus the adaptive eval. A few hours on one GPU.
"""
import os, sys, json, csv, time, subprocess
import numpy as np
import torch
import torch.nn as nn

from model import IDSNet, DEVICE, evaluate, make_loader
from attack import ConstraintSpec, pgd_attack

# ----------------------------------------------------------------------
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))

MODES        = ["binary", "multiclass"]
EPS          = 0.3
ALPHA        = 0.05
MIN_CLASS    = 10
EVAL_BATCH   = 4096

# --- experiment A: adaptive attack ---
ADAPTIVE_RESTARTS = 5
ADAPTIVE_STEPS    = 50

# --- experiment B: multi-seed ---
SEEDS             = [0, 1, 2]          # add more (e.g. [0,1,2,3,4]) if time allows
VICTIM_EPOCHS     = 30
DEF_EPOCHS        = 20
DEF_STEPS         = 7
ATTACK_STEPS_EVAL = 20                 # standard attack for the seed table
# ----------------------------------------------------------------------
os.makedirs(EXP_DIR, exist_ok=True)


# ============ shared helpers ============
def regen(mode):
    cmd = [sys.executable, os.path.join(SRC_DIR, "preprocess.py"),
           "--data_dir", DATA_DIR, "--out_dir", OUT_DIR, "--min_class", str(MIN_CLASS)]
    if mode == "binary":
        cmd.append("--binary")
    subprocess.run(cmd, check=True)


def load_split():
    d = np.load(os.path.join(OUT_DIR, "data.npz"))
    return d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"]


def load_spec():
    with open(os.path.join(OUT_DIR, "constraint_spec.json")) as f:
        return json.load(f)


def cw(y, k):
    c = np.bincount(y, minlength=k).astype(float)
    w = torch.tensor(c.sum() / (c + 1e-6), dtype=torch.float32)
    return (w / w.sum() * k).to(DEVICE)


@torch.no_grad()
def clean_acc(m, X, y):
    m.eval(); xb = torch.from_numpy(X).float().to(DEVICE); yb = torch.from_numpy(y).long().to(DEVICE)
    p = torch.cat([m(xb[i:i+EVAL_BATCH]).argmax(1) for i in range(0, len(xb), EVAL_BATCH)])
    return (p == yb).float().mean().item()


def atk_acc(m, X, y, spec, constrained, steps):
    m.eval(); Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    cor = tot = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(m, xb, yb, spec, eps=EPS, alpha=ALPHA, steps=steps, constrained=constrained)
        cor += (m(adv).argmax(1) == yb).sum().item(); tot += yb.size(0)
    return cor / tot


def train_clean(Xtr, ytr, Xv, yv, k, in_dim, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = IDSNet(in_dim, k).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=cw(ytr, k))
    Xt = torch.from_numpy(Xtr).float(); yt = torch.from_numpy(ytr).long()
    idx = np.arange(len(Xt)); best = 0; bs = None
    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]; xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        va, _, _ = evaluate(m, make_loader(Xv, yv, bs=EVAL_BATCH))
        if va > best: best = va; bs = {kk: v.cpu().clone() for kk, v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    return m


def train_adv(Xtr, ytr, Xv, yv, spec, k, in_dim, constrained, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = IDSNet(in_dim, k).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=cw(ytr, k))
    Xt = torch.from_numpy(Xtr).float(); yt = torch.from_numpy(ytr).long()
    idx = np.arange(len(Xt)); best = 0; bs = None
    for _ in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]; xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            m.eval()
            adv, _ = pgd_attack(m, xb, yb, spec, eps=EPS, alpha=ALPHA, steps=DEF_STEPS, constrained=constrained)
            m.train()
            xm = torch.cat([xb, adv]); ym = torch.cat([yb, yb])
            opt.zero_grad(); crit(m(xm), ym).backward(); opt.step()
        va, _, _ = evaluate(m, make_loader(Xv, yv, bs=EVAL_BATCH))
        if va > best: best = va; bs = {kk: v.cpu().clone() for kk, v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    return m


# ============ Experiment A: adaptive constrained attack ============
def adaptive_constrained_acc(model, X, y, spec, restarts, steps):
    """Constrained PGD with multiple random restarts. A sample counts as
    correctly classified only if it resists ALL restarts (i.e., the attacker
    keeps the best/most-successful restart per sample)."""
    model.eval()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    correct = tot = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        # start: assume all still correctly classified; attacker tries to flip
        still_correct = torch.ones_like(yb, dtype=torch.bool)
        for r in range(restarts):
            adv, _ = pgd_attack(model, xb, yb, spec, eps=EPS, alpha=ALPHA,
                                steps=steps, constrained=True)
            pred = model(adv).argmax(1)
            # a restart that flips the sample means the attacker succeeded there
            flipped = pred != yb
            still_correct = still_correct & (~flipped)
        correct += still_correct.sum().item(); tot += yb.size(0)
    return correct / tot


def run_adaptive():
    print("\n" + "="*60 + "\nEXPERIMENT A: ADAPTIVE ATTACK (defended model)\n" + "="*60)
    res = {}
    for mode in MODES:
        regen(mode)
        Xtr, ytr, Xv, yv, Xte, yte = load_split()
        spec_raw = load_spec(); spec = ConstraintSpec(spec_raw, DEVICE)
        k = len(spec_raw["classes"]); in_dim = Xtr.shape[1]
        # load defended model if cached, else train it (constrained adv-training)
        dpath = os.path.join(EXP_DIR, f"defended_{mode}.pt")
        m = IDSNet(in_dim, k).to(DEVICE)
        if os.path.exists(dpath):
            m.load_state_dict(torch.load(dpath, map_location=DEVICE, weights_only=True))
            print(f"[{mode}] loaded cached defended model")
        else:
            print(f"[{mode}] training defended model (constrained adv-training)...")
            m = train_adv(Xtr, ytr, Xv, yv, spec, k, in_dim, True, DEF_EPOCHS, seed=0)
            torch.save(m.state_dict(), dpath)
        std = atk_acc(m, Xte, yte, spec, True, ATTACK_STEPS_EVAL)
        adv = adaptive_constrained_acc(m, Xte, yte, spec, ADAPTIVE_RESTARTS, ADAPTIVE_STEPS)
        res[mode] = {"defended_standard_constrained": round(std, 4),
                     "defended_adaptive_constrained": round(adv, 4),
                     "restarts": ADAPTIVE_RESTARTS, "steps": ADAPTIVE_STEPS}
        print(f"[{mode}] defended: standard-attack acc={std:.4f} | "
              f"ADAPTIVE ({ADAPTIVE_RESTARTS}x{ADAPTIVE_STEPS}) acc={adv:.4f}")
    with open(os.path.join(EXP_DIR, "adaptive_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


# ============ Experiment B: multi-seed variance ============
def run_seeds():
    print("\n" + "="*60 + f"\nEXPERIMENT B: MULTI-SEED VARIANCE ({len(SEEDS)} seeds)\n" + "="*60)
    # accumulate per-mode lists of each metric across seeds
    acc = {mode: {kk: [] for kk in
                  ["undef_clean", "undef_unc", "undef_con",
                   "def_clean", "def_unc", "def_con",
                   "uncadv_clean", "uncadv_con"]} for mode in MODES}
    for mode in MODES:
        regen(mode)
        Xtr, ytr, Xv, yv, Xte, yte = load_split()
        spec_raw = load_spec(); spec = ConstraintSpec(spec_raw, DEVICE)
        k = len(spec_raw["classes"]); in_dim = Xtr.shape[1]
        for s in SEEDS:
            print(f"\n[{mode}] seed {s} ...")
            victim = train_clean(Xtr, ytr, Xv, yv, k, in_dim, VICTIM_EPOCHS, s)
            defcon = train_adv(Xtr, ytr, Xv, yv, spec, k, in_dim, True, DEF_EPOCHS, s)
            defunc = train_adv(Xtr, ytr, Xv, yv, spec, k, in_dim, False, DEF_EPOCHS, s)
            a = acc[mode]
            a["undef_clean"].append(clean_acc(victim, Xte, yte))
            a["undef_unc"].append(atk_acc(victim, Xte, yte, spec, False, ATTACK_STEPS_EVAL))
            a["undef_con"].append(atk_acc(victim, Xte, yte, spec, True, ATTACK_STEPS_EVAL))
            a["def_clean"].append(clean_acc(defcon, Xte, yte))
            a["def_unc"].append(atk_acc(defcon, Xte, yte, spec, False, ATTACK_STEPS_EVAL))
            a["def_con"].append(atk_acc(defcon, Xte, yte, spec, True, ATTACK_STEPS_EVAL))
            a["uncadv_clean"].append(clean_acc(defunc, Xte, yte))
            a["uncadv_con"].append(atk_acc(defunc, Xte, yte, spec, True, ATTACK_STEPS_EVAL))
    # summarize
    summary = {}
    for mode in MODES:
        summary[mode] = {kk: {"mean": round(float(np.mean(v)), 4),
                              "std": round(float(np.std(v)), 4),
                              "runs": v}
                         for kk, v in acc[mode].items()}
    with open(os.path.join(EXP_DIR, "seed_results.json"), "w") as f:
        json.dump({"seeds": SEEDS, "summary": summary}, f, indent=2)
    return summary


# ============ output ============
def write_summary(adaptive, seeds):
    L = []
    def p(s=""): L.append(s)
    p("="*70); p("ADAPTIVE ATTACK + MULTI-SEED SUMMARY (paste into paper)"); p("="*70)

    p("\n--- ADAPTIVE ATTACK (defended model, constrained) ---")
    for mode in MODES:
        r = adaptive[mode]
        p(f"{mode:11s} standard={r['defended_standard_constrained']:.4f}  "
          f"adaptive({r['restarts']}x{r['steps']})={r['defended_adaptive_constrained']:.4f}")

    p("\n--- MULTI-SEED (mean +/- std over "
      f"{len(SEEDS)} seeds), eps=0.3, MLP ---")
    label = {"undef_clean": "undef clean", "undef_unc": "undef unconstrained",
             "undef_con": "undef constrained", "def_clean": "def clean",
             "def_unc": "def unconstrained", "def_con": "def constrained",
             "uncadv_clean": "unc-advtrain clean", "uncadv_con": "unc-advtrain constrained"}
    for mode in MODES:
        p(f"\n[{mode}]")
        for kk in ["undef_clean", "undef_unc", "undef_con", "def_clean", "def_unc",
                   "def_con", "uncadv_clean", "uncadv_con"]:
            s = seeds[mode][kk]
            p(f"  {label[kk]:26s} {s['mean']:.4f} +/- {s['std']:.4f}")
    out = os.path.join(EXP_DIR, "summary_adaptive_seeds.txt")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[saved] {out}")


def main():
    t0 = time.time()
    adaptive = run_adaptive()
    seeds = run_seeds()
    write_summary(adaptive, seeds)
    print(f"\n[ALL DONE] {(time.time()-t0)/60:.1f} min. Files in {EXP_DIR}")


if __name__ == "__main__":
    main()
