"""
run_all.py
==========
ONE script that runs the entire experimental pipeline for the paper and writes
every number and figure, for both binary and multiclass, after the leaky-feature
fix. Run this once; it regenerates data, trains models, runs every attack/defense
experiment, and dumps a single results bundle you can paste into the paper.

Produces (under ./outputs/experiments/):
  - all_results.json         : every number, structured, for the whole paper
  - summary_for_paper.txt    : human-readable, table-by-table, ready to paste
  - sweep_binary.csv, sweep_multiclass.csv
  - arch_compare.csv, baseline_compare.csv
  - perclass_multiclass.csv
  - class_distribution.csv, feature_lists.json
  - violation_breakdown.json
  - targeted_benign.csv
  - figures/*.pdf + table .tex snippets (via make_figures)

Requires in the same folder: preprocess.py (PATCHED), model.py, model_cnn.py,
attack.py, make_figures.py.

Config at top. Budget: a few hours on one GPU (trains many models across two
label settings and five eps values). Trim EPS_SWEEP / ARCHES to go faster.
"""
import os, sys, json, csv, subprocess, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from model import IDSNet, DEVICE, evaluate, make_loader
from model_cnn import IDSNetCNN
from attack import ConstraintSpec, pgd_attack

# ----------------------------------------------------------------------
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
EXP_DIR  = os.path.join(OUT_DIR, "experiments")
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))

MODES        = ["binary", "multiclass"]
ARCHES       = ["mlp", "cnn"]
EPS_MAIN     = 0.3
EPS_SWEEP    = [0.05, 0.1, 0.2, 0.3, 0.5]
ALPHA, STEPS = 0.05, 20
VICTIM_EPOCHS, DEF_EPOCHS, DEF_STEPS = 30, 20, 7
MIN_CLASS    = 10
EVAL_BATCH   = 4096
SEED         = 42
# ----------------------------------------------------------------------
os.makedirs(EXP_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
R = {"config": {"eps_main": EPS_MAIN, "eps_sweep": EPS_SWEEP, "seed": SEED}}


# ============ helpers ============
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


def build(arch, in_dim, k):
    return (IDSNetCNN if arch == "cnn" else IDSNet)(in_dim, k).to(DEVICE)


def cw(y, k):
    c = np.bincount(y, minlength=k).astype(float)
    w = torch.tensor(c.sum() / (c + 1e-6), dtype=torch.float32)
    return (w / w.sum() * k).to(DEVICE)


def train_clean(arch, Xtr, ytr, Xv, yv, k, in_dim, epochs):
    m = build(arch, in_dim, k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=cw(ytr, k))
    Xt = torch.from_numpy(Xtr).float(); yt = torch.from_numpy(ytr).long()
    idx = np.arange(len(Xt)); best = 0; bs_state = None
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]; xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        va, _, _ = evaluate(m, make_loader(Xv, yv, bs=EVAL_BATCH))
        if va > best: best = va; bs_state = {k_: v.cpu().clone() for k_, v in m.state_dict().items()}
    if bs_state: m.load_state_dict(bs_state)
    return m


def train_adv(arch, Xtr, ytr, Xv, yv, spec, k, in_dim, constrained, epochs):
    m = build(arch, in_dim, k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=cw(ytr, k))
    Xt = torch.from_numpy(Xtr).float(); yt = torch.from_numpy(ytr).long()
    idx = np.arange(len(Xt)); best = 0; bs_state = None
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i+512]; xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            m.eval()
            adv, _ = pgd_attack(m, xb, yb, spec, eps=EPS_MAIN, alpha=ALPHA,
                                steps=DEF_STEPS, constrained=constrained)
            m.train()
            xm = torch.cat([xb, adv]); ym = torch.cat([yb, yb])
            opt.zero_grad(); crit(m(xm), ym).backward(); opt.step()
        va, _, _ = evaluate(m, make_loader(Xv, yv, bs=EVAL_BATCH))
        if va > best: best = va; bs_state = {k_: v.cpu().clone() for k_, v in m.state_dict().items()}
    if bs_state: m.load_state_dict(bs_state)
    return m


@torch.no_grad()
def clean_acc(m, X, y):
    m.eval(); xb = torch.from_numpy(X).float().to(DEVICE); yb = torch.from_numpy(y).long().to(DEVICE)
    p = torch.cat([m(xb[i:i+EVAL_BATCH]).argmax(1) for i in range(0, len(xb), EVAL_BATCH)])
    return (p == yb).float().mean().item()


def atk_acc(m, X, y, spec, constrained, eps):
    m.eval(); Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    cor = tot = vc = vt = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, valid = pgd_attack(m, xb, yb, spec, eps=eps, alpha=ALPHA, steps=STEPS, constrained=constrained)
        cor += (m(adv).argmax(1) == yb).sum().item(); tot += yb.size(0)
        vc += valid.sum().item(); vt += valid.numel()
    return cor / tot, vc / vt


def preds_attacked(m, X, y, spec, constrained):
    m.eval(); Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long(); out = []
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(m, xb, yb, spec, eps=EPS_MAIN, alpha=ALPHA, steps=STEPS, constrained=constrained)
        out.append(m(adv).argmax(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def preds_clean(m, X):
    m.eval(); xb = torch.from_numpy(X).float().to(DEVICE)
    return torch.cat([m(xb[i:i+EVAL_BATCH]).argmax(1) for i in range(0, len(xb), EVAL_BATCH)]).cpu().numpy()


def targeted_benign(m, X, y, spec, benign_idx):
    m.eval(); Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    mal = yt != benign_idx; Xt = Xt[mal]
    succ = tot = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); x0 = xb.clone()
        tgt = torch.full((xb.size(0),), benign_idx, dtype=torch.long, device=DEVICE)
        delta = torch.empty_like(xb).uniform_(-EPS_MAIN, EPS_MAIN)
        xa = spec.project((xb + delta).detach(), x0).detach().requires_grad_(True)
        for _ in range(STEPS):
            loss = nn.functional.cross_entropy(m(xa), tgt)
            g = torch.autograd.grad(loss, xa)[0]
            xa = xa.detach() - ALPHA * g.sign()
            xa = torch.max(torch.min(xa, x0 + EPS_MAIN), x0 - EPS_MAIN)
            xa = spec.project(xa, x0).detach().requires_grad_(True)
        succ += (m(xa.detach()).argmax(1) == benign_idx).sum().item(); tot += xb.size(0)
    return succ / tot


def violation_breakdown(victim, X, y, spec):
    m = victim; m.eval(); Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    mean, scale, fmin, fmax = spec.mean, spec.scale, spec.fmin, spec.fmax
    is_int, is_imm = spec.is_int, spec.is_immutable; tol = 1e-4
    n = vb = vi = vm = vmul = vany = 0
    for i in range(0, len(Xt), EVAL_BATCH):
        xb = Xt[i:i+EVAL_BATCH].to(DEVICE); yb = yt[i:i+EVAL_BATCH].to(DEVICE)
        adv, _ = pgd_attack(m, xb, yb, spec, eps=EPS_MAIN, alpha=ALPHA, steps=STEPS, constrained=False)
        xr = adv * scale + mean; x0 = xb * scale + mean
        b = ((xr < fmin - tol) | (xr > fmax + tol)).any(1)
        it = (is_int & (torch.abs(xr - torch.round(xr)) > tol)).any(1)
        im = (is_imm & (torch.abs(xr - x0) > tol)).any(1)
        c = b.int() + it.int() + im.int()
        n += yb.size(0); vb += b.sum().item(); vi += it.sum().item(); vm += im.sum().item()
        vmul += (c >= 2).sum().item(); vany += (c >= 1).sum().item()
    f = lambda x: round(100.0 * x / n, 1)
    return {"n": n, "bounds": f(vb), "integrality": f(vi), "immutability": f(vm),
            "multiple": f(vmul), "any": f(vany)}


def four_way_ablation(victim, X, y, spec_raw):
    def acc(bounds, integ, immut):
        s = ConstraintSpec(spec_raw, DEVICE)
        if not integ: s.is_int = torch.zeros_like(s.is_int)
        if not immut: s.is_immutable = torch.zeros_like(s.is_immutable)
        if not bounds:
            s.fmin = torch.full_like(s.fmin, -1e30); s.fmax = torch.full_like(s.fmax, 1e30)
        a, _ = atk_acc(victim, X, y, s, True, EPS_MAIN); return round(a, 4)
    return {"bounds_only": acc(True, False, False),
            "bounds_integrality": acc(True, True, False),
            "bounds_immutability": acc(True, False, True),
            "all": acc(True, True, True)}


# ============ per-mode driver ============
def run_mode(mode):
    print(f"\n{'#'*60}\n# MODE = {mode}\n{'#'*60}")
    regen(mode)
    Xtr, ytr, Xv, yv, Xte, yte = load_split()
    spec_raw = load_spec(); spec = ConstraintSpec(spec_raw, DEVICE)
    classes = spec_raw["classes"]; k = len(classes); in_dim = Xtr.shape[1]
    benign_idx = classes.index("BENIGN") if "BENIGN" in classes else 0
    m = {"n_classes": k, "in_dim": in_dim, "classes": classes}

    # feature lists + class distribution (once, on multiclass)
    if mode == "multiclass":
        cols = spec_raw["columns"]
        R["feature_lists"] = {
            "immutable": [c for c, f in zip(cols, spec_raw["is_immutable"]) if f],
            "integer":   [c for c, f in zip(cols, spec_raw["is_integer"]) if f]}
        yall = np.concatenate([ytr, yv, yte])
        R["class_distribution"] = [
            {"class": c, "total": int((yall == i).sum()), "test": int((yte == i).sum())}
            for i, c in enumerate(classes)]

    # ---- victims + defenses for each architecture ----
    arch_rows, base_rows = [], []
    for arch in ARCHES:
        print(f"\n--- arch={arch} ---")
        victim = train_clean(arch, Xtr, ytr, Xv, yv, k, in_dim, VICTIM_EPOCHS)
        defended = train_adv(arch, Xtr, ytr, Xv, yv, spec, k, in_dim, True, DEF_EPOCHS)
        vc = clean_acc(victim, Xte, yte)
        vu, val = atk_acc(victim, Xte, yte, spec, False, EPS_MAIN)
        vk, _   = atk_acc(victim, Xte, yte, spec, True, EPS_MAIN)
        dc = clean_acc(defended, Xte, yte)
        du, _ = atk_acc(defended, Xte, yte, spec, False, EPS_MAIN)
        dk, _ = atk_acc(defended, Xte, yte, spec, True, EPS_MAIN)
        arch_rows += [[mode, arch, "undefended", vc, vu, vk],
                      [mode, arch, "defended",   dc, du, dk]]
        print(f"  victim  clean={vc:.4f} unc={vu:.4f}(val {val*100:.1f}%) con={vk:.4f}")
        print(f"  def     clean={dc:.4f} unc={du:.4f} con={dk:.4f}")

        if arch == "mlp":
            m["undef"] = {"clean": vc, "unc": vu, "con": vk, "validity": val}
            m["def"]   = {"clean": dc, "unc": du, "con": dk}
            # unconstrained-adv-training baseline (Table III)
            unc_def = train_adv(arch, Xtr, ytr, Xv, yv, spec, k, in_dim, False, DEF_EPOCHS)
            uc = clean_acc(unc_def, Xte, yte); uk, _ = atk_acc(unc_def, Xte, yte, spec, True, EPS_MAIN)
            base_rows += [[mode, "undefended", vc, vk],
                          [mode, "unconstrained_advtrain", uc, uk],
                          [mode, "constrained_advtrain", dc, dk]]
            m["baseline_unc_advtrain"] = {"clean": uc, "con": uk}
            # violation breakdown + 4-way ablation + macro-F1 + targeted (MLP)
            m["violations"] = violation_breakdown(victim, Xte, yte, spec)
            a4 = four_way_ablation(victim, Xte, yte, spec_raw)
            m["ablation4"] = a4
            m["ablation_immutable"] = {"yes": a4["all"], "no": a4["bounds_integrality"]}
            if mode == "multiclass":
                m["macro_f1"] = {
                    "undef_clean": round(f1_score(yte, preds_clean(victim, Xte), average="macro", zero_division=0), 4),
                    "undef_unc": round(f1_score(yte, preds_attacked(victim, Xte, yte, spec, False), average="macro", zero_division=0), 4),
                    "undef_con": round(f1_score(yte, preds_attacked(victim, Xte, yte, spec, True), average="macro", zero_division=0), 4),
                    "def_clean": round(f1_score(yte, preds_clean(defended, Xte), average="macro", zero_division=0), 4),
                    "def_unc": round(f1_score(yte, preds_attacked(defended, Xte, yte, spec, False), average="macro", zero_division=0), 4),
                    "def_con": round(f1_score(yte, preds_attacked(defended, Xte, yte, spec, True), average="macro", zero_division=0), 4)}
                # per-class constrained acc
                pc = preds_attacked(victim, Xte, yte, spec, True); pcd = preds_attacked(defended, Xte, yte, spec, True)
                R.setdefault("perclass", [])
                for ci, cn in enumerate(classes):
                    mask = yte == ci; n = int(mask.sum())
                    if n == 0: continue
                    R["perclass"].append({"class": cn, "test_n": n,
                        "undef": round(float((pc[mask] == ci).mean()), 4),
                        "def": round(float((pcd[mask] == ci).mean()), 4)})
            m["targeted"] = {"undef": round(targeted_benign(victim, Xte, yte, spec, benign_idx), 4),
                             "def": round(targeted_benign(defended, Xte, yte, spec, benign_idx), 4)}

        # ---- eps sweep (both archs, but store MLP for the figure/table) ----
        if arch == "mlp":
            sweep = []
            for eps in EPS_SWEEP:
                uu, uval = atk_acc(victim, Xte, yte, spec, False, eps)
                uk2, _   = atk_acc(victim, Xte, yte, spec, True, eps)
                du2, _   = atk_acc(defended, Xte, yte, spec, False, eps)
                dk2, _   = atk_acc(defended, Xte, yte, spec, True, eps)
                sweep.append({"eps": eps, "undef_unc": uu, "undef_con": uk2,
                              "def_unc": du2, "def_con": dk2, "validity": uval})
            m["sweep"] = sweep
            with open(os.path.join(EXP_DIR, f"sweep_{mode}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["eps", "model", "unconstrained_acc", "constrained_acc", "unconstrained_validity"])
                for s in sweep:
                    w.writerow([s["eps"], "undefended", s["undef_unc"], s["undef_con"], s["validity"]])
                    w.writerow([s["eps"], "defended", s["def_unc"], s["def_con"], s["validity"]])

    R[mode] = m
    # write arch + baseline csvs (append across modes)
    _append_csv(os.path.join(EXP_DIR, "arch_compare.csv"),
                ["mode", "arch", "model", "clean_acc", "unconstrained_atk_acc", "constrained_atk_acc"], arch_rows)
    _append_csv(os.path.join(EXP_DIR, "baseline_compare.csv"),
                ["mode", "defense", "clean_acc", "constrained_atk_acc"], base_rows)


def _append_csv(path, header, rows):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(header)
        w.writerows(rows)


# ============ output rendering ============
def write_outputs():
    with open(os.path.join(EXP_DIR, "all_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    # per-class csv
    if "perclass" in R:
        with open(os.path.join(EXP_DIR, "perclass_multiclass.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["class", "test_n", "undef_constrained_acc", "def_constrained_acc"])
            for r in R["perclass"]: w.writerow([r["class"], r["test_n"], r["undef"], r["def"]])
    if "class_distribution" in R:
        with open(os.path.join(EXP_DIR, "class_distribution.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["class", "total", "test"])
            for r in R["class_distribution"]: w.writerow([r["class"], r["total"], r["test"]])
    if "feature_lists" in R:
        with open(os.path.join(EXP_DIR, "feature_lists.json"), "w") as f:
            json.dump(R["feature_lists"], f, indent=2)

    # human-readable summary, table by table
    L = []
    def p(s=""): L.append(s)
    b = R.get("binary", {}); mc = R.get("multiclass", {})
    p("="*70); p("SUMMARY FOR PAPER (paste into tables)"); p("="*70)
    p("\n--- TABLE I (main results, acc) ---")
    for name, d in [("Binary", b), ("Multiclass", mc)]:
        u, df = d["undef"], d["def"]
        p(f"{name}:")
        p(f"  Clean         undef={u['clean']:.4f}  def={df['clean']:.4f}")
        p(f"  Unconstrained undef={u['unc']:.4f}  def={df['unc']:.4f}")
        p(f"  Constrained   undef={u['con']:.4f}  def={df['con']:.4f}")
    p(f"  validity rate (both): {b['undef']['validity']*100:.1f}% / {mc['undef']['validity']*100:.1f}%")

    p("\n--- TABLE II (architectures, binary) --- see arch_compare.csv")
    p("\n--- TABLE III (baseline) ---")
    for name, d in [("Binary", b), ("Multiclass", mc)]:
        bl = d["baseline_unc_advtrain"]
        p(f"{name}: undef con={d['undef']['con']:.4f} | unc-advtrain clean={bl['clean']:.4f} con={bl['con']:.4f} | con-advtrain clean={d['def']['clean']:.4f} con={d['def']['con']:.4f}")

    p("\n--- TABLE IV (immutable ablation) + 4-WAY ---")
    for name, d in [("Binary", b), ("Multiclass", mc)]:
        a4 = d["ablation4"]
        p(f"{name}: bounds={a4['bounds_only']:.4f} b+int={a4['bounds_integrality']:.4f} b+imm={a4['bounds_immutability']:.4f} all={a4['all']:.4f}")

    p("\n--- MACRO-F1 (multiclass) ---")
    if "macro_f1" in mc:
        f = mc["macro_f1"]
        p(f"  undef: clean={f['undef_clean']:.4f} unc={f['undef_unc']:.4f} con={f['undef_con']:.4f}")
        p(f"  def:   clean={f['def_clean']:.4f} unc={f['def_unc']:.4f} con={f['def_con']:.4f}")

    p("\n--- VIOLATION BREAKDOWN (binary, unconstrained) ---")
    v = b["violations"]
    p(f"  bounds={v['bounds']}% integ={v['integrality']}% immut={v['immutability']}% multi={v['multiple']}% any={v['any']}%")

    p("\n--- TARGETED BENIGN-CLASS SUCCESS RATE ---")
    for name, d in [("Binary", b), ("Multiclass", mc)]:
        t = d["targeted"]; p(f"  {name}: undefended={t['undef']:.4f}  defended={t['def']:.4f}")

    with open(os.path.join(EXP_DIR, "summary_for_paper.txt"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    # figures + table snippets
    try:
        import make_figures as MF
        MF.EXP_DIR = EXP_DIR; MF.FIG_DIR = os.path.join(EXP_DIR, "figures")
        os.makedirs(MF.FIG_DIR, exist_ok=True)
        MF.main()
    except Exception as e:
        print(f"[warn] make_figures failed ({e}); CSVs are still written.")


def main():
    t0 = time.time()
    # fresh csvs
    for fn in ["arch_compare.csv", "baseline_compare.csv"]:
        pth = os.path.join(EXP_DIR, fn)
        if os.path.exists(pth): os.remove(pth)
    for mode in MODES:
        run_mode(mode)
    write_outputs()
    print(f"\n[ALL DONE] {(time.time()-t0)/60:.1f} min. Everything in {EXP_DIR}")
    print("Paste from summary_for_paper.txt; figures in figures/.")


if __name__ == "__main__":
    main()
