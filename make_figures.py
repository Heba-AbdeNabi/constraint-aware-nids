"""
make_figures.py
===============
Reads the CSV outputs of run_experiments.py and produces publication-quality
PDF figures + LaTeX table snippets for the paper.

INPUTS  (in EXP_DIR):
  sweep_binary.csv, sweep_multiclass.csv
  ablation_binary.csv, ablation_multiclass.csv
  perclass_multiclass.csv
  summary_table.csv

OUTPUTS (in EXP_DIR/figures):
  fig_sweep_binary.pdf, fig_sweep_multiclass.pdf   -> accuracy vs eps curves
  fig_perclass_multiclass.pdf                       -> per-class bar chart
  tab_ablation.tex                                  -> ablation LaTeX table
  tab_main.tex                                       -> main results LaTeX table

Just Run in PyCharm after run_experiments.py finishes. Matplotlib only.
"""
import os, csv
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
BASE_DIR = "."
EXP_DIR  = os.path.join(BASE_DIR, "outputs", "experiments")
FIG_DIR  = os.path.join(EXP_DIR, "figures")
# ----------------------------------------------------------------------
os.makedirs(FIG_DIR, exist_ok=True)

# consistent, print-friendly styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------- sweep curves ----------------
def plot_sweep(mode):
    path = os.path.join(EXP_DIR, f"sweep_{mode}.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    rows = read_csv(path)

    def series(model, col):
        pts = [(float(r["eps"]), float(r[col])) for r in rows if r["model"] == model]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    # undefended
    ex, uc = series("undefended", "constrained_acc")
    _, uu = series("undefended", "unconstrained_acc")
    # defended
    dx, dc = series("defended", "constrained_acc")
    _, du = series("defended", "unconstrained_acc")

    ax.plot(ex, uc, "o-",  color="#c0392b", label="Undefended, constrained")
    ax.plot(ex, uu, "o--", color="#e67e22", label="Undefended, unconstrained")
    ax.plot(dx, dc, "s-",  color="#27ae60", label="Defended, constrained")
    ax.plot(dx, du, "s--", color="#2980b9", label="Defended, unconstrained")

    ax.set_xlabel(r"Perturbation budget $\epsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="lower left")
    out = os.path.join(FIG_DIR, f"fig_sweep_{mode}.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}")


# ---------------- per-class bars ----------------
def plot_perclass():
    path = os.path.join(EXP_DIR, "perclass_multiclass.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    rows = read_csv(path)

    # Optionally declutter: drop the low-sample "- Attempted" sub-classes so the
    # figure shows the main attack families. Set DROP_ATTEMPTED=False to keep all.
    DROP_ATTEMPTED = True
    if DROP_ATTEMPTED:
        rows = [r for r in rows if "attempted" not in r["class"].lower()]

    classes = [r["class"] for r in rows]
    undef = [float(r["undef_constrained_acc"]) for r in rows]
    defd  = [float(r["def_constrained_acc"]) for r in rows]

    x = range(len(classes)); w = 0.4
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.bar([i - w/2 for i in x], undef, w, label="Undefended", color="#c0392b")
    ax.bar([i + w/2 for i in x], defd,  w, label="Defended",   color="#27ae60")
    ax.set_xticks(list(x)); ax.set_xticklabels(classes, rotation=45, ha="right",
                                               fontsize=7)
    ax.set_ylabel("Constrained-attack acc.")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    out = os.path.join(FIG_DIR, "fig_perclass_multiclass.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}")


# ---------------- ablation LaTeX table ----------------
def table_ablation():
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Immutability-constraint ablation: constrained-attack "
             r"accuracy on the undefended detector, with and without the "
             r"immutable-feature mask ($\epsilon=0.3$).}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"\textbf{Setting} & \textbf{Immutable mask} & "
             r"\textbf{Constrained acc.}\\", r"\midrule"]
    for mode in ["binary", "multiclass"]:
        path = os.path.join(EXP_DIR, f"ablation_{mode}.csv")
        if not os.path.exists(path):
            continue
        for r in read_csv(path):
            mask = "Yes" if r["immutable_mask"].lower() == "true" else "No"
            lines.append(f"{mode.capitalize()} & {mask} & "
                         f"{float(r['constrained_acc']):.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = os.path.join(FIG_DIR, "tab_ablation.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[tab] {out}")


# ---------------- main results LaTeX table (from summary) ----------------
def table_main():
    path = os.path.join(EXP_DIR, "summary_table.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    rows = read_csv(path)
    def get(mode, setting, col):
        for r in rows:
            if r["mode"] == mode and r["setting"] == setting:
                return r[col]
        return ""
    def fmt(v):
        try: return f"{float(v):.4f}"
        except: return "--"

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Detection accuracy under clean, unconstrained, and "
             r"constrained (realizable) attacks ($\epsilon=0.3$).}",
             r"\label{tab:main}",
             r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"\textbf{Setting} & \textbf{Condition} & \textbf{Undefended} & "
             r"\textbf{Defended}\\", r"\midrule"]
    for mode in ["binary", "multiclass"]:
        lines.append(r"\multirow{3}{*}{" + mode.capitalize() + "}")
        lines.append(f" & Clean & {fmt(get(mode,'clean','undefended'))} & "
                     f"{fmt(get(mode,'clean','defended'))}\\\\")
        lines.append(f" & Unconstrained attack & "
                     f"{fmt(get(mode,'unconstrained_atk@0.3','undefended'))} & "
                     f"{fmt(get(mode,'unconstrained_atk@0.3','defended'))}\\\\")
        lines.append(f" & Constrained attack & "
                     f"{fmt(get(mode,'constrained_atk@0.3','undefended'))} & "
                     f"\\textbf{{{fmt(get(mode,'constrained_atk@0.3','defended'))}}}\\\\")
        lines.append(r"\midrule" if mode == "binary" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    out = os.path.join(FIG_DIR, "tab_main.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[tab] {out}")


def table_arch():
    path = os.path.join(EXP_DIR, "arch_compare.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    rows = read_csv(path)
    def g(mode, arch, model, col):
        for r in rows:
            if r["mode"]==mode and r["arch"]==arch and r["model"]==model:
                return f"{float(r[col]):.4f}"
        return "--"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Robustness across victim architectures (MLP vs.\ "
             r"1D-CNN) at $\epsilon=0.3$, binary setting.}",
             r"\label{tab:arch}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"\textbf{Arch.} & \textbf{Model} & \textbf{Clean} & "
             r"\textbf{Unconstr.} & \textbf{Constr.}\\", r"\midrule"]
    for arch, disp in [("mlp","MLP"), ("cnn","1D-CNN")]:
        lines.append(r"\multirow{2}{*}{" + disp + "}")
        for model, mdisp in [("undefended","Undefended"), ("defended","Defended")]:
            lines.append(f" & {mdisp} & "
                         f"{g('binary',arch,model,'clean_acc')} & "
                         f"{g('binary',arch,model,'unconstrained_atk_acc')} & "
                         f"{g('binary',arch,model,'constrained_atk_acc')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(FIG_DIR, "tab_arch.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"[tab] {os.path.join(FIG_DIR,'tab_arch.tex')}")


def table_baseline():
    path = os.path.join(EXP_DIR, "baseline_compare.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return
    rows = read_csv(path)
    # default to MLP architecture for the main-text baseline table
    def g(mode, defense, col, arch="mlp"):
        for r in rows:
            if r["mode"]==mode and r["defense"]==defense and r.get("arch","mlp")==arch:
                return f"{float(r[col]):.4f}"
        return "--"
    disp = {"undefended":"Undefended",
            "unconstrained_advtrain":"Unconstrained adv.\\ training",
            "constrained_advtrain":"Constrained adv.\\ training (ours)"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Effect of the training-time attack on robustness to the "
             r"realizable (constrained) attack at $\epsilon=0.3$ (MLP, binary).}",
             r"\label{tab:baseline}",
             r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"\textbf{Defense} & \textbf{Clean acc.} & "
             r"\textbf{Constrained-atk acc.}\\", r"\midrule"]
    for d in ["undefended","unconstrained_advtrain","constrained_advtrain"]:
        lines.append(f"{disp[d]} & {g('binary',d,'clean_acc')} & "
                     f"{g('binary',d,'constrained_atk_acc')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(FIG_DIR, "tab_baseline.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"[tab] {os.path.join(FIG_DIR,'tab_baseline.tex')}")


def main():
    plot_sweep("binary")
    plot_sweep("multiclass")
    plot_perclass()
    table_ablation()
    table_main()
    table_arch()
    table_baseline()
    print(f"\n[done] figures + tables in:\n  {FIG_DIR}")
    print("Copy the .pdf files next to main.tex and \\includegraphics them;")
    print("paste the .tex table snippets to replace the placeholders.")


if __name__ == "__main__":
    main()
