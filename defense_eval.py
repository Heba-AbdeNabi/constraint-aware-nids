
import argparse, os, json
import numpy as np
import torch
from model import IDSNet, load_data, make_loader, evaluate, DEVICE
from attack import ConstraintSpec, pgd_attack


@torch.no_grad()
def _acc_on(model, X, y):
    model.eval()
    xb = torch.from_numpy(X).float().to(DEVICE)
    yb = torch.from_numpy(y).long().to(DEVICE)
    return (model(xb).argmax(1) == yb).float().mean().item()


def attack_accuracy(model, X, y, spec, constrained, eps, steps, alpha, bs=1024):
    """Accuracy under attack (lower = attack more successful). Only ATTACK-class
    rows are perturbed for evasion; benign rows pass through."""
    model.eval()
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    correct = total = 0
    valid_count = valid_total = 0
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i+bs].to(DEVICE); yb = yt[i:i+bs].to(DEVICE)
        adv, valid = pgd_attack(model, xb, yb, spec, eps=eps, alpha=alpha,
                                steps=steps, constrained=constrained)
        pred = model(adv).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
        valid_count += valid.sum().item(); valid_total += valid.numel()
    return correct / total, valid_count / valid_total


def adversarial_train(model, X_tr, y_tr, X_val, y_val, spec, n_classes,
                      epochs=20, lr=1e-3, eps=0.3, alpha=0.05, steps=7, bs=512):
    """Adversarial training using CONSTRAINED PGD examples mixed with clean."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)
    w = (w / w.sum() * n_classes).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=w)

    Xt = torch.from_numpy(X_tr).float(); yt = torch.from_numpy(y_tr).long()
    idx = np.arange(len(Xt))
    best_val, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            # generate constrained adversarial half
            model.eval()
            adv, _ = pgd_attack(model, xb, yb, spec, eps=eps, alpha=alpha,
                                steps=steps, constrained=True)
            model.train()
            x_mix = torch.cat([xb, adv], 0); y_mix = torch.cat([yb, yb], 0)
            opt.zero_grad(); loss = crit(model(x_mix), y_mix)
            loss.backward(); opt.step()
        va = _acc_on(model, X_val, y_val)
        if va > best_val:
            best_val = va; best_state = {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  [adv-train] epoch {ep:3d} | val_acc {va:.4f} (best {best_val:.4f})")
    if best_state: model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(args.out_dir)
    with open(os.path.join(args.out_dir, "constraint_spec.json")) as f:
        spec_raw = json.load(f)
    n_classes = len(spec_raw["classes"]); in_dim = X_tr.shape[1]
    spec = ConstraintSpec(spec_raw, DEVICE)

    # ---- Undefended victim ----
    victim = IDSNet(in_dim, n_classes).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(args.out_dir, "victim.pt"),
                                      map_location=DEVICE))
    R = {}
    R["undef_clean"] = _acc_on(victim, X_te, y_te)
    R["undef_unconstrained"], R["unconstrained_validity"] = attack_accuracy(
        victim, X_te, y_te, spec, constrained=False,
        eps=args.eps, steps=args.steps, alpha=args.alpha)
    R["undef_constrained"], _ = attack_accuracy(
        victim, X_te, y_te, spec, constrained=True,
        eps=args.eps, steps=args.steps, alpha=args.alpha)

    print("\n--- UNDEFENDED ---")
    print(f"clean acc              : {R['undef_clean']:.4f}")
    print(f"unconstrained-atk acc  : {R['undef_unconstrained']:.4f} "
          f"(only {R['unconstrained_validity']*100:.1f}% of adv examples are VALID)")
    print(f"constrained-atk acc    : {R['undef_constrained']:.4f}")

    # ---- Defended model (adversarial training on constrained examples) ----
    print("\n[defense] adversarial training on constrained PGD...")
    defended = IDSNet(in_dim, n_classes).to(DEVICE)
    defended = adversarial_train(defended, X_tr, y_tr, X_val, y_val, spec,
                                 n_classes, epochs=args.epochs,
                                 eps=args.eps, alpha=args.alpha, steps=7)

    R["def_clean"] = _acc_on(defended, X_te, y_te)
    R["def_unconstrained"], _ = attack_accuracy(
        defended, X_te, y_te, spec, constrained=False,
        eps=args.eps, steps=args.steps, alpha=args.alpha)
    R["def_constrained"], _ = attack_accuracy(
        defended, X_te, y_te, spec, constrained=True,
        eps=args.eps, steps=args.steps, alpha=args.alpha)

    print("\n--- DEFENDED (constrained adversarial training) ---")
    print(f"clean acc              : {R['def_clean']:.4f}")
    print(f"unconstrained-atk acc  : {R['def_unconstrained']:.4f}")
    print(f"constrained-atk acc    : {R['def_constrained']:.4f}")

    torch.save(defended.state_dict(), os.path.join(args.out_dir, "defended.pt"))
    with open(os.path.join(args.out_dir, "results.json"), "w") as fp:
        json.dump(R, fp, indent=2)

    # Pretty results table for the paper
    print("\n================ MAIN RESULTS TABLE ================")
    print(f"{'Setting':<22}{'Undefended':>12}{'Defended':>12}")
    print(f"{'Clean':<22}{R['undef_clean']:>12.4f}{R['def_clean']:>12.4f}")
    print(f"{'Unconstrained atk':<22}{R['undef_unconstrained']:>12.4f}{R['def_unconstrained']:>12.4f}")
    print(f"{'Constrained atk':<22}{R['undef_constrained']:>12.4f}{R['def_constrained']:>12.4f}")
    print(f"\nUnconstrained-attack validity rate: {R['unconstrained_validity']*100:.1f}%")
    print("[save] defended.pt, results.json")


if __name__ == "__main__":
    main()
