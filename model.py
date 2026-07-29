
import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IDSNet(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256, 128, 64), p=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(out_dir):
    d = np.load(os.path.join(out_dir, "data.npz"))
    return (d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"])


def make_loader(X, y, bs=512, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    all_p, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
        all_p.append(pred.cpu()); all_y.append(yb.cpu())
    acc = correct / total
    return acc, torch.cat(all_p).numpy(), torch.cat(all_y).numpy()


def prf(y_true, y_pred, n_classes):
    """Macro precision/recall/f1 without sklearn dependency at call site."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return p, r, f


def train(model, tr_loader, val_loader, n_classes, epochs=30, lr=1e-3, wd=1e-5):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # class weights help with CIC-IDS2017 imbalance
    ys = np.concatenate([yb.numpy() for _, yb in tr_loader])
    counts = np.bincount(ys, minlength=n_classes).astype(float)
    w = torch.tensor((counts.sum() / (counts + 1e-6)), dtype=torch.float32)
    w = (w / w.sum() * n_classes).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)

    best_val, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        val_acc, _, _ = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d} | val_acc {val_acc:.4f} (best {best_val:.4f})")
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(args.out_dir)
    with open(os.path.join(args.out_dir, "constraint_spec.json")) as f:
        spec = json.load(f)
    n_classes = len(spec["classes"]); in_dim = X_tr.shape[1]
    print(f"[model] in_dim={in_dim} n_classes={n_classes} device={DEVICE}")

    tr = make_loader(X_tr, y_tr, shuffle=True)
    val = make_loader(X_val, y_val)
    te = make_loader(X_te, y_te)

    model = IDSNet(in_dim, n_classes)
    model = train(model, tr, val, n_classes, epochs=args.epochs)

    acc, yp, yt = evaluate(model, te)
    p, r, f = prf(yt, yp, n_classes)
    print(f"\n[CLEAN BASELINE] test_acc={acc:.4f} macro-P={p:.4f} "
          f"macro-R={r:.4f} macro-F1={f:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "victim.pt"))
    with open(os.path.join(args.out_dir, "baseline_metrics.json"), "w") as fp:
        json.dump({"clean_acc": acc, "macro_p": p, "macro_r": r, "macro_f1": f}, fp)
    print("[save] victim.pt, baseline_metrics.json")


if __name__ == "__main__":
    main()
