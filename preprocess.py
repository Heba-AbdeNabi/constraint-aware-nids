import argparse, os, json, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

IMMUTABLE_HINTS = [
    "protocol", "destination port", "flow id", "source", "destination",
    "timestamp", "fwd psh flags", "bwd psh flags", "fin flag", "syn flag",
    "rst flag", "psh flag", "ack flag", "urg flag", "cwe flag", "ece flag",
]
INTEGER_HINTS = [
    "total", "count", "packets", "flags", "number", "min", "max", "port",
]
LABEL_CANDIDATES = ["label", " label", "attack", "class"]


def _norm(col: str) -> str:
    return col.strip().lower()


def find_label_column(df):
    for c in df.columns:
        if _norm(c) in [x.strip() for x in LABEL_CANDIDATES]:
            return c
    return df.columns[-1]


def load_raw(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}.")
    print(f"[load] found {len(files)} CSV files")
    dfs = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        dfs.append(d)
        print(f"       {os.path.basename(f)}: {d.shape}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[load] combined shape: {df.shape}")
    return df


def clean(df):
    label_col = find_label_column(df)
    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    drop_like = ["flow id", "source ip", "src ip", "destination ip", "dst ip",
                 "timestamp", "unnamed"]
    to_drop = [c for c in X.columns if any(k in _norm(c) for k in drop_like)]
    if to_drop:
        X = X.drop(columns=to_drop)
        print(f"[clean] dropped id-like columns: {to_drop}")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    before = len(X)
    mask = X.notna().all(axis=1)
    X, y_raw = X[mask], y_raw[mask]
    print(f"[clean] dropped {before - len(X)} rows with nan/inf")
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        print(f"[clean] dropped {len(const_cols)} constant columns")
    return X.reset_index(drop=True), y_raw.reset_index(drop=True)


def build_constraint_spec(X_df):
    cols = list(X_df.columns)
    feat_min = X_df.min().values.astype(float)
    feat_max = X_df.max().values.astype(float)
    is_integer, is_immutable = [], []
    for c in cols:
        nc = _norm(c)
        col_vals = X_df[c].values
        int_by_hint = any(h in nc for h in INTEGER_HINTS)
        int_by_data = np.allclose(col_vals, np.round(col_vals))
        is_integer.append(bool(int_by_hint or int_by_data))
        is_immutable.append(bool(any(h in nc for h in IMMUTABLE_HINTS)))
    spec = {
        "columns": cols, "min": feat_min.tolist(), "max": feat_max.tolist(),
        "is_integer": is_integer, "is_immutable": is_immutable,
    }
    print(f"[spec] {len(cols)} features | {sum(is_immutable)} immutable | "
          f"{sum(is_integer)} integer")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="./data")
    ap.add_argument("--out_dir",
                    default="./outputs")
    ap.add_argument("--binary", action="store_true", default=False,
                    help="collapse labels to BENIGN vs ATTACK (recommended for v1)")
    ap.add_argument("--min_class", type=int, default=10,
                    help="drop classes with fewer than this many samples")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_raw(args.data_dir)
    X_df, y_raw = clean(df)

    if args.binary:
        y = (~y_raw.str.upper().str.contains("BENIGN")).astype(int).values
        classes = ["BENIGN", "ATTACK"]
        print(f"[label] binary -> benign={np.sum(y==0)} attack={np.sum(y==1)}")
    else:
        vc = y_raw.value_counts()
        rare = vc[vc < args.min_class].index.tolist()
        if rare:
            keep = ~y_raw.isin(rare)
            X_df = X_df[keep].reset_index(drop=True)
            y_raw = y_raw[keep].reset_index(drop=True)
            print(f"[label] dropped {len(rare)} rare classes "
                  f"(<{args.min_class} samples): {rare}")
        le = LabelEncoder()
        y = le.fit_transform(y_raw.values)
        classes = le.classes_.tolist()
        print(f"[label] multiclass -> {len(classes)} classes")

    spec = build_constraint_spec(X_df)

    X = X_df.values.astype(np.float32)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    spec["scaler_mean"] = scaler.mean_.tolist()
    spec["scaler_scale"] = scaler.scale_.tolist()
    spec["classes"] = classes

    np.savez_compressed(
        os.path.join(args.out_dir, "data.npz"),
        X_tr=X_tr_s, y_tr=y_tr, X_val=X_val_s, y_val=y_val,
        X_te=X_te_s, y_te=y_te)
    with open(os.path.join(args.out_dir, "constraint_spec.json"), "w") as f:
        json.dump(spec, f)

    print(f"[save] data.npz  train={X_tr_s.shape} val={X_val_s.shape} test={X_te_s.shape}")
    print(f"[save] constraint_spec.json")
    print("[done] preprocessing complete.")


if __name__ == "__main__":
    main()
