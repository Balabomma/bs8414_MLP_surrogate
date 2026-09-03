"""Side-by-side comparison: MLP ablation vs the KAN champion.

Reads the metrics each project's own frozen eval script already wrote — this
script never re-scores anything with different code, it only tabulates:

    <model_dir>/outputs/overall_metrics.csv         (evaluate.py / evaluate_v9.py)
    <model_dir>/outputs/killer_excluded_metrics.txt (killer_excluded*.py)

Families compared (each a 3-replicate population, same clean-60 40/12/8 split):
    KAN : bs8414_KAN_surrogate/models_v9, models_v9_r2, models_v9_r3
    MLP : bs8414_MLP_surrogate/models_mlp, models_mlp_r2, models_mlp_r3

The verdict applies the project's standing rule: a delta inside the +/-0.02 R2
retrain-variance band is reported as INCONCLUSIVE, not as a win.

Optional `--bench` times a full 60-sim forward pass for both architectures
(MLP here, KAN in a subprocess using its own venv) to quantify inference cost.

Usage:
    python compare_with_kan.py
    python compare_with_kan.py --bench
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from config import PROJECT_DIR, KAN_PROJECT_DIR

VARIANCE_BAND = 0.02

KAN_DIRS = ["models_v9", "models_v9_r2", "models_v9_r3"]
MLP_DIRS = ["models_mlp", "models_mlp_r2", "models_mlp_r3"]

KAN_BENCH = r"""
import time, numpy as np, torch
from config import N_SENSORS, N_TIMESTEPS
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6
from anchor_features import build_bank, anchors_for
from model_v7 import KANAttentionLSTMv7
from train_v3 import HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, NUM_KNOTS
from features_v6 import N_V6_FEATURES
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params, outputs, masks, meta, _ = build_dataset()
_, _, _, scaler, si, ta = prepare_data_splits(params, outputs, masks, meta)
bank = build_bank(params, outputs, si["train_idx"])
pv6 = build_params_v6(params, meta, bank)
an = anchors_for(params, bank)
m = KANAttentionLSTMv7(n_timesteps=N_TIMESTEPS, n_sensors=N_SENSORS,
                       hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                       n_heads=N_HEADS, dropout=DROPOUT, num_knots=NUM_KNOTS,
                       n_continuous=2 + 13 + N_V6_FEATURES).to(dev).eval()
p = torch.FloatTensor(pv6).to(dev); a = torch.FloatTensor(an).to(dev)
t = torch.FloatTensor(ta).to(dev)
with torch.no_grad():
    for _ in range(3): m(p, t, a)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(20): m(p, t, a)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    dt = (time.perf_counter() - t0) / 20
n_par = sum(q.numel() for q in m.parameters() if q.requires_grad)
print(f"RESULT {n_par} {dt*1000:.2f} {len(pv6)}")
"""


def read_metrics(model_dir):
    """-> dict(set -> dict) plus killer-excluded R2, or None if not evaluated yet."""
    csv = os.path.join(model_dir, "outputs", "overall_metrics.csv")
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv).set_index("Set")
    out = {s: df.loc[s].to_dict() for s in df.index}
    kx = os.path.join(model_dir, "outputs", "killer_excluded_metrics.txt")
    if os.path.isfile(kx):
        txt = open(kx).read()
        blocks = re.split(r"\n\[(Valid|Test)\]", txt)
        for i in range(1, len(blocks) - 1, 2):
            name, body = blocks[i], blocks[i + 1]
            m = re.search(r"excl killer,\s*\d+\):\s*R2=([\d.\-]+)", body)
            if m and name in out:
                out[name]["r2_xkiller"] = float(m.group(1))
    return out


def family(dirs, root, label):
    rows = []
    for d in dirs:
        full = os.path.join(root, d)
        met = read_metrics(full)
        if met is None:
            print(f"  [skip] {label} {d}: no outputs/overall_metrics.csv "
                  f"(run its evaluate script first)")
            continue
        rows.append({
            "replicate": d,
            "valid_r2": met["Valid"]["r2"], "test_r2": met["Test"]["r2"],
            "valid_xk": met["Valid"].get("r2_xkiller", np.nan),
            "test_xk": met["Test"].get("r2_xkiller", np.nan),
            "valid_rmse": met["Valid"]["rmse"], "test_rmse": met["Test"]["rmse"],
            "valid_mae": met["Valid"]["mae"], "test_mae": met["Test"]["mae"],
            "valid_mape": met["Valid"]["mape"], "test_mape": met["Test"]["mape"],
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["combined"] = (df["valid_r2"] + df["test_r2"]) / 2
    return df


def summarise(df):
    return {c: (df[c].mean(), df[c].std(ddof=1) if len(df) > 1 else 0.0)
            for c in df.columns if c != "replicate"}


def fmt(ms):
    return f"{ms[0]:.4f} +/- {ms[1]:.4f}"


def bench_mlp():
    import time
    import torch
    from config import N_SENSORS, N_TIMESTEPS
    from data_loader import build_dataset, prepare_data_splits
    from features_v6 import build_params_v6, N_V6_FEATURES
    from anchor_features import build_bank, anchors_for
    from model import MLPAttentionLSTM, count_parameters
    from train import HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, NUM_KNOTS

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, si, ta = prepare_data_splits(params, outputs, masks, meta)
    bank = build_bank(params, outputs, si["train_idx"])
    pv6 = build_params_v6(params, meta, bank)
    an = anchors_for(params, bank)
    m = MLPAttentionLSTM(n_timesteps=N_TIMESTEPS, n_sensors=N_SENSORS,
                         hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                         n_heads=N_HEADS, dropout=DROPOUT, num_knots=NUM_KNOTS,
                         n_continuous=2 + 13 + N_V6_FEATURES).to(dev).eval()
    p = torch.FloatTensor(pv6).to(dev)
    a = torch.FloatTensor(an).to(dev)
    t = torch.FloatTensor(ta).to(dev)
    with torch.no_grad():
        for _ in range(3):
            m(p, t, a)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            m(p, t, a)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    return count_parameters(m), dt * 1000, len(pv6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true",
                    help="also time a 60-sim forward pass for both architectures")
    ap.add_argument("--out", default=os.path.join(PROJECT_DIR,
                                                  "comparison_kan_vs_mlp.md"))
    args = ap.parse_args()

    print("=" * 78)
    print("  KAN vs MLP — same data, same split, same recipe, same eval contract")
    print("=" * 78)

    kan = family(KAN_DIRS, KAN_PROJECT_DIR, "KAN")
    mlp = family(MLP_DIRS, PROJECT_DIR, "MLP")
    if kan is None or mlp is None:
        print("\n  Not enough evaluated replicates on one side — nothing to compare.")
        return 1

    lines = []
    lines.append("# KAN vs MLP — BS 8414 thermocouple surrogate\n")
    lines.append("Same 60 FDS simulations, same deterministic 70/15/15 CHID-hash split "
                 "(40/12/8), same 39-feature physics-causal input vector, same training "
                 "recipe (tau=1.8 robust loss, ambient clamp, 12 candidates -> greedy "
                 "ensemble), same frozen eval contract. The only architectural "
                 "difference is KAN B-spline edge blocks vs parameter-matched MLP "
                 "blocks.\n")

    for label, df in (("KAN (models_v9 family)", kan), ("MLP (models_mlp family)", mlp)):
        lines.append(f"\n## {label} — {len(df)} replicate(s)\n")
        lines.append("| replicate | Valid R² | Test R² | Combined | Valid-xk | Test-xk |"
                     " Test RMSE | Test MAE | Test MAPE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            lines.append(f"| {r['replicate']} | {r['valid_r2']:.4f} | {r['test_r2']:.4f} "
                         f"| {r['combined']:.4f} | {r['valid_xk']:.4f} | {r['test_xk']:.4f} "
                         f"| {r['test_rmse']:.1f} | {r['test_mae']:.1f} "
                         f"| {r['test_mape']:.1f}% |")

    ks, ms = summarise(kan), summarise(mlp)
    lines.append("\n## Head-to-head (mean +/- sample std over replicates)\n")
    lines.append("| metric | KAN | MLP | delta (MLP-KAN) | verdict |")
    lines.append("|---|---|---|---:|---|")
    for key, name, band in [
        ("valid_r2", "Valid R²", VARIANCE_BAND),
        ("test_r2", "Test R²", VARIANCE_BAND),
        ("combined", "Combined R²", VARIANCE_BAND),
        ("valid_xk", "Valid R² (excl. killer)", VARIANCE_BAND),
        ("test_xk", "Test R² (excl. killer)", VARIANCE_BAND),
    ]:
        d = ms[key][0] - ks[key][0]
        if np.isnan(d):
            verdict = "n/a"
        elif abs(d) < band:
            verdict = f"INCONCLUSIVE (|Δ| < {band} band)"
        else:
            verdict = "MLP better" if d > 0 else "KAN better"
        lines.append(f"| {name} | {fmt(ks[key])} | {fmt(ms[key])} | {d:+.4f} | {verdict} |")
    for key, name, unit in [("test_rmse", "Test RMSE", "°C"),
                            ("test_mae", "Test MAE", "°C"),
                            ("test_mape", "Test MAPE(T>100)", "%")]:
        d = ms[key][0] - ks[key][0]
        lines.append(f"| {name} ({unit}) | {ks[key][0]:.2f} +/- {ks[key][1]:.2f} | "
                     f"{ms[key][0]:.2f} +/- {ms[key][1]:.2f} | {d:+.2f} | "
                     f"{'MLP lower' if d < 0 else 'KAN lower'} |")

    if args.bench:
        lines.append("\n## Cost (untrained instances, identical batch = all 60 sims)\n")
        n_mlp, ms_mlp, n_sims = bench_mlp()
        kan_py = os.path.join(KAN_PROJECT_DIR, "venv", "Scripts", "python.exe")
        if not os.path.isfile(kan_py):
            kan_py = sys.executable
        r = subprocess.run([kan_py, "-c", KAN_BENCH], cwd=KAN_PROJECT_DIR,
                           capture_output=True, text=True)
        m = re.search(r"RESULT (\d+) ([\d.]+) (\d+)", r.stdout or "")
        lines.append("| architecture | parameters | forward pass "
                     f"({n_sims} sims x 181 steps x 24 TC) |")
        lines.append("|---|---:|---:|")
        if m:
            lines.append(f"| KAN | {int(m.group(1)):,} | {float(m.group(2)):.2f} ms |")
        else:
            lines.append("| KAN | (bench failed) | — |")
        lines.append(f"| MLP | {n_mlp:,} | {ms_mlp:.2f} ms |")

    lines.append("\n---\n")
    lines.append("Standing rules applied: deltas inside the ±0.02 R² retrain-variance "
                 "band are inconclusive, not wins (VARIANCE_RECORD.md); the killer "
                 "family `Test_1_PE_PIR HRR1333` is LES-chaotic and reported both "
                 "included and excluded; physics sanity is reported separately by "
                 "`validate_physics.py` and must pass for any result to count.\n")

    report = "\n".join(lines)
    print("\n" + report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
