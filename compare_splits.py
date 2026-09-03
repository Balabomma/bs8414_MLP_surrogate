"""Compare the MLP surrogate trained under 70/15/15 vs 80/10/10 split ratios.

Why this script exists rather than just diffing the two `overall_metrics.csv`
files: the two ratios do not hold out the same simulations, so their "Test R2"
numbers are computed on different sims and are NOT comparable. The only honest
arena is the set of simulations held out from training AND from model selection
by BOTH ratios. That arena is derived from the split indices each side actually
used — never hardcoded — so this works for the configuration-grouped split and
the legacy CHID split alike.

Sims that one side holds out but the other uses for validation are excluded from
the head-to-head and listed as a contamination note: the second side chose its
ensemble on them, so it is not blind to them.

Each ensemble is reconstructed exactly as its own evaluate.py would — its own
anchor bank, its own output scaler, its own split — and scored with the frozen
metric definitions (masked-point pooling; R2/RMSE/MAE/MAPE(T>100) never averaged
per-sim).

Each side may carry several replicates: identical reruns whose spread IS the
retrain-variance band the verdict is judged against.

THE LEAKAGE APPENDIX
--------------------
The legacy CHID split hashes the CHID string, so the three mesh siblings of one
physical configuration scatter across train/valid/test — a held-out case whose
0.09 m and 0.10 m twins are in training is not a blind prediction. When CHID-line
replicates are supplied the report quantifies that: how many test cases had a
mesh sibling in training, and what it did to the reported score.

Usage:
    python compare_splits.py                    # grouped headline + CHID appendix
    python compare_splits.py --no-appendix
    python compare_splits.py --dirs-a A1 A2 --dirs-b B1 B2 --split-mode chid
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import PROJECT_DIR, N_SENSORS, OUTPUT_DIR

# The report is UTF-8, but this console (and the redirected .log) is cp1252, which
# cannot encode the greek delta used in the verdict strings. Degrade instead of
# dying after all the work is done — the markdown file keeps the real characters.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

VARIANCE_BAND = 0.02          # standing rule: |delta| < 0.02 R2 is inconclusive

GROUPED_A = ["models_mlp_70_15_15_grouped", "models_mlp_70_15_15_grouped_r2",
             "models_mlp_70_15_15_grouped_r3"]
GROUPED_B = ["models_mlp_80_10_10_grouped", "models_mlp_80_10_10_grouped_r2",
             "models_mlp_80_10_10_grouped_r3"]
CHID_A = ["models_mlp_70_15_15", "models_mlp_70_15_15_r2", "models_mlp_70_15_15_r3"]
CHID_B = ["models_mlp_80_10_10", "models_mlp_80_10_10_r2", "models_mlp_80_10_10_r3"]

# Predictions are produced in a subprocess with MLP_SPLIT / MLP_SPLIT_MODE set to
# that replicate's scheme: assign_split binds the ratio as a default argument at
# import time, so the split cannot be changed after data_loader is imported.
PREDICT_CODE = r"""
import sys, numpy as np
model_dir, out = sys.argv[1], sys.argv[2]
from data_loader import build_dataset, prepare_data_splits
import grouped_split
grouped_split.install()                     # honours MLP_SPLIT_MODE
from features_v6 import build_params_v6
from anchor_features import anchors_for
from evaluate import load_ensemble, predict
from plotting import inverse_scale
models, weights, mean, scale, sensor_names, bank = load_ensemble(model_dir)
params, outputs, masks, meta, _ = build_dataset()
_, _, _, _, si, time_array = prepare_data_splits(params, outputs, masks, meta)
pv6 = build_params_v6(params, meta, bank)
anchors = anchors_for(params, bank)
pred = inverse_scale(predict(models, weights, pv6, anchors, time_array), mean, scale)
np.savez(out,
         chids=np.array([m["chid"] for m in meta]),
         cladding=np.array([m["cladding"] for m in meta]),
         hrr=np.array([m["hrr"] for m in meta]),
         pred=pred, actual=outputs, masks=masks,
         n_members=len(models),
         train_idx=si["train_idx"], valid_idx=si["valid_idx"], test_idx=si["test_idx"],
         sensor_names=np.array(sensor_names))
"""


def run_replicate(model_dir, split_name, split_mode):
    """Predict all sims with one ensemble under its own split. -> npz dict."""
    py = os.path.join(PROJECT_DIR, "venv", "Scripts", "python.exe")
    if not os.path.isfile(py):
        py = sys.executable
    tag = os.path.basename(model_dir)
    dump = os.path.join(OUTPUT_DIR, "comparison", f"_preds_{tag}.npz")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    env = dict(os.environ, MLP_SPLIT=split_name, MLP_SPLIT_MODE=split_mode)
    r = subprocess.run([py, "-c", PREDICT_CODE, model_dir, dump],
                       cwd=PROJECT_DIR, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"prediction failed for {model_dir} "
                         f"({split_name}, mode={split_mode})")
    return dict(np.load(dump, allow_pickle=True))


def pooled_metrics(actual, pred, mask, rows):
    """Frozen evaluate.py definitions: concatenate masked points, then score."""
    a = np.concatenate([actual[j][mask[j].astype(bool)].flatten() for j in rows])
    p = np.concatenate([pred[j][mask[j].astype(bool)].flatten() for j in rows])
    hot = a > 100
    return {
        "r2": r2_score(a, p),
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae": float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a[hot] - p[hot]) / a[hot])) * 100) if hot.any() else np.nan,
        "n": int(a.size),
    }


def per_sensor_r2(actual, pred, mask, rows):
    out = []
    for s in range(N_SENSORS):
        a = np.concatenate([actual[j][mask[j].astype(bool), s] for j in rows])
        p = np.concatenate([pred[j][mask[j].astype(bool), s] for j in rows])
        out.append(r2_score(a, p))
    return np.array(out)


def is_killer(cladding, hrr):
    return cladding == "Test_1_PE_PIR" and int(hrr) == 1333


def sibling_leakage(cladding, hrr, train_idx, test_idx):
    """How many held-out sims share a physical configuration with a training sim.

    A test case whose mesh siblings are in training is interpolation in a
    numerical parameter, not a blind prediction — this counts exactly that.
    """
    train_cfg = {(cladding[i], int(hrr[i])) for i in train_idx}
    hit = [j for j in test_idx if (cladding[j], int(hrr[j])) in train_cfg]
    return len(hit), len(test_idx)


def mstd(vals):
    """mean, sample std (0.0 for a single replicate — one draw has no spread)."""
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt_ms(m, s, dp=4):
    return f"{m:.{dp}f} ± {s:.{dp}f}" if s else f"{m:.{dp}f}"


def resolve(dirs, label):
    out = []
    for d in dirs:
        full = d if os.path.isabs(d) else os.path.join(PROJECT_DIR, d)
        if os.path.isfile(os.path.join(full, "best_model.pt")):
            out.append(full)
        else:
            print(f"  [skip] {label} {os.path.basename(full)}: no best_model.pt")
    return out


def load_side(dirs, split_name, split_mode):
    return [run_replicate(d, split_name, split_mode) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs-a", nargs="+", default=GROUPED_A)
    ap.add_argument("--dirs-b", nargs="+", default=GROUPED_B)
    ap.add_argument("--split-mode", default="grouped", choices=["grouped", "chid"],
                    help="split mode the headline comparison was trained under")
    ap.add_argument("--chid-dirs-a", nargs="+", default=CHID_A)
    ap.add_argument("--chid-dirs-b", nargs="+", default=CHID_B)
    ap.add_argument("--no-appendix", action="store_true",
                    help="skip the CHID-vs-grouped leakage appendix")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "comparison",
                                                  "split_comparison.md"))
    args = ap.parse_args()

    mode_label = ("configuration-grouped" if args.split_mode == "grouped"
                  else "legacy CHID-hash")
    print("=" * 78)
    print(f"  MLP surrogate — 70/15/15 vs 80/10/10 ({mode_label} split)")
    print("=" * 78)

    a_dirs = resolve(args.dirs_a, "70/15/15")
    b_dirs = resolve(args.dirs_b, "80/10/10")
    if not a_dirs or not b_dirs:
        print("\n  Nothing trained on one side — nothing to compare.")
        return 1

    A = load_side(a_dirs, "70_15_15", args.split_mode)
    B = load_side(b_dirs, "80_10_10", args.split_mode)

    chids = list(A[0]["chids"])
    for R in A + B:
        assert list(R["chids"]) == chids, "replicates saw different simulations"
        assert np.array_equal(R["actual"], A[0]["actual"]), "targets differ"
    actual, masks = A[0]["actual"], A[0]["masks"]
    cladding, hrr = A[0]["cladding"], A[0]["hrr"]

    # Arena derived from the split indices each side actually used — never
    # hardcoded, so this is correct for grouped and CHID splits alike.
    a_test, b_test = set(A[0]["test_idx"].tolist()), set(B[0]["test_idx"].tolist())
    a_valid, b_valid = set(A[0]["valid_idx"].tolist()), set(B[0]["valid_idx"].tolist())
    core = sorted(a_test & b_test)
    a_only = sorted(a_test - b_test)
    b_only = sorted(b_test - a_test)
    core_clean = [j for j in core if not is_killer(cladding[j], hrr[j])]

    if not core:
        print("\n  The two ratios share NO held-out simulation — no honest arena "
              "exists. Nothing comparable to report.")
        return 1

    print(f"\n  Replicates: 70/15/15 x{len(A)}, 80/10/10 x{len(B)}")
    print(f"  Splits: 70/15/15 = {len(A[0]['train_idx'])}/{len(A[0]['valid_idx'])}/"
          f"{len(A[0]['test_idx'])}   80/10/10 = {len(B[0]['train_idx'])}/"
          f"{len(B[0]['valid_idx'])}/{len(B[0]['test_idx'])}")
    print(f"  Common held-out core (test under BOTH ratios): {len(core)} sims")
    for j in core:
        print(f"    {chids[j]}")

    sides = [("70/15/15", a_dirs, A), ("80/10/10", b_dirs, B)]

    lines = []
    lines.append(f"# MLP surrogate — 70/15/15 vs 80/10/10 ({mode_label} split)\n")
    lines.append("Same 60 FDS simulations, same 39-feature physics-causal input vector, "
                 "same v9 training recipe (tau=1.8 robust tail loss, ambient clamp, "
                 "12 candidates → greedy ensemble), same frozen eval contract, same "
                 "parameter-matched MLP architecture. **One variable: the train/valid/"
                 "test ratio.**\n")
    for label, dirs, reps in sides:
        lines.append(f"- **{label}** — {len(reps)} replicate(s): "
                     + ", ".join(f"`{os.path.basename(d)}`" for d in dirs))
        lines.append(f"  - {len(reps[0]['train_idx'])} train / "
                     f"{len(reps[0]['valid_idx'])} valid / "
                     f"{len(reps[0]['test_idx'])} test; ensemble members kept: "
                     + ", ".join(str(int(r["n_members"])) for r in reps))
    lines.append("\nReplicates are identical reruns — same code, same split, same seeds. "
                 "Their spread comes from `cudnn.benchmark` non-determinism and IS the "
                 "retrain-variance band, measured rather than assumed.\n")

    lines.append("## Why the headline Test R² numbers cannot be compared directly\n")
    lines.append("The two ratios hold out different simulations, so each run's own "
                 "`overall_metrics.csv` scores a different set. Everything below is "
                 "scored on the intersection of the two held-out sets instead — "
                 "derived from the split indices each side actually used.\n")

    lines.append("### Common held-out core — test under BOTH ratios\n")
    lines.append(f"{len(core)} simulations, never seen in training or model selection "
                 "by either side:\n")
    for j in core:
        lines.append(f"- `{chids[j]}`"
                     f"{'  ← killer family' if is_killer(cladding[j], hrr[j]) else ''}")
    lines.append("")
    for only, label, other_valid, other in ((a_only, "70/15/15", b_valid, "80/10/10"),
                                            (b_only, "80/10/10", a_valid, "70/15/15")):
        if only:
            lines.append(f"Excluded — held out by {label} but used by {other} for "
                         f"{'validation (model selection)' if set(only) & other_valid else 'training'}"
                         f", so {other} is not blind to them:\n")
            for j in only:
                where = ("validation" if j in other_valid else "training")
                lines.append(f"- `{chids[j]}` — {other} {where}"
                             f"{'  ← killer family' if is_killer(cladding[j], hrr[j]) else ''}")
            lines.append("")

    arenas = [("Common core (all)", list(core))]
    if len(core_clean) != len(core) and core_clean:
        arenas.append(("Common core (excl. killer)", core_clean))

    METRICS = [("r2", "R²", 4), ("rmse", "RMSE (°C)", 2),
               ("mae", "MAE (°C)", 2), ("mape", "MAPE T>100 (%)", 2)]

    per_rep = {}
    for arena_name, rows in arenas:
        for label, dirs, reps in sides:
            per_rep[(label, arena_name)] = [
                pooled_metrics(actual, r["pred"], masks, rows) for r in reps]

    lines.append("## Per-replicate results on the common held-out core\n")
    for arena_name, rows in arenas:
        lines.append(f"\n### {arena_name} — {len(rows)} sims\n")
        lines.append("| side | replicate | R² | RMSE (°C) | MAE (°C) | MAPE T>100 (%) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for label, dirs, reps in sides:
            for d, m in zip(dirs, per_rep[(label, arena_name)]):
                lines.append(f"| {label} | `{os.path.basename(d)}` | {m['r2']:.4f} | "
                             f"{m['rmse']:.2f} | {m['mae']:.2f} | {m['mape']:.2f} |")

    lines.append("\n## Head-to-head — mean ± sample std over replicates\n")
    console = []
    for arena_name, rows in arenas:
        lines.append(f"\n### {arena_name} — {len(rows)} sims\n")
        lines.append("| metric | 70/15/15 | 80/10/10 | Δ of means (80−70) | verdict |")
        lines.append("|---|---:|---:|---:|---|")
        stats = {}
        for key, name, dp in METRICS:
            a_m, a_s = mstd([m[key] for m in per_rep[("70/15/15", arena_name)]])
            b_m, b_s = mstd([m[key] for m in per_rep[("80/10/10", arena_name)]])
            d = b_m - a_m
            if key == "r2":
                spread = max(a_s, b_s)
                if abs(d) < VARIANCE_BAND:
                    verdict = f"INCONCLUSIVE (\\|Δ\\| < {VARIANCE_BAND} band)"
                elif abs(d) < 2 * spread:
                    verdict = "INCONCLUSIVE (Δ within replicate spread)"
                else:
                    verdict = "80/10/10 better" if d > 0 else "70/15/15 better"
            else:
                verdict = "80/10/10 lower" if d < 0 else "70/15/15 lower"
            stats[key] = (a_m, a_s, b_m, b_s, d, verdict)
            lines.append(f"| {name} | {fmt_ms(a_m, a_s, dp)} | {fmt_ms(b_m, b_s, dp)} | "
                         f"{d:+.{dp}f} | {verdict} |")
        console.append((arena_name, len(rows), stats))

    lines.append("\n## Measured retrain variance (within-side spread on the core)\n")
    lines.append("| side | replicates | R² min | R² max | range | sample std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, dirs, reps in sides:
        r2s = [m["r2"] for m in per_rep[(label, "Common core (all)")]]
        _, s = mstd(r2s)
        lines.append(f"| {label} | {len(r2s)} | {min(r2s):.4f} | {max(r2s):.4f} | "
                     f"{max(r2s) - min(r2s):.4f} | {s:.4f} |")
    lines.append(f"\nThe ±{VARIANCE_BAND} rule is a standing project convention; the "
                 "range column is what this experiment actually measured.\n")

    lines.append("\n## Per-simulation R² on the common core (mean ± std over replicates)\n")
    lines.append("| simulation | 70/15/15 | 80/10/10 | Δ |")
    lines.append("|---|---:|---:|---:|")
    for j in core:
        mk = masks[j].astype(bool)
        vals = {}
        for label, dirs, reps in sides:
            vals[label] = mstd([r2_score(actual[j][mk].flatten(), r["pred"][j][mk].flatten())
                                for r in reps])
        d = vals["80/10/10"][0] - vals["70/15/15"][0]
        lines.append(f"| `{chids[j]}` | {fmt_ms(*vals['70/15/15'])} | "
                     f"{fmt_ms(*vals['80/10/10'])} | {d:+.4f} |")

    sens = {}
    for label, dirs, reps in sides:
        sens[label] = np.stack([per_sensor_r2(actual, r["pred"], masks, list(core))
                                for r in reps])
    sa, sb = sens["70/15/15"].mean(axis=0), sens["80/10/10"].mean(axis=0)
    lines.append("\n## Per-sensor R² on the common core (mean over replicates)\n")
    lines.append("| statistic | 70/15/15 | 80/10/10 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| mean per-sensor R² | {sa.mean():.4f} | {sb.mean():.4f} |")
    lines.append(f"| sensors R² > 0.9 | {int((sa > 0.9).sum())}/{N_SENSORS} | "
                 f"{int((sb > 0.9).sum())}/{N_SENSORS} |")
    lines.append(f"| sensors R² > 0.8 | {int((sa > 0.8).sum())}/{N_SENSORS} | "
                 f"{int((sb > 0.8).sum())}/{N_SENSORS} |")
    lines.append(f"| worst sensor R² | {sa.min():.4f} | {sb.min():.4f} |")

    pd.DataFrame({"sensor": list(A[0]["sensor_names"]),
                  "r2_70_15_15": sa, "r2_80_10_10": sb, "delta": sb - sa}).to_csv(
        os.path.join(os.path.dirname(args.out), "split_per_sensor_r2.csv"), index=False)

    # ── leakage appendix ────────────────────────────────────────────────
    appendix_rows = []
    if not args.no_appendix and args.split_mode == "grouped":
        ca = resolve(args.chid_dirs_a, "CHID 70/15/15")
        cb = resolve(args.chid_dirs_b, "CHID 80/10/10")
        if ca and cb:
            print("\n  Building leakage appendix from the CHID-split line…")
            CA = load_side(ca, "70_15_15", "chid")
            CB = load_side(cb, "80_10_10", "chid")
            lines.append("\n---\n")
            lines.append("## Appendix — what the legacy CHID split was measuring\n")
            lines.append("The CHID split hashes the CHID string. Because the three mesh "
                         "siblings of one physical configuration have different CHIDs, "
                         "they hash independently and scatter across train/valid/test. A "
                         "held-out case whose 0.09 m and 0.10 m twins sit in training is "
                         "interpolation in a numerical parameter, not a blind "
                         "prediction. The grouped split keys on the physical "
                         "configuration instead, so siblings always move together.\n")
            lines.append("Each line is scored on **its own** held-out set, as its own "
                         "contract defines — these are not the same simulations, which "
                         "is precisely the point.\n")
            lines.append("| split | ratio | reps | test sims | test sims with a mesh "
                         "sibling in train | test R² (own held-out set) |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for split_label, pairs in (
                    ("legacy CHID", (("70/15/15", CA), ("80/10/10", CB))),
                    ("grouped", (("70/15/15", A), ("80/10/10", B)))):
                for ratio, reps in pairs:
                    leak, ntest = sibling_leakage(cladding, hrr, reps[0]["train_idx"],
                                                  reps[0]["test_idx"])
                    r2s = [pooled_metrics(actual, r["pred"], masks,
                                          list(r["test_idx"]))["r2"] for r in reps]
                    m, s = mstd(r2s)
                    appendix_rows.append((split_label, ratio, leak, ntest, m, s))
                    lines.append(f"| {split_label} | {ratio} | {len(reps)} | {ntest} | "
                                 f"**{leak}/{ntest}** | {fmt_ms(m, s)} |")
            lines.append("")
            for ratio in ("70/15/15", "80/10/10"):
                c = next(r for r in appendix_rows if r[0] == "legacy CHID" and r[1] == ratio)
                g = next(r for r in appendix_rows if r[0] == "grouped" and r[1] == ratio)
                lines.append(f"- **{ratio}**: {c[4]:.4f} → {g[4]:.4f} on removing the "
                             f"sibling leakage — an inflation of **{c[4] - g[4]:+.4f} R²** "
                             f"attributable to mesh interpolation.")
            lines.append("\nThe same leakage exists in `bs8414_KAN_surrogate`, which "
                         "shares this pipeline. Any KAN-vs-MLP comparison must apply the "
                         "grouped split to BOTH projects before its numbers mean "
                         "anything.\n")
        else:
            lines.append("\n---\n")
            lines.append("## Appendix — leakage quantification\n")
            lines.append("Skipped: no CHID-split replicates available to compare against.\n")

    n_min = min(len(A), len(B))
    lines.append("\n---\n")
    lines.append(f"**Reading rule.** A delta inside ±{VARIANCE_BAND} R², or inside the "
                 f"measured replicate spread, is reported as inconclusive in either "
                 f"direction — never as a win. Physics sanity (`validate_physics.py`, "
                 f"per replicate) must pass for any accuracy claim to count.\n")
    if n_min < 3:
        lines.append(f"**Population size.** Only {len(A)} vs {len(B)} replicate(s); the "
                     f"KAN reference protocol uses 3 per family. Treat the spread column "
                     f"as provisional.\n")
    lines.append(f"**Arena size.** The common core is {len(core)} simulations "
                 f"({per_rep[('70/15/15', 'Common core (all)')][0]['n']:,} masked "
                 f"points). It is the largest set both sides are genuinely blind to.\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 78)
    for arena_name, n, stats in console:
        print(f"  {arena_name} ({n} sims)   70/15/15 x{len(A)} vs 80/10/10 x{len(B)}")
        print(f"    {'':<14}{'70/15/15':>20}{'80/10/10':>20}{'delta':>12}")
        for key, name, dp in METRICS:
            a_m, a_s, b_m, b_s, d, _ = stats[key]
            print(f"    {name:<14}{fmt_ms(a_m, a_s, dp):>20}"
                  f"{fmt_ms(b_m, b_s, dp):>20}{d:>+12.4f}")
        print(f"    verdict (R2): {stats['r2'][5]}")
    if appendix_rows:
        print("\n  Leakage appendix (each on its own held-out set):")
        for split_label, ratio, leak, ntest, m, s in appendix_rows:
            print(f"    {split_label:<12} {ratio}  siblings-in-train {leak}/{ntest}"
                  f"   test R2 {fmt_ms(m, s)}")
    print("=" * 78)
    print(f"\n  Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
