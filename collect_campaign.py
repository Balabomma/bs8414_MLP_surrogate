"""Aggregate the review-response campaign into the numbers Sections 4.3, 6.1,
7.1 and 8.4 of the manuscript need.

Reads only the frozen-contract artefacts each run leaves behind
(`outputs/campaign/<run>/metrics/*.csv`) plus the existing grouped baseline
(`outputs/split_70_15_15_grouped{,_r2,_r3}/metrics/*.csv`). No GPU, no model
loading, so it is safe to run repeatedly while the campaign is still going —
missing runs are reported as pending rather than silently averaged away.

    python collect_campaign.py                 # everything available so far
    python collect_campaign.py --section loco  # one section
    python collect_campaign.py --out report.md

FOUR SECTIONS
-------------
  loco      C3  leave-one-configuration-out: the distribution that replaces a
                point estimate on three configurations
  scaler    C5  train-only output scaler: the delta IS the leakage the
                pre-split scaler was contributing
  seeds     C4  seed-varied band vs the seed-fixed band the paper reports
  ablation  M2  single-resolution run: separates the mesh-sibling floor from
                learnable error

READING RULE
------------
A delta smaller than the measured retrain band is reported as INCONCLUSIVE, in
either direction, per the project convention. The band used is the seed-varied
one once it exists, because the seed-fixed band is a lower bound (Section 7.1).
"""
import argparse
import csv
import os
import statistics as st
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "outputs")
CAMP = os.path.join(OUT, "campaign")

BASELINE = ["split_70_15_15_grouped", "split_70_15_15_grouped_r2",
            "split_70_15_15_grouped_r3"]
SCALER_RUN = "models_mlp_70_15_15_grouped_trainscaler"
SEED_RUNS = ["models_mlp_70_15_15_grouped_seed1337",
             "models_mlp_70_15_15_grouped_seed2024"]
ABLATION_RUN = "models_mlp_singleres_M009"
N_FOLDS = 20


# ---------------------------------------------------------------- readers ---
def _metrics_dir(name):
    """Campaign runs live under outputs/campaign/<name>/metrics; the pre-existing
    baseline replicates live under outputs/<name>/metrics."""
    for base in (CAMP, OUT):
        d = os.path.join(base, name, "metrics")
        if os.path.isdir(d):
            return d
    return None


def overall(name):
    d = _metrics_dir(name)
    if not d:
        return None
    p = os.path.join(d, "overall_metrics.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = {r["Set"]: r for r in csv.DictReader(f)}
    if "Test" not in rows:
        return None
    t = rows["Test"]
    return {k: float(t[k]) for k in ("r2", "rmse", "mae", "mape")} | {"n": int(t["n"])}


def per_channel(name):
    d = _metrics_dir(name)
    if not d:
        return None
    p = os.path.join(d, "per_sensor_metrics.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        vals = [float(r["r2"]) for r in csv.DictReader(f)]
    return st.mean(vals) if vals else None


def spread(vals):
    """mean and half-range — the dispersion convention used throughout the paper."""
    return st.mean(vals), (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0


def fmt(m, h, dp=4):
    return f"{m:.{dp}f} +/- {h:.{dp}f}" if h else f"{m:.{dp}f}"


# --------------------------------------------------------------- baseline ---
def baseline_stats(L):
    present = [b for b in BASELINE if overall(b)]
    if not present:
        L.append("_Baseline replicates not found; deltas cannot be computed._\n")
        return None
    r2 = [overall(b)["r2"] for b in present]
    ch = [per_channel(b) for b in present if per_channel(b) is not None]
    m, h = spread(r2)
    cm, chh = spread(ch) if ch else (float("nan"), 0.0)
    L.append(f"Grouped 70/15/15 baseline, {len(present)} replicate(s), "
             f"**seeds held fixed**:\n")
    L.append(f"- pooled test R2 **{fmt(m, h)}**  (raw: "
             + ", ".join(f"{v:.4f}" for v in r2) + ")")
    L.append(f"- per-channel test R2 **{fmt(cm, chh)}**\n")

    # Reading rule: once seed-varied runs exist they define the band. The
    # seed-fixed spread is a lower bound (manuscript Section 7.1) and must not
    # be used as the comparator for any claim.
    seed_vals = [o["r2"] for o in (overall(n) for n in SEED_RUNS) if o]
    band, kind = h, "seed-fixed, a lower bound"
    if seed_vals:
        _, band = spread([m] + seed_vals)
        kind = f"seed-varied over {len(seed_vals) + 1} seed bases"
        L.append(f"- **retrain band in use: +/-{band:.4f}** ({kind}); the "
                 f"seed-fixed spread of +/-{h:.4f} is a lower bound and is not "
                 f"used as a comparator\n")
    return {"r2": m, "band": band, "band_kind": kind, "seed_fixed_band": h,
            "chan": cm, "chan_band": chh, "n": len(present)}


# ------------------------------------------------------------------- loco ---
def section_loco(L, base):
    L.append("\n## C3 - Leave-one-configuration-out\n")
    try:
        import campaign_split
        order = campaign_split.config_order()
    except Exception as e:                                   # pragma: no cover
        order = None
        L.append(f"_(fold->configuration map unavailable: {e})_\n")

    rows, missing = [], []
    for f in range(N_FOLDS):
        name = f"models_mlp_loco_f{f:02d}"
        o = overall(name)
        if o is None:
            missing.append(f)
            continue
        cfg = f"{order[f][0]} HRR{order[f][1]}" if order else f"fold {f}"
        rows.append((f, cfg, o, per_channel(name)))

    if not rows:
        L.append(f"**Pending** - none of the {N_FOLDS} folds has completed yet.\n")
        return
    L.append(f"{len(rows)} of {N_FOLDS} folds complete"
             + (f"; pending: {', '.join(str(m) for m in missing)}" if missing else "")
             + ".\n")
    L.append("| fold | held-out configuration | sims | pooled R2 | per-chan R2 | RMSE (degC) | MAE (degC) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for f, cfg, o, c in rows:
        cs = f"{c:.3f}" if c is not None else "-"
        L.append(f"| {f} | {cfg} | {o['n']} | {o['r2']:.4f} | {cs} | "
                 f"{o['rmse']:.1f} | {o['mae']:.1f} |")

    r2 = sorted(o["r2"] for _, _, o, _ in rows)
    q1 = r2[len(r2) // 4] if len(r2) >= 4 else r2[0]
    q3 = r2[(3 * len(r2)) // 4] if len(r2) >= 4 else r2[-1]
    L.append(f"\n**Distribution of held-out pooled R2 over "
             f"{len(r2)} configurations**\n")
    L.append(f"- median **{st.median(r2):.4f}**, IQR {q1:.4f}-{q3:.4f}, "
             f"min {min(r2):.4f}, max {max(r2):.4f}, range **{max(r2)-min(r2):.4f}**")
    L.append(f"- mean {st.mean(r2):.4f}"
             + (f", sample SD {st.stdev(r2):.4f}" if len(r2) > 1 else ""))
    if base:
        L.append(f"- for comparison, the retrain band is +/-{base['band']:.4f} "
                 f"({base['band_kind']}); the configuration range above is "
                 f"**{(max(r2)-min(r2))/max(base['band'], 1e-9):.0f}x** that band")
    if missing:
        L.append(f"\n_Provisional: {len(missing)} fold(s) outstanding. Do not quote "
                 f"the distribution in the manuscript until all {N_FOLDS} have run._")
    L.append("\n**Caveat on the per-channel column.** Each fold's per-channel R2 is "
             "computed against the mean of that fold's own three simulations, a much "
             "narrower reference than the pooled test set, so it is systematically "
             "harsher and is *not* comparable with the 0.57 reported in Section 6.1. "
             "The pooled column is the like-for-like statistic.\n")


# ----------------------------------------------------------------- scaler ---
def section_scaler(L, base):
    L.append("\n## C5 - Train-only output scaler\n")
    o = overall(SCALER_RUN)
    if o is None:
        L.append("**Pending** - run has not completed.\n")
        return
    if not base:
        L.append("_No baseline to compare against._\n")
        return
    d = o["r2"] - base["r2"]
    dc = (per_channel(SCALER_RUN) or float("nan")) - base["chan"]
    L.append(f"| quantity | baseline (scaler on all 60) | train-only scaler | delta |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| pooled test R2 | {base['r2']:.4f} | {o['r2']:.4f} | **{d:+.4f}** |")
    L.append(f"| per-channel test R2 | {base['chan']:.4f} | "
             f"{per_channel(SCALER_RUN):.4f} | **{dc:+.4f}** |")
    L.append(f"| test RMSE (degC) | - | {o['rmse']:.1f} | |")
    verdict = ("INCONCLUSIVE (|delta| inside the retrain band)"
               if abs(d) < base["band"] else
               "OUTSIDE the retrain band - the pre-split scaler was contributing")
    L.append(f"\nRetrain band +/-{base['band']:.4f}. Verdict: **{verdict}**.\n")
    L.append("Either outcome is reportable. If inconclusive, Section 4.3 can state "
             "that the disclosed normalisation path was measured and found smaller "
             "than retrain noise; if outside, the corrected number replaces the "
             "headline and the disclosure becomes a correction.\n")


# ------------------------------------------------------------------ seeds ---
def section_seeds(L, base):
    L.append("\n## C4 - Seed-varied replicate band\n")
    got = [(n, overall(n)) for n in SEED_RUNS if overall(n)]
    if not got:
        L.append("**Pending** - neither seed-varied run has completed.\n")
        return
    if not base:
        L.append("_No baseline to compare against._\n")
        return
    vals = [base["r2"]] + [o["r2"] for _, o in got]
    m, h = spread(vals)
    L.append(f"Seed bases 42 (baseline mean), " +
             ", ".join(n.split("seed")[-1] for n, _ in got) + ".\n")
    L.append(f"- seed-**fixed** band (existing r1-r3): +/-{base['band']:.4f}")
    L.append(f"- seed-**varied** band ({len(vals)} distinct seed bases): "
             f"**+/-{h:.4f}**  (raw: " + ", ".join(f"{v:.4f}" for v in vals) + ")")
    if base["band"] > 0:
        L.append(f"- ratio: the seed-varied band is **{h/base['band']:.1f}x** the "
                 f"seed-fixed band")
    L.append("\nSection 7.1 currently reports the seed-fixed band and labels it a "
             "lower bound. Replace it with the seed-varied figure, and re-check "
             "every 'several times the band' comparison in Sections 6.1, 6.2 and 6.6 "
             "against the new value.\n")
    if len(got) < len(SEED_RUNS):
        L.append(f"_Provisional: {len(SEED_RUNS)-len(got)} seed run(s) outstanding._\n")


# --------------------------------------------------------------- ablation ---
def section_ablation(L, base):
    L.append("\n## M2 - Single-resolution ablation\n")
    o = overall(ABLATION_RUN)
    if o is None:
        L.append("**Pending** - run has not completed.\n")
        return
    c = per_channel(ABLATION_RUN)
    L.append(f"Trained and tested within the 0.09 m subset only: no mesh siblings "
             f"anywhere in the corpus, {o['n']} masked test points.\n")
    L.append("| quantity | multi-resolution baseline | single-resolution |")
    L.append("|---|---:|---:|")
    if base:
        L.append(f"| pooled test R2 | {base['r2']:.4f} | {o['r2']:.4f} |")
        L.append(f"| per-channel test R2 | {base['chan']:.4f} | "
                 + (f"{c:.4f} |" if c is not None else "- |"))
    L.append(f"| test RMSE (degC) | - | {o['rmse']:.1f} |")
    L.append(f"| test MAE (degC) | - | {o['mae']:.1f} |")
    L.append("\n**How to read this.** The two test sets are not the same simulations "
             "- three configurations at one mesh here, three at three meshes in the "
             "baseline - so the R2 values are not directly comparable and no delta is "
             "quoted. What *is* comparable is RMSE against the 72.1 degC mesh-sibling "
             "floor of Section 6.1: this run has no siblings, so its RMSE contains no "
             "irreducible mesh component. An RMSE materially below 129.1 degC supports "
             "the reading that roughly half the baseline error is that floor; an RMSE "
             "near 129.1 degC says the floor is not what limits the model.\n")


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["loco", "scaler", "seeds", "ablation", "all"],
                    default="all")
    ap.add_argument("--out", default=os.path.join(CAMP, "CAMPAIGN_RESULTS.md"))
    a = ap.parse_args()

    if not os.path.isdir(CAMP):
        print(f"no campaign output at {CAMP} - has run_review_campaign.ps1 started?")
        return 1

    L = ["# Review-response campaign - collected results\n",
         "Generated by `collect_campaign.py` from the frozen-contract metrics of "
         "each run. Sections marked **Pending** have not finished; nothing is "
         "averaged over an incomplete set.\n",
         "\n## Baseline\n"]
    base = baseline_stats(L)

    if a.section in ("all", "scaler"):
        section_scaler(L, base)
    if a.section in ("all", "ablation"):
        section_ablation(L, base)
    if a.section in ("all", "seeds"):
        section_seeds(L, base)
    if a.section in ("all", "loco"):
        section_loco(L, base)

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print(text)
    print(f"written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
