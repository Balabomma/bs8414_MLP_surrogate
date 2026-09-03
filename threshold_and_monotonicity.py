"""Threshold/timing metrics (manuscript Sec 6.4) and monotonicity probe (Sec 7.4).

Sec 6.4 — BR 135 is a threshold crossing sustained for 30 s inside a 15-minute
window whose start is set by a Level 1 trigger. R2 does not measure any of that,
so the regulatory-relevant errors are computed directly, per held-out case:

    start time      first time DG1_1003 exceeds T_start + 200 K for >= 30 s
    exceedance time first time DG1_1029 exceeds T_start + 600 K for >= 30 s
    duration        total time DG1_1029 spends above the 600 K criterion
    peak            max of DG1_1029

Sec 7.4 — the pooled heat-release/temperature check is ill-posed on the
configuration-grouped buckets (each cladding system appears at a single source
level), so the surrogate's response to its principal continuous input is probed
directly: hold cladding and mesh fixed, sweep HRRPUA over the five corpus
levels, and require the predicted mean temperature to be non-decreasing.

Usage:  python threshold_and_monotonicity.py --model-dir models_mlp_70_15_15_grouped
"""
import argparse
import os
import re

import numpy as np
import torch

from config import (N_TIMESTEPS, N_SENSORS, T_END, HRR_LEVELS,
                    CLADDING_SYSTEMS, MESH_SIZES, SPLIT_LABEL)
from data_loader import build_dataset, prepare_data_splits
import grouped_split
grouped_split.install()
from features_v6 import build_params_v6
from anchor_features import anchors_for
from evaluate import load_ensemble, predict, inverse_scale

T_AMBIENT = 18.0
TRIGGER_RISE = 200.0        # BR 135 start-time trigger, K above start temp
CRITERION_RISE = 600.0      # BR 135 external Level 2 criterion, K above start
SUSTAIN_S = 30.0            # both require a 30 s sustained period
WINDOW_S = 15 * 60.0        # 15-minute assessment window
DT = T_END / (N_TIMESTEPS - 1)
SUSTAIN_STEPS = max(1, int(round(SUSTAIN_S / DT)))

CH_TRIGGER = "External_LV1_main02(1003)"   # Level 1 - sets the start time
CH_CRITERION = "External_LV2_main02(1029)"  # Level 2 - carries pass/fail



def _mesh_of(chid):
    """Canonical mesh token for a CHID. Exclusive by construction.

    Two naming conventions coexist in the corpus:
        DCLG_Test_1_PE_PIR_HRR1333_M008_0_1   -> explicit M0xx token
        DCLG_Test_5_LCM_PIR_08 / _0_09 / _0_1 -> cell size encoded in the suffix
    The explicit token wins where present; only then is the suffix consulted.
    """
    m = re.search(r"M(\d{3})", chid)
    if m:
        return "M" + m.group(1)
    m = re.search(r"_0?_?(08|09|1|10)$", chid)
    if m:
        return {"08": "M008", "09": "M009", "1": "M010", "10": "M010"}[m.group(1)]
    return None


def _sustained_cross(series, thresh, n_steps):
    """First index at which `series` is above `thresh` for n_steps consecutively."""
    above = series > thresh
    if n_steps <= 1:
        idx = np.flatnonzero(above)
        return int(idx[0]) if idx.size else None
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= n_steps:
            return int(i - n_steps + 1)
    return None


def _time_above(series, thresh):
    return float(np.count_nonzero(series > thresh) * DT)


def threshold_metrics(curve_trig, curve_crit):
    """BR 135-relevant quantities for one simulation."""
    t_start_temp = curve_trig[0]
    i_start = _sustained_cross(curve_trig, t_start_temp + TRIGGER_RISE, SUSTAIN_STEPS)
    crit_abs = curve_crit[0] + CRITERION_RISE
    i_exc = _sustained_cross(curve_crit, crit_abs, SUSTAIN_STEPS)
    return {
        "start_s": None if i_start is None else i_start * DT,
        "exceed_s": None if i_exc is None else i_exc * DT,
        "duration_s": _time_above(curve_crit, crit_abs),
        "peak_C": float(curve_crit.max()),
        "peak_rise_K": float(curve_crit.max() - curve_crit[0]),
        "in_window": (i_start is not None and i_exc is not None
                      and (i_exc - i_start) * DT <= WINDOW_S),
    }


def _fmt(v, unit=""):
    return "  n/a" if v is None else f"{v:6.0f}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    models, weights, mean, scale, names, bank = load_ensemble(args.model_dir)
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, split_info, time_array = prepare_data_splits(params, outputs, masks, meta)
    params_v6 = build_params_v6(params, meta, bank)

    i_trig = names.index(CH_TRIGGER)
    i_crit = names.index(CH_CRITERION)
    test_idx = split_info["test_idx"]

    print("=" * 78)
    print(f"  Sec 6.4  BR 135 threshold and timing errors - {args.model_dir}")
    print(f"  split {SPLIT_LABEL} (configuration-grouped), {len(test_idx)} held-out sims")
    print(f"  trigger  {CH_TRIGGER}  (+{TRIGGER_RISE:.0f} K, {SUSTAIN_S:.0f} s sustained)")
    print(f"  criterion {CH_CRITERION} (+{CRITERION_RISE:.0f} K, {SUSTAIN_S:.0f} s sustained)")
    print("=" * 78)

    anchors = anchors_for(params[test_idx], bank)
    pred = inverse_scale(predict(models, weights, params_v6[test_idx], anchors, time_array),
                         mean, scale)

    hdr = (f"{'simulation':<44}{'peak_rise K':>13}{'d_peak':>9}"
           f"{'d_start':>9}{'d_exceed':>10}{'d_dur':>9}")
    print(hdr)
    print("-" * len(hdr))
    dpk, dst, dex, ddu = [], [], [], []
    for k, i in enumerate(test_idx):
        act = threshold_metrics(outputs[i][:, i_trig], outputs[i][:, i_crit])
        prd = threshold_metrics(pred[k][:, i_trig], pred[k][:, i_crit])
        d_peak = prd["peak_C"] - act["peak_C"]
        d_start = (None if None in (act["start_s"], prd["start_s"])
                   else prd["start_s"] - act["start_s"])
        d_exc = (None if None in (act["exceed_s"], prd["exceed_s"])
                 else prd["exceed_s"] - act["exceed_s"])
        d_dur = prd["duration_s"] - act["duration_s"]
        dpk.append(d_peak); ddu.append(d_dur)
        if d_start is not None: dst.append(d_start)
        if d_exc is not None: dex.append(d_exc)
        print(f"{meta[i]['chid'][:43]:<44}{act['peak_rise_K']:>13.0f}"
              f"{d_peak:>9.1f}{_fmt(d_start,'s')}{_fmt(d_exc,'s')}{d_dur:>8.0f}s")

    print("-" * len(hdr))
    def _s(name, v, unit):
        if not v:
            print(f"  {name:<28} n/a (criterion never reached)"); return
        a = np.array(v, dtype=float)
        print(f"  {name:<28} mean {a.mean():+8.1f}{unit}   "
              f"MAE {np.abs(a).mean():7.1f}{unit}   max|.| {np.abs(a).max():7.1f}{unit}")
    _s("peak error", dpk, " C")
    _s("start-time error", dst, " s")
    _s("exceedance-time error", dex, " s")
    _s("duration error", ddu, " s")

    # margin of every corpus run against the criterion
    rises = np.array([outputs[i][:, i_crit].max() - outputs[i][0, i_crit]
                      for i in range(len(meta))])
    near = np.abs(rises - CRITERION_RISE) <= 50.0
    print(f"\n  Corpus margin: {int(near.sum())}/{len(rises)} runs have peak Level 2 rise "
          f"within +/-50 K of the {CRITERION_RISE:.0f} K criterion")
    print(f"  peak rise range across corpus: {rises.min():.0f} - {rises.max():.0f} K; "
          f"{int((rises > CRITERION_RISE).sum())}/{len(rises)} exceed it")

    # ---------------- Sec 7.4 monotonicity probe ----------------
    print()
    print("=" * 78)
    print("  Sec 7.4  Monotonicity probe - predicted mean T vs source strength")
    print("  cladding and mesh held fixed; HRRPUA swept over the five corpus levels")
    print("=" * 78)

    hrr_sorted = sorted(HRR_LEVELS)
    lo, hi = float(min(hrr_sorted)), float(max(hrr_sorted))
    by_key = {}
    for i, m in enumerate(meta):
        by_key[grouped_split.config_key(m["chid"]) + (m["chid"],)] = i
    # index one representative run per (cladding, hrr) to source the raw param row
    rep = {}
    for i, m in enumerate(meta):
        ck = grouped_split.config_key(m["chid"])
        rep.setdefault(ck, []).append(i)

    total_viol = 0
    for clad in sorted(CLADDING_SYSTEMS):
        for mesh_key in sorted(MESH_SIZES):
            rows, labels = [], []
            for h in hrr_sorted:
                # Mesh identification must be exclusive. The earlier test combined a
                # substring check with suffix checks, but every swept run is named
                # ..._M0xx_0_1 -- the trailing _0_1 is part of the base convention, not
                # a mesh tag -- so an M008 run also satisfied endswith("_0_1") and was
                # collected as an M010 candidate. cand[0] then returned the wrong mesh,
                # duplicating the 0.08 m sweep into the 0.10 m row at every swept source
                # level. Resolve the mesh once, canonically, instead.
                cand = [i for i in rep.get((clad, str(h)), [])
                        if _mesh_of(meta[i]["chid"]) == mesh_key]
                if not cand:
                    continue
                rows.append(cand[0]); labels.append(h)
            if len(rows) < 3:
                continue
            idx = np.array(rows)
            a = anchors_for(params[idx], bank)
            pr = inverse_scale(predict(models, weights, params_v6[idx], a, time_array),
                               mean, scale)
            means = np.array([pr[k].mean() for k in range(len(idx))])
            truth = np.array([outputs[i].mean() for i in idx])
            diffs = np.diff(means)
            viol = int((diffs < 0).sum())
            total_viol += viol
            tag = "OK " if viol == 0 else f"VIOL x{viol}"
            print(f"  {clad:<22}{mesh_key}  pred " +
                  " ".join(f"{v:6.1f}" for v in means) +
                  f"   [{tag}]")
            print(f"  {'':<22}{'':4}  FDS  " +
                  " ".join(f"{v:6.1f}" for v in truth))
    print("-" * 78)
    print(f"  monotonicity violations across all cladding x mesh sweeps: {total_viol}")


if __name__ == "__main__":
    main()
