"""Physics-sanity validation for an MLP ensemble.

Identical checks and thresholds to `bs8414_KAN_surrogate/validate_physics_v6.py`
so KAN and MLP physics verdicts are directly comparable:
  1. Sub-ambient excursions   (T < 17 degC must not occur — the clamp enforces 18)
  2. Growth-limb monotonicity (drops > 2 degC on the rising limb; FLAG above 2%)
  3. Energy/HRR correlation   (per-sim mean T vs HRR level; PASS if r > 0.2)
  4. Steady-state plateau     (late-time mean |dT/dt| < 0.5 degC/s)

Usage: python validate_physics.py --model-dir models_mlp
"""
import argparse
import os

import numpy as np

from config import N_SENSORS, N_TIMESTEPS, T_END, PROJECT_DIR, SPLIT_LABEL
from data_loader import build_dataset, prepare_data_splits
import grouped_split          # configuration-grouped split (mesh-sibling leakage fix)
grouped_split.install()
from features_v6 import build_params_v6
from anchor_features import anchors_for
from evaluate import load_ensemble, predict
from plotting import inverse_scale

T_AMB = 18.0


def growth_violations(pred, mask):
    """Count rising-limb steps where T drops more than 2 degC."""
    viol = 0
    tot = 0
    for i in range(pred.shape[0]):
        valid_t = np.where(mask[i].astype(bool))[0]
        if len(valid_t) < 5:
            continue
        for s in range(N_SENSORS):
            tr = pred[i, valid_t, s]
            peak = int(np.argmax(tr))
            if peak < 2:
                continue
            d = np.diff(tr[:peak + 1])
            viol += int(np.sum(d < -2.0))
            tot += len(d)
    return viol, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_mlp")
    args = ap.parse_args()
    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(PROJECT_DIR, model_dir)

    models, weights, mean, scale, sensor_names, bank = load_ensemble(model_dir)
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, split_info, time_array = prepare_data_splits(params, outputs, masks, meta)
    params_v6 = build_params_v6(params, meta, bank)

    print("=" * 68)
    print(f"  Physics-sanity validation  ({model_dir}, split {SPLIT_LABEL})")
    print("=" * 68)

    for set_name in ("Valid", "Test"):
        idx = split_info[f"{set_name.lower()}_idx"]
        anchors = anchors_for(params[idx], bank)
        pred = inverse_scale(predict(models, weights, params_v6[idx], anchors,
                                     time_array), mean, scale)
        mk = masks[idx]
        m3 = mk.astype(bool)[..., None] * np.ones((1, 1, N_SENSORS), bool)

        masked_pred = np.where(m3, pred, np.nan)
        tmin = np.nanmin(masked_pred)
        n_sub = int(np.nansum(masked_pred < T_AMB - 1.0))
        n_tot = int(m3.sum())

        viol, gtot = growth_violations(pred, mk)

        hrr = np.array([meta[i]["hrr"] for i in idx], dtype=float)
        mean_T = np.array([np.nanmean(np.where(m3[j], pred[j], np.nan))
                           for j in range(len(idx))])
        corr = np.corrcoef(hrr, mean_T)[0, 1] if len(idx) > 2 else float("nan")

        late = []
        for j in range(len(idx)):
            vt = np.where(mk[j].astype(bool))[0]
            if len(vt) < 10:
                continue
            for s in range(N_SENSORS):
                tail = pred[j, vt[int(0.8 * len(vt)):], s]
                if len(tail) > 1:
                    late.append(np.mean(np.abs(np.diff(tail))) / (T_END / N_TIMESTEPS))
        plateau = float(np.mean(late)) if late else float("nan")

        print(f"\n[{set_name}]  ({len(idx)} sims)")
        print(f"  1. Sub-ambient (<17degC):  {n_sub}/{n_tot} pts "
              f"({100*n_sub/n_tot:.3f}%)  min={tmin:.1f}degC  "
              f"-> {'PASS' if n_sub == 0 else 'FLAG'}")
        print(f"  2. Growth-limb drops >2degC: {viol}/{gtot} steps "
              f"({100*viol/max(gtot,1):.2f}%)  "
              f"-> {'PASS' if viol/max(gtot,1) < 0.02 else 'FLAG'}")
        print(f"  3. HRR-vs-meanT corr:      r={corr:+.3f}  "
              f"-> {'PASS' if corr > 0.2 else 'FLAG (weak/neg)'}")
        print(f"  4. Late-time |dT/dt|:      {plateau:.3f} degC/s (mean over tail) "
              f"-> {'PASS' if plateau < 0.5 else 'CHECK'}")


if __name__ == "__main__":
    main()
