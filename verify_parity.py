"""Prove this project sees EXACTLY the same data/split/features as the KAN project.

The MLP result is only a valid ablation if the data pipeline is identical, so this
does not assume it — it checks it:

  1. File-level: SHA-256 of the four transplanted modules (data_loader,
     anchor_features, features_v4, features_v6) vs the KAN originals.
  2. Pipeline-level: runs the KAN project's own pipeline in a subprocess using the
     KAN venv, dumps (CHIDs, split indices, 39-feature input matrix, anchors,
     scaler mean/scale) and compares against this project's arrays element-wise.

Exit code 0 = parity proven. Any mismatch is printed and exits 1.

Usage: python verify_parity.py
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

from config import PROJECT_DIR, KAN_PROJECT_DIR, N_SENSORS, SPLIT_NAME
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6
from anchor_features import build_bank, anchors_for

SHARED_MODULES = ["data_loader.py", "anchor_features.py",
                  "features_v4.py", "features_v6.py"]

DUMP_CODE = r"""
import sys, numpy as np
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6
from anchor_features import build_bank, anchors_for
out = sys.argv[1]
params, outputs, masks, meta, sensor_names = build_dataset()
_, _, _, scaler, si, time_array = prepare_data_splits(params, outputs, masks, meta)
bank = build_bank(params, outputs, si["train_idx"])
pv6 = build_params_v6(params, meta, bank)
np.savez(out,
         chids=np.array([m["chid"] for m in meta]),
         train_idx=si["train_idx"], valid_idx=si["valid_idx"], test_idx=si["test_idx"],
         params_v6=pv6, outputs=outputs, masks=masks,
         valid_anchors=anchors_for(params[si["valid_idx"]], bank),
         test_anchors=anchors_for(params[si["test_idx"]], bank),
         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
         sensor_names=np.array(sensor_names))
"""


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ok = True

    # Parity is defined against the KAN project's hard-coded 70/15/15 split. The
    # KAN subprocess below inherits this process's environment but reads its own
    # config, so running under MLP_SPLIT=80_10_10 would report a split mismatch
    # that is an artefact of the harness, not a real divergence. Refuse instead.
    if SPLIT_NAME != "70_15_15":
        print(f"  MLP_SPLIT={SPLIT_NAME} — parity is only defined against the KAN "
              f"project's 70/15/15 split.\n  Unset MLP_SPLIT and re-run.")
        return 2

    print("=" * 70)
    print("  Data-pipeline parity: bs8414_MLP_surrogate vs bs8414_KAN_surrogate")
    print("=" * 70)
    print("\n[1] Transplanted modules — SHA-256 vs KAN originals")
    for name in SHARED_MODULES:
        a = sha(os.path.join(PROJECT_DIR, name))
        b = sha(os.path.join(KAN_PROJECT_DIR, name))
        same = a == b
        ok &= same
        print(f"  {name:<22} {'IDENTICAL' if same else 'DIFFERS'}  {a[:16]}")

    print("\n[2] Running the KAN project's own pipeline (its venv, its cwd)…")
    kan_py = os.path.join(KAN_PROJECT_DIR, "venv", "Scripts", "python.exe")
    if not os.path.isfile(kan_py):
        kan_py = sys.executable
    dump = os.path.join(tempfile.gettempdir(), "kan_pipeline_dump.npz")
    r = subprocess.run([kan_py, "-c", DUMP_CODE, dump], cwd=KAN_PROJECT_DIR,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        print("  FAILED to run the KAN pipeline")
        return 1
    ref = np.load(dump, allow_pickle=True)

    params, outputs, masks, meta, sensor_names = build_dataset()
    _, _, _, scaler, si, time_array = prepare_data_splits(params, outputs, masks, meta)
    bank = build_bank(params, outputs, si["train_idx"])
    pv6 = build_params_v6(params, meta, bank)

    checks = [
        ("simulation CHIDs", np.array([m["chid"] for m in meta]), ref["chids"]),
        ("sensor names", np.array(sensor_names), ref["sensor_names"]),
        ("train indices", si["train_idx"], ref["train_idx"]),
        ("valid indices", si["valid_idx"], ref["valid_idx"]),
        ("test indices", si["test_idx"], ref["test_idx"]),
        ("input matrix (39 feat)", pv6, ref["params_v6"]),
        ("targets (degC)", outputs, ref["outputs"]),
        ("masks", masks, ref["masks"]),
        ("valid anchors", anchors_for(params[si["valid_idx"]], bank), ref["valid_anchors"]),
        ("test anchors", anchors_for(params[si["test_idx"]], bank), ref["test_anchors"]),
        ("scaler mean", scaler.mean_, ref["scaler_mean"]),
        ("scaler scale", scaler.scale_, ref["scaler_scale"]),
    ]

    print("\n[3] Element-wise comparison")
    for label, mine, theirs in checks:
        if mine.shape != theirs.shape:
            same = False
            detail = f"shape {mine.shape} vs {theirs.shape}"
        elif mine.dtype.kind in "US":
            same = bool((mine == theirs).all())
            detail = "exact"
        else:
            same = bool(np.array_equal(mine, theirs))
            detail = "exact" if same else f"max|diff|={np.abs(mine - theirs).max():.3e}"
        ok &= same
        print(f"  {label:<24} {'MATCH' if same else 'MISMATCH':<9} {detail}")

    n = len(ref["chids"])
    print(f"\n  Dataset: {n} sims "
          f"({len(si['train_idx'])}/{len(si['valid_idx'])}/{len(si['test_idx'])}), "
          f"{N_SENSORS} sensors")
    print("\n" + ("  PARITY PROVEN — the two projects train on identical data."
                  if ok else "  PARITY BROKEN — do not compare results."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
