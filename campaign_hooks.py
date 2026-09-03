"""Two dataset hooks for the review-response campaign, both env-guarded.

Each is a no-op unless its environment variable is set, so a patched `train.py`
reproduces the current behaviour byte-for-byte when the campaign variables are
absent. This matters: the existing replicates must stay comparable.

  MLP_SCALER_SCOPE=train    refit the per-sensor output scaler on TRAINING
                            simulations only  (review item C5)
  MLP_ABLATION_MESH=M009    keep only one mesh resolution, so the surrogate
                            never sees mesh siblings  (review item M2)

WHY A train.py HOOK AND NOT A MONKEY-PATCH
------------------------------------------
`train.py` does `from data_loader import build_dataset, prepare_data_splits`,
which binds both function objects at import time. Rebinding
`data_loader.prepare_data_splits` afterwards therefore has no effect on
train.py — unlike `data_loader.assign_split`, which `prepare_data_splits` looks
up from module globals at call time and which `grouped_split.py` legitimately
patches. There is no import-time hook that can reach these two behaviours, so
they are applied explicitly at the call site instead.

`data_loader.py` and the other parity-locked files are not touched.
"""
import os
import re

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import N_SENSORS


def _mesh_of(chid):
    m = re.search(r"M(\d{3})", chid)
    if m:
        return "M" + m.group(1)
    m = re.search(r"_0_(\d+)$", chid)
    if m:
        return {"08": "M008", "09": "M009", "1": "M010", "10": "M010"}.get(m.group(1))
    return None


def filter_mesh(params, outputs, masks, meta):
    """M2 — restrict the corpus to a single resolution. Call immediately after
    build_dataset(), before prepare_data_splits().

    Also applies the defective-run exclusion (MLP_DROP_DEFECTIVE), which must
    happen here rather than later: the dropped simulations would otherwise enter
    the anchor bank and the output scaler before the split is formed."""
    import defective_runs
    keep = defective_runs.keep_index(meta)
    if keep is not None:
        n0 = len(meta)
        keep = np.array(keep)
        params, outputs, masks = params[keep], outputs[keep], masks[keep]
        meta = [meta[i] for i in keep]
        print(f"\n  [corpus] excluding {n0 - len(meta)} simulations with a "
              f"non-combusting cladding core; {len(meta)} of {n0} retained")

    mode = os.environ.get("MLP_CAMPAIGN", "").lower()
    if mode.startswith("leakctrl"):
        # prepare_data_splits routes every label that is not train/valid into
        # test, so simulations the design excludes must leave the dataset here.
        import data_loader
        keep = [i for i, m in enumerate(meta)
                if data_loader.assign_split(m["chid"]) != "drop"]
        keep = np.array(keep)
        print(f"  [campaign] leakage control: {len(keep)} of {len(meta)} "
              f"simulations retained ({len(meta) - len(keep)} dropped by design)")
        return (params[keep], outputs[keep], masks[keep], [meta[i] for i in keep])

    mesh = os.environ.get("MLP_ABLATION_MESH")
    if not mesh:
        return params, outputs, masks, meta
    keep = [i for i, m in enumerate(meta) if _mesh_of(m["chid"]) == mesh]
    if not keep:
        raise ValueError(f"MLP_ABLATION_MESH={mesh} matched no simulations")
    keep = np.array(keep)
    print(f"\n  [campaign] single-resolution ablation: keeping {mesh} only "
          f"({len(keep)} of {len(meta)} simulations)")
    return (params[keep], outputs[keep], masks[keep], [meta[i] for i in keep])


def refit_scaler(outputs, split_info, scaler):
    """C5 — refit the output scaler on training simulations only. Call
    immediately after prepare_data_splits(); train.py re-derives every scaled
    array and the ambient clamp from this object, so replacing it is sufficient."""
    if os.environ.get("MLP_SCALER_SCOPE", "all").lower() != "train":
        return scaler
    tr = split_info["train_idx"]
    new = StandardScaler()
    new.fit(outputs[tr].reshape(-1, N_SENSORS))
    d_mean = float(np.abs(new.mean_ - scaler.mean_).max())
    d_scale = float(np.abs(new.scale_ - scaler.scale_).max())
    print(f"  [campaign] output scaler refitted on {len(tr)} training sims only; "
          f"max |Δmean| = {d_mean:.3f} °C, max |Δscale| = {d_scale:.3f} °C")
    return new
