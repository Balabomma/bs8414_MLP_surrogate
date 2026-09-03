"""Loader for the Part1 variant corpus: 185 configurations, one mesh, one source.

WHY THIS EXISTS
---------------
The DCLG corpus this project was built on holds 60 runs but only 20 distinct
physical configurations, because it varies the prescribed source strength and the
cell size while holding the wall build-up fixed. Every diagnostic in the study
points at that as the binding constraint: leave-one-configuration-out accuracy
spans 0.171-0.978, regularisation moves held-out accuracy by less than the
retrain band, and the one intervention that helped changed which configurations
were trained on.

The Part1 corpus removes that constraint. It varies the axes a specifier actually
controls:

    cladding core   9 levels   ACM_A2 ACM_PE AL BRK CDR HPL OSB PLY DCLG
    insulation      5 levels   MW MWBC PF PIR WC
    cavity          3 binary   noair (no ventilated air gap)
                               nogap (no cavity gap)
                               nocb  (no cavity barrier)

41 (core, insulation) stems are realised at 2-7 of the 8 possible cavity states,
giving 185 distinct configurations with one simulation each.

Three properties follow, and each removes a problem the DCLG corpus created:

  single mesh (0.1 m)       there are no mesh siblings, so the configuration-
                            grouped partitioning correction is unnecessary here
                            and mesh-sibling leakage cannot occur
  single source (2333.3     every run sits at the BS 8414-1 prescribed value, so
  kW/m2, the prescribed     every BR 135 outcome is regulatorily meaningful --
  value)                    unlike the DCLG sweep, 48 of whose 60 runs were at
                            off-standard source strengths carrying no such status
  one run per configuration each simulation IS a configuration, so a split is
                            leakage-free by construction

Targets are the 16 external thermocouples -- Level 1 (8) and Level 2 (8). The
insulation-embedded Level 2 array is not recorded in most of these runs, and the
BR 135 *external* criterion is assessed on External_LV2 alone.

Design vector: params[:, 0] is the core index (consumed by the model's embedding,
matching the DCLG loader's convention); the remaining columns are continuous.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

PART1_ROOT = r"D:/Bs8414_05052026/Part1"
N_TIMESTEPS = 181
EXT_PREFIXES = ("External_LV1", "External_LV2")
FLAGS = ("noair", "nogap", "nocb")

CORES = ["ACM_A2", "ACM_PE", "AL", "BRK", "CDR", "DCLG", "HPL", "OSB", "PLY"]
INSUL = ["MW", "MWBC", "PF", "PIR", "WC"]

HRRPUA = 2333.3      # kW/m2, identical in every deck
CELL_SIZE = 0.10     # m, identical in every deck
HRR_MW_NORM = 0.875  # 3.5 MW crib peak / HRR_MW_SCALE (=4.0)
HRR_COL = 12         # index of the source column in the design vector


def _parse(chid):
    """(core, insulation, flags) from a CHID. Returns None if unrecognised."""
    s = chid.replace("BS8414_", "")
    flags = tuple(f for f in FLAGS if f in s)
    for f in FLAGS:
        s = s.replace("_" + f, "")
    # DCLG replications carry their own naming
    if s.startswith("DCLG"):
        return "DCLG", _dclg_insulation(s), flags
    core = next((c for c in sorted(CORES, key=len, reverse=True)
                 if s.startswith(c)), None)
    if core is None:
        return None
    rest = s[len(core):].lstrip("_")
    ins = next((i for i in sorted(INSUL, key=len, reverse=True)
                if rest.startswith(i)), None)
    if ins is None:
        return None
    return core, ins, flags


def _dclg_insulation(s):
    """DCLG replications: Test1/Test3 use PIR, Test7 phenolic."""
    if "Test7" in s:
        return "PF"
    return "PIR"


def _load_devc(path):
    """(181, 16) external-channel array in degC, and the channel names."""
    df = pd.read_csv(path, skiprows=1)
    cols = [c.strip() for c in df.columns]
    keep = [i for i, c in enumerate(cols) if c.startswith(EXT_PREFIXES)]
    if len(keep) != 16 or df.shape[0] != N_TIMESTEPS:
        return None, None
    return df.iloc[:, keep].values.astype(np.float32), [cols[i] for i in keep]


def build_dataset(root=PART1_ROOT, verbose=True, include_dclg=False):
    """(params, outputs, masks, meta, sensor_names) matching the DCLG loader shape."""
    files = sorted(glob.glob(os.path.join(root, "**", "*_devc.csv"), recursive=True))
    params, outputs, meta, names = [], [], [], None
    skipped = {"shape": 0, "unparsed": 0, "dclg": 0}

    for f in files:
        chid = os.path.basename(f)[:-9]
        arr, cols = _load_devc(f)
        if arr is None:
            skipped["shape"] += 1
            continue
        p = _parse(chid)
        if p is None:
            skipped["unparsed"] += 1
            continue
        core, ins, flags = p
        # The DCLG entries are replications of the four physical tests, not
        # members of the variant matrix: they share a build-up convention with
        # each other rather than with the ACM/AL/BRK/... family, and lumping
        # three different cladding systems under one "DCLG" label would put
        # duplicate design vectors on distinct simulations. Excluded by default
        # and available separately as the experimental-anchor subset.
        if core == "DCLG" and not include_dclg:
            skipped["dclg"] = skipped.get("dclg", 0) + 1
            continue
        names = names or cols

        row = [float(CORES.index(core))]                       # embedding index
        row += [1.0 if ins == k else 0.0 for k in INSUL]        # insulation one-hot
        row += [1.0 if f_ in flags else 0.0 for f_ in FLAGS]    # cavity flags
        # Structural summaries: a ventilated cavity exists only if neither the
        # gap nor the air path is removed; barrier present unless nocb.
        row += [0.0 if ("nogap" in flags or "noair" in flags) else 1.0,
                0.0 if "nocb" in flags else 1.0,
                float(len(flags)) / len(FLAGS)]
        # Final column: normalised source strength, consumed by the model's
        # plume channels (q, Q, q^2/3). Constant across this corpus by design --
        # every run is at the BS 8414-1 prescribed value -- so it carries no
        # information between configurations and acts only as a temporal basis.
        row += [HRR_MW_NORM]
        params.append(row)
        outputs.append(arr)
        meta.append({"chid": chid, "core": core, "insulation": ins,
                     "flags": flags, "stem": "%s_%s" % (core, ins),
                     "config": "%s_%s|%s" % (core, ins, "+".join(flags) or "base"),
                     "hrrpua": HRRPUA, "cell_size": CELL_SIZE})

    params = np.asarray(params, dtype=np.float32)
    outputs = np.asarray(outputs, dtype=np.float32)
    masks = np.ones((len(outputs), N_TIMESTEPS), dtype=np.float32)

    if verbose:
        print("  [part1] %d configurations loaded (skipped: %d wrong shape, "
              "%d unparsed, %d DCLG replications)"
              % (len(meta), skipped["shape"], skipped["unparsed"], skipped["dclg"]))
        print("  [part1] design vector %d columns; targets %s"
              % (params.shape[1], outputs.shape[1:]))
        print("  [part1] single mesh %.2f m, single source %.1f kW/m2"
              % (CELL_SIZE, HRRPUA))
    return params, outputs, masks, meta, names


if __name__ == "__main__":
    import collections
    p, o, m, meta, names = build_dataset()
    print("\n  params", p.shape, " outputs", o.shape)
    print("  cores      :", dict(collections.Counter(x["core"] for x in meta)))
    print("  insulation :", dict(collections.Counter(x["insulation"] for x in meta)))
    print("  flag states:", dict(collections.Counter(
        "+".join(x["flags"]) or "base" for x in meta)))
    print("  stems      : %d ; distinct configs: %d"
          % (len({x["stem"] for x in meta}), len({x["config"] for x in meta})))
    print("  peak temp  : %.1f degC   min %.1f" % (np.nanmax(o), np.nanmin(o)))
    print("  channels   :", names[:3], "...", names[-2:])
