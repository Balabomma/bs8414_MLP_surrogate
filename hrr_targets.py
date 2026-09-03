"""Total heat release rate trajectories, used to supervise the mediating variable.

The surrogate's worst errors sit at the top of the source-strength sweep, where
FDS accelerates by up to a factor of ten between 2100 and 2333 kW m-2 because the
cladding core ignites. That is a regime change, and a smooth function of the
PRESCRIBED source cannot represent it. The causal variant inserts the physical
mediator on the path

    prescribed source  ->  cladding involvement  ->  total HRR  ->  temperature

and this module supplies the ground truth for the third node: Q_total(t) from the
FDS _hrr.csv output, resampled onto the 181-step, 10 s device grid.

Coverage is partial -- 48 of the 60 runs have an _hrr.csv on disk -- so the
auxiliary loss is masked to the runs that have one. The 12 without still train
normally on temperature; they simply contribute nothing to the HRR term.

Units are MW throughout, matching the model's internal hrr_mw scaling.
"""
import glob
import os

import numpy as np

from config import N_TIMESTEPS, T_END

SEARCH_ROOTS = [r"D:/New_BS8414_simulations_08-03-2026"]
try:
    from config import SIMS_DIR
    SEARCH_ROOTS.append(SIMS_DIR)
except Exception:
    pass

_CACHE = {}


def _find(chid):
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", chid + "_hrr.csv"), recursive=True)
        if hits:
            return hits[0]
    return None


def _read_mw(path):
    """Return (time_s, HRR_MW) from an FDS _hrr.csv."""
    import csv
    rows = list(csv.reader(open(path)))
    header = [c.strip() for c in rows[1]]
    ti, hi = header.index("Time"), header.index("HRR")
    t, q = [], []
    for r in rows[2:]:
        if len(r) > max(ti, hi):
            try:
                t.append(float(r[ti]))
                q.append(float(r[hi]))
            except ValueError:
                continue
    return np.asarray(t), np.asarray(q) / 1000.0     # kW -> MW


def build(meta):
    """(N, N_TIMESTEPS) total HRR in MW, and an (N,) availability mask."""
    grid = np.linspace(0.0, T_END, N_TIMESTEPS)
    out = np.zeros((len(meta), N_TIMESTEPS), dtype=np.float32)
    have = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        chid = m["chid"]
        if chid not in _CACHE:
            p = _find(chid)
            _CACHE[chid] = _read_mw(p) if p else None
        rec = _CACHE[chid]
        if rec is None:
            continue
        t, q = rec
        if len(t) < 2:
            continue
        out[i] = np.interp(grid, t, q, left=q[0], right=q[-1]).astype(np.float32)
        have[i] = 1.0
    n = int(have.sum())
    print("  [hrr] total-HRR supervision available for %d of %d simulations "
          "(peak %.1f MW, mean %.1f MW)"
          % (n, len(meta), out[have > 0].max() if n else 0.0,
             out[have > 0].mean() if n else 0.0))
    return out, have
