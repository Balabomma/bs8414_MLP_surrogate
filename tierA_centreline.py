"""Recompute Tier A on centreline-conforming thermocouple positions.

WHY
---
The four DCLG validation decks sample the external thermocouples 0.208 m off the
combustion chamber centreline. The corrected deck (BS8414_DCLG_Test1_adv, dated
after the others) places the middle main-wall channel at x = 1.822 m, which is
exactly the centreline of the 2.000 m opening spanning x = [0.822, 2.822], and
moves its diagnostic slice to PBX = 1.822 to match. The 169-configuration
training corpus uses that same convention throughout.

Tier A as first computed therefore compared measurements taken on the centreline
against simulations sampled to one side of it. Rather than re-run four cases,
temperatures are resampled from slice output that already exists:

    PBY = 1.4   vertical plane at the thermocouple standoff -> 10 main-wall channels
    PBX = 0.6   vertical plane through the wing            ->  6 wing channels

Both planes are declared in every validation deck.

CAVEAT, to be stated in the paper
---------------------------------
Slice data is on the 0.10 m solution grid, so resampling lands on the nearest
cell rather than the exact coordinate -- a residual displacement of up to 0.05 m,
introduced to remove one of 0.208 m. Both conventions are extracted from the same
slices, so the CONTRAST between them is not affected by that residual.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import fdsreader

ROOT = r"D:/Bs8414_05052026/Part1"
STANDOFF_Y = 1.4
WING_X = 0.6
Z_LV1, Z_LV2 = 4.51, 7.01

# main-wall x positions under each convention
X_CENTRELINE = [0.922, 1.372, 1.822, 2.272, 2.722]     # symmetric about 1.822
X_DISPLACED = [0.7143, 1.1643, 1.6143, 2.0643, 2.5143]  # as-run in Tests 3/5/7
Y_WING = [0.37, 0.87, 1.37]

# test: (measured L2 external peak degC, truncation time s, chid)
CASES = {
    "Test 1": (813.9, 395.0, "BS8414_DCLG_Test1_adv"),
    "Test 3": (877.0, 1402.0, "BS8414_DCLG_Test3"),
    "Test 5": (565.0, 1800.0, "BS8414_DCLG_Test5"),
    "Test 7": (939.0, 1694.0, "BS8414_DCLG_Test7"),
}


def _plane(sim, orientation, position, tol=0.06):
    """Return (data, coords) for the slice with the requested orientation/position."""
    best = None
    for sl in sim.slices:
        if sl.orientation != orientation:
            continue
        e = sl.extent
        pos = (e.x_start if orientation == 1 else e.y_start)
        if abs(pos - position) < tol:
            best = sl
            break
    if best is None:
        return None, None
    data, coords = best.to_global(return_coordinates=True, masked=False)
    return np.asarray(data), {k: np.asarray(v) for k, v in coords.items()}


def _at(data, coords, a_name, a_val, z_val):
    """Nearest-cell time series at (a_val, z_val) on a vertical plane."""
    a = coords[a_name]
    z = coords["z"]
    ia = int(np.argmin(np.abs(a - a_val)))
    iz = int(np.argmin(np.abs(z - z_val)))
    return data[:, ia, iz], float(a[ia]), float(z[iz])


def lv2_external(sim, xs):
    """(times, array of the 8 external Level 2 channels) for a given x convention."""
    main, cm = _plane(sim, 2, STANDOFF_Y)
    wing, cw = _plane(sim, 1, WING_X)
    if main is None or wing is None:
        return None, None, None
    series, used = [], []
    for x in xs:
        s, ax, az = _at(main, cm, "x", x, Z_LV2)
        series.append(s)
        used.append(("main", x, ax, az))
    for y in Y_WING:
        s, ay, az = _at(wing, cw, "y", y, Z_LV2)
        series.append(s)
        used.append(("wing", y, ay, az))
    sl = [s for s in sim.slices if s.orientation == 2][0]
    return np.asarray(sl.times), np.asarray(series), used


def main():
    print("\n" + "=" * 78)
    print("  TIER A ON CENTRELINE-CONFORMING POSITIONS (resampled from slice output)")
    print("=" * 78)
    rows = []
    for label, (meas, tend, chid) in CASES.items():
        smv = sorted(glob.glob(os.path.join(ROOT, "**", chid + ".smv"), recursive=True))
        if not smv:
            print("  %-8s .smv not found" % label)
            continue
        sim = fdsreader.Simulation(os.path.dirname(smv[0]))
        out = {}
        for tag, xs in (("centreline", X_CENTRELINE), ("displaced", X_DISPLACED)):
            t, arr, used = lv2_external(sim, xs)
            if t is None:
                out[tag] = None
                continue
            win = t <= tend
            out[tag] = float(np.nanmax(arr[:, win])) if win.any() else float("nan")
        rows.append((label, meas, out.get("centreline"), out.get("displaced"),
                     tend, len(t) if t is not None else 0, float(t[-1]) if t is not None else 0))
        print("  %-8s measured %7.1f | centreline %7s | displaced %7s | t_end %.0f s (%d frames to %.0f s)"
              % (label, meas,
                 ("%.1f" % out["centreline"]) if out.get("centreline") else "-",
                 ("%.1f" % out["displaced"]) if out.get("displaced") else "-",
                 tend, rows[-1][5], rows[-1][6]))

    print("\n" + "-" * 78)
    for tag, idx in (("CENTRELINE (standard-conforming)", 2), ("DISPLACED (as originally run)", 3)):
        ok = [r for r in rows if r[idx] and np.isfinite(r[idx])]
        if len(ok) < 2:
            print("\n  %s: insufficient complete cases (%d)" % (tag, len(ok)))
            continue
        ratio = np.array([r[idx] / r[1] for r in ok])
        lg = np.log(ratio)
        e = np.array([r[1] for r in ok])
        s = np.array([r[idx] for r in ok])
        print("\n  %s   n = %d tests" % (tag, len(ok)))
        print("    ratios      : %s" % ", ".join("%.3f" % v for v in ratio))
        print("    delta       : %.3f" % float(np.exp(lg.mean())))
        print("    sigma~      : %.3f" % float(lg.std(ddof=1)))
        print("    Pearson r   : %.3f" % float(np.corrcoef(e, s)[0, 1]))
        print("    SE(log d)   : %.3f" % float(lg.std(ddof=1) / np.sqrt(len(ok))))
    both = [r for r in rows if r[2] and r[3] and np.isfinite(r[2]) and np.isfinite(r[3])]
    if both:
        d = np.array([abs(r[2] - r[3]) for r in both])
        rel = np.array([abs(r[2] - r[3]) / r[2] * 100 for r in both])
        print("\n  SENSITIVITY TO THE 0.208 m OFFSET, on identical slice data:")
        print("    mean |difference| = %.1f degC (%.1f %%), max %.1f degC"
              % (d.mean(), rel.mean(), d.max()))


if __name__ == "__main__":
    main()
