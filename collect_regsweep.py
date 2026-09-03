"""Aggregate the regularisation sweep against the stated targets.

Targets: R2 >= 0.85 AND MAPE < 20 % on train, valid and test simultaneously.

Selection is on VALIDATION. The test column is printed because it is the
quantity of interest, but the arm is chosen by validation score and the test
figure for the chosen arm is read once. Picking the best test column across
seven arms would be a seven-attempt fit to eight held-out simulations.
"""
import csv
import os

PROJ = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(PROJ, "outputs", "regsweep")

ARMS = [
    ("reg_A0_baseline", "defaults, 50-run corpus"),
    ("reg_A1_drop30", "dropout 0.30"),
    ("reg_A2_drop45", "dropout 0.45"),
    ("reg_A3_wd1e3", "weight decay 1e-3"),
    ("reg_A4_hidden64", "hidden 64"),
    ("reg_A5_drop30_wd1e3", "dropout 0.30 + wd 1e-3"),
    ("reg_A6_h64_drop30", "hidden 64 + dropout 0.30"),
]
R2_TARGET, MAPE_TARGET = 0.85, 20.0


def read(arm):
    p = os.path.join(SWEEP, arm, "overall_metrics.csv")
    if not os.path.exists(p):
        return None
    return {r["Set"]: r for r in csv.DictReader(open(p))}


def main():
    print("\n" + "=" * 88)
    print("  REGULARISATION SWEEP -- targets: R2 >= 0.85 and MAPE < 20 % on all three sets")
    print("=" * 88)
    print("  %-26s%17s%17s%17s%7s" % ("arm", "train R2/MAPE", "valid R2/MAPE",
                                      "test R2/MAPE", "meets"))
    print("  " + "-" * 86)

    done, pending = [], []
    for arm, note in ARMS:
        m = read(arm)
        if not m:
            pending.append(arm)
            continue
        cells, ok = [], True
        for k in ("Train", "Valid", "Test"):
            r = m.get(k)
            if not r:
                cells.append("      --      ")
                ok = False
                continue
            r2, mp = float(r["r2"]), float(r["mape"])
            ok &= (r2 >= R2_TARGET and mp < MAPE_TARGET)
            cells.append("%7.3f /%6.1f%%" % (r2, mp))
        print("  %-26s%17s%17s%17s%7s" % (note[:26], cells[0], cells[1], cells[2],
                                          "YES" if ok else "no"))
        done.append((arm, note, m, ok))

    if not done:
        print("\n  no arms complete yet")
        return 1

    print("  " + "-" * 86)
    best = max(done, key=lambda d: float(d[2]["Valid"]["r2"]))
    arm, note, m, ok = best
    print("\n  Selected on VALIDATION: %s (%s)" % (arm, note))
    print("    valid R2 = %.4f   MAPE = %.2f %%"
          % (float(m["Valid"]["r2"]), float(m["Valid"]["mape"])))
    print("  Its test score, read once:")
    print("    test  R2 = %.4f   MAPE = %.2f %%   RMSE = %.1f degC"
          % (float(m["Test"]["r2"]), float(m["Test"]["mape"]),
             float(m["Test"]["rmse"])))

    t_r2, t_mp = float(m["Test"]["r2"]), float(m["Test"]["mape"])
    print("\n  Against the targets on the test set:")
    print("    R2   %.3f vs >= 0.85   -> %s (shortfall %+.3f)"
          % (t_r2, "MET" if t_r2 >= R2_TARGET else "NOT MET", t_r2 - R2_TARGET))
    print("    MAPE %.1f %% vs < 20 %%    -> %s (excess %+.1f pp)"
          % (t_mp, "MET" if t_mp < MAPE_TARGET else "NOT MET", t_mp - MAPE_TARGET))

    n_all = sum(1 for _, _, _, o in done if o)
    print("\n  arms meeting all six conditions: %d of %d complete" % (n_all, len(done)))
    if pending:
        print("  PROVISIONAL -- %d arm(s) outstanding: %s"
              % (len(pending), ", ".join(pending)))
    print("\n  Reading. If no arm clears the test targets, the honest conclusion is")
    print("  that regularisation does not close a gap this large, and the binding")
    print("  constraint is the 20 distinct physical configurations the corpus holds")
    print("  -- which is what the leave-one-configuration-out spread already says")
    print("  (per-configuration pooled R2 ranges 0.171 to 0.978). Reverting to")
    print("  identifier-level splitting would show test R2 ~ 0.88, but that is the")
    print("  mesh-sibling leakage this project withdrew, not an improvement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
