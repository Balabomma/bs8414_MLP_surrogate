"""Aggregate the leakage-control experiment.

Each run trains once and is scored twice on matched sets -- 4 simulations,
all at 0.10 m, one per cladding system -- differing only in whether the scored
case has two mesh siblings in training. Training and validation are identical
between arms by construction, so the paired difference is the leakage effect
with nothing else moving.

    leaked - clean > 0   siblings in training help  (leakage inflates scores)
    leaked - clean ~ 0   no detectable leakage benefit
"""
import csv
import os
import statistics as st

PROJ = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(PROJ, "outputs", "campaign")

RUNS = [
    ("models_mlp_leakctrl", "seed 42, as designed"),
    ("models_mlp_leakctrl_seed1337", "seed 1337"),
    ("models_mlp_leakctrl_seed2024", "seed 2024"),
    ("models_mlp_leakctrl_swapped", "seed 42, arms swapped"),
    ("models_mlp_leakctrl_swap1337", "seed 1337, arms swapped"),
    ("models_mlp_leakctrl_swap2024", "seed 2024, arms swapped"),
]


def read(name, arm):
    p = os.path.join(CAMP, name, f"metrics_{arm}.csv")
    if not os.path.exists(p):
        return None
    rows = {r["Set"]: r for r in csv.DictReader(open(p))}
    t = rows.get("Test")
    return {k: float(t[k]) for k in ("r2", "rmse", "mae")} | {"n": int(t["n"])} if t else None


def main():
    print("\n" + "=" * 74)
    print("  LEAKAGE CONTROL -- does having mesh siblings in training help?")
    print("=" * 74)
    print("  training held at 38 simulations / 14 configurations in BOTH arms;")
    print("  scored sets matched on size (4), mesh (0.10 m) and system (one each)\n")
    print(f"  {'run':<34}{'clean R2':>10}{'leaked R2':>11}{'delta':>9}{'dRMSE':>9}")
    print("  " + "-" * 70)

    deltas, pending = [], []
    for name, label in RUNS:
        c, l = read(name, "clean"), read(name, "leaked")
        if not c or not l:
            pending.append(label)
            continue
        d = l["r2"] - c["r2"]
        deltas.append(d)
        print(f"  {label:<34}{c['r2']:>10.4f}{l['r2']:>11.4f}{d:>+9.4f}"
              f"{l['rmse'] - c['rmse']:>+9.1f}")

    if not deltas:
        print("\n  no completed runs yet")
        return 1
    print("  " + "-" * 70)
    m = st.mean(deltas)
    print(f"  {'mean paired delta (leaked - clean)':<34}{'':>10}{'':>11}{m:>+9.4f}")
    if len(deltas) > 1:
        sd = st.stdev(deltas)
        se = sd / len(deltas) ** 0.5
        print(f"  {'SD / SE over runs':<34}{'':>10}{'':>11}{sd:>9.4f}{se:>9.4f}")
        print(f"\n  interval on the mean delta: {m:+.4f} +/- {2.0 * se:.4f} (approx 2 SE)")

    print()
    if pending:
        print(f"  PROVISIONAL -- {len(pending)} run(s) outstanding: "
              + ", ".join(pending))
    print("\n  Reading. A positive delta means siblings in training inflate the")
    print("  score, i.e. the leakage mechanism the manuscript attributes the")
    print("  0.16-0.22 split-construction effect to. A delta at or below zero")
    print("  means that mechanism is not reproduced once the number of distinct")
    print("  training configurations is held fixed, and the split-construction")
    print("  effect must be attributed to training-set diversity instead.")
    print("\n  Power note: 4 simulations per arm against a configuration-level SD")
    print("  of about 0.20 (Section 6.1). Report the interval, not the point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
