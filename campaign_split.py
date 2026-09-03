"""Split patches for the review-response campaign: leave-one-configuration-out
and single-resolution ablation.

WHY THIS WORKS WITHOUT TOUCHING data_loader.py
----------------------------------------------
`data_loader.prepare_data_splits` resolves `assign_split` from module globals at
CALL time, so rebinding `data_loader.assign_split` changes its behaviour. This is
the same hook `grouped_split.py` uses and it is the only hook that works — note
that patching `prepare_data_splits` itself does NOT work, because `train.py` does
`from data_loader import prepare_data_splits` and binds the function object at
import time.

`data_loader.py`, `anchor_features.py`, `features_v4.py` and `features_v6.py`
stay byte-identical on disk, so `verify_parity.py` still passes.

MODES (selected by env var MLP_CAMPAIGN)
---------------------------------------
  loco        leave-one-configuration-out. MLP_LOCO_FOLD=0..19 selects which of
              the 20 physical configurations is held out as test. Validation is
              the next 3 configurations in the same deterministic hash order,
              wrapping around; the remaining 16 are training.
              -> 3 test sims / 9 valid sims / 48 train sims per fold.

Anything else (or unset) leaves the configuration-grouped split untouched.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
The single-resolution ablation cannot be expressed through this hook.
`prepare_data_splits` routes every label that is not "train" or "valid" into
`test_idx`, so there is no discard bucket: simulations we want excluded would
silently become test cases. Filtering by mesh therefore happens in the
`train.py` hook described in `PATCH_train_py.md`, not here.

USAGE
-----
    import grouped_split;  grouped_split.install()
    import campaign_split; campaign_split.install()   # after grouped_split

`install()` is a no-op unless MLP_CAMPAIGN is set, so importing it
unconditionally is safe.
"""
import os
import re

import data_loader
import grouped_split
from config import CLADDING_SYSTEMS, HRR_LEVELS

_ORIGINAL = data_loader.assign_split

# Deterministic configuration order: the same md5 ordering grouped_split uses,
# so fold k means the same configuration on every machine and every rerun.
def config_order():
    claddings = sorted(CLADDING_SYSTEMS)
    configs = [(c, str(h)) for c in claddings for h in sorted(HRR_LEVELS)]
    return sorted(configs, key=lambda ch: grouped_split._group_hash(*ch))


def _mesh_of(chid):
    """Recover the mesh token from a CHID. Handles both naming conventions."""
    m = re.search(r"M(\d{3})", chid)
    if m:
        return "M" + m.group(1)
    m = re.search(r"_0_(\d+)$", chid)          # e.g. ..._0_08, ..._0_1
    if m:
        tok = m.group(1)
        return {"08": "M008", "09": "M009", "1": "M010", "10": "M010"}.get(tok)
    return None


def make_loco(fold):
    order = config_order()
    n = len(order)
    if not 0 <= fold < n:
        raise ValueError(f"MLP_LOCO_FOLD must be 0..{n-1}, got {fold}")
    test = {order[fold]}
    valid = {order[(fold + i) % n] for i in range(1, 4)}

    def assign(chid, train_ratio=None, valid_ratio=None):
        key = grouped_split.config_key(chid)
        if key in test:
            return "test"
        if key in valid:
            return "valid"
        return "train"
    return assign, test, valid


def install():
    mode = os.environ.get("MLP_CAMPAIGN", "").lower()
    if not mode:
        return False

    if mode == "loco":
        fold = int(os.environ.get("MLP_LOCO_FOLD", "0"))
        assign, test, valid = make_loco(fold)
        data_loader.assign_split = assign
        print(f"  [campaign] LOCO fold {fold}: test={sorted(test)} "
              f"valid={sorted(valid)}")
        return True

    if mode in ("leakctrl_clean", "leakctrl_leaked"):
        arm = mode.split("_")[1]
        assign, leak, clean, valid = make_leakctrl(arm)
        data_loader.assign_split = assign
        print(f"  [campaign] leakage control, arm={arm}: train/valid identical "
              f"in both arms; scored set = 4 x {_LEAK_MESH}, one per system")
        return True

    if mode == "single_res":
        raise ValueError(
            "single_res is handled by the train.py hook (mesh filter), not by "
            "this module — see PATCH_train_py.md. Do not set MLP_CAMPAIGN for it.")

    raise ValueError(f"unknown MLP_CAMPAIGN mode {mode!r}")


def uninstall():
    data_loader.assign_split = _ORIGINAL


if __name__ == "__main__":
    # Self-check: print the fold table without training anything.
    order = config_order()
    print(f"{len(order)} configurations in deterministic fold order:\n")
    for i, c in enumerate(order):
        print(f"  fold {i:2d}  test = {c[0]:<22} HRR{c[1]}")


# ---------------------------------------------------------------------------
# Leakage control (review item: DA CRITICAL 1)
#
# The 0.16-0.22 figure of Section 6.1 compares an identifier-level split against
# a configuration-grouped one, but those two protocols differ in MORE than
# leakage: the grouped split also cuts the number of distinct training
# configurations from ~20 to 14. Training-set diversity and test-side
# near-duplication move together, so the reported magnitude is confounded.
#
# This mode holds training and validation EXACTLY constant and varies only
# whether the scored cases have mesh siblings in training:
#
#   train : 10 configs x 3 siblings + 4 "leak" configs x 2 siblings  = 38 sims
#           (14 distinct configurations in BOTH arms)
#   valid : 2 configs x 3 siblings                                    =  6 sims
#   test  : leakctrl_leaked -> the 4 held-back M010 siblings of the leak configs
#           leakctrl_clean  -> the M010 sibling of 4 entirely unseen configs
#
# Both scored sets are 4 simulations, all at 0.10 m, one per cladding system.
# One training serves both arms; the model is simply scored twice.
# ---------------------------------------------------------------------------
_LEAK_MESH = "M010"


def _leak_design():
    """Pick 4 leak-test, 4 clean-test and 2 valid configurations, balancing
    cladding systems so the two scored sets are compositionally matched."""
    order = config_order()
    by_sys = {}
    for c in order:
        by_sys.setdefault(c[0], []).append(c)
    systems = sorted(by_sys)
    # MLP_LEAKCTRL_SWAP=1 exchanges the leak-test and clean-test configurations,
    # so the result cannot be an artefact of which draw landed in which arm.
    swap = os.environ.get("MLP_LEAKCTRL_SWAP", "0") == "1"
    a, b = (1, 0) if swap else (0, 1)
    leak, clean = [], []
    for s in systems:                      # one per system, deterministic
        leak.append(by_sys[s][a])
        clean.append(by_sys[s][b])
    rest = [c for c in order if c not in leak and c not in clean]
    valid, train_full = rest[:2], rest[2:]
    return set(leak), set(clean), set(valid), set(train_full)


def make_leakctrl(arm):
    leak, clean, valid, train_full = _leak_design()

    def assign(chid, train_ratio=None, valid_ratio=None):
        key = grouped_split.config_key(chid)
        mesh = _mesh_of(chid)
        if key in valid:
            return "valid"
        if key in train_full:
            return "train"
        if key in leak:
            # two siblings train; the third is the leaked scored case
            if mesh != _LEAK_MESH:
                return "train"
            return "test" if arm == "leaked" else "drop"
        if key in clean:
            return "test" if (arm == "clean" and mesh == _LEAK_MESH) else "drop"
        return "drop"
    return assign, leak, clean, valid
