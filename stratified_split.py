"""System-stratified configuration-grouped split.

WHY THIS EXISTS
---------------
The standing 70/15/15 grouped split leaves one cladding system -- Test 7,
FR-PE with phenolic insulation -- in training only. It appears in neither the
validation nor the test set, so no held-out number in the paper says anything
about it, and one of four systems is effectively untested at the reported split.
That is a design defect independent of any score.

RULE S -- declared before training, and it consults no performance quantity
--------------------------------------------------------------------------
1. Order the 20 configurations by the project's existing md5 group hash
   (grouped_split._group_hash), the same ordering already used elsewhere.
2. Walking that order, assign a configuration to TEST if its cladding system is
   not yet represented in TEST, until TEST holds three configurations from three
   distinct systems.
3. Continue walking. Assign to VALID if its system is not yet represented in
   VALID, until VALID holds three configurations from three distinct systems,
   with the constraint that VALID must include the one system absent from TEST.
4. Everything remaining is TRAIN.

The rule is a function of the configuration identifiers and the pre-existing
hash only. It cannot be tuned toward a result, which is the point: the split is
chosen for coverage, and whatever accuracy it yields is reported as found.

Enable with MLP_SPLIT_MODE=stratified. Default is off, so every existing model
stays reproducible.
"""
import os

import data_loader
import grouped_split
from config import CLADDING_SYSTEMS, HRR_LEVELS

_ORIGINAL = data_loader.assign_split


def config_order():
    """The 20 configurations, in the project's canonical md5 hash order."""
    configs = [(c, str(h)) for c in sorted(CLADDING_SYSTEMS) for h in sorted(HRR_LEVELS)]
    return sorted(configs, key=lambda ch: grouped_split._group_hash(*ch))


def build():
    """Apply Rule S. Returns (test_set, valid_set) as sets of config keys."""
    order = config_order()
    all_systems = {c for c, _ in order}

    test, test_sys = [], set()
    for c, h in order:
        if len(test) == 3:
            break
        if c not in test_sys:
            test.append((c, h))
            test_sys.add(c)

    missing = all_systems - test_sys           # the system TEST cannot reach
    valid, valid_sys = [], set()
    # the absent system is placed first, so the constraint binds by construction
    for c, h in order:
        if (c, h) in test:
            continue
        if c in missing and c not in valid_sys:
            valid.append((c, h))
            valid_sys.add(c)
            break
    for c, h in order:
        if len(valid) == 3:
            break
        if (c, h) in test or (c, h) in valid:
            continue
        if c not in valid_sys:
            valid.append((c, h))
            valid_sys.add(c)

    return set(test), set(valid)


_TEST, _VALID = build()


def assign_split_stratified(chid, train_ratio=None, valid_ratio=None):
    key = grouped_split.config_key(chid)
    if key in _TEST:
        return "test"
    if key in _VALID:
        return "valid"
    return "train"


def install():
    if os.environ.get("MLP_SPLIT_MODE", "").lower() != "stratified":
        return False
    data_loader.assign_split = assign_split_stratified
    print("  [split] system-stratified grouped split active (Rule S)")
    return True


if __name__ == "__main__":
    SYS = {"Test_1_PE_PIR": "Test 1 PE", "Test_3_FRPE_PIR": "Test 3 FR-PE",
           "Test_5_LCM_PIR": "Test 5 A2", "Test_7_FRPE_Phenolic": "Test 7 FR-PE/phen"}
    t, v = build()
    order = config_order()
    print("\nRULE S -- system-stratified grouped split, declared before training\n")
    for label, grp in (("TEST", t), ("VALID", v)):
        print("  %-6s %d configurations, %d systems"
              % (label, len(grp), len({c for c, _ in grp})))
        for c, h in sorted(grp, key=lambda x: order.index(x)):
            print("      %-20s @ %s" % (SYS.get(c, c), h))
    tr = [x for x in order if x not in t and x not in v]
    print("  TRAIN  %d configurations, %d systems"
          % (len(tr), len({c for c, _ in tr})))
    covered = {c for c, _ in t} | {c for c, _ in v}
    print("\n  systems covered by valid + test: %d of 4  %s"
          % (len(covered), sorted(SYS.get(c, c) for c in covered)))
    print("  system absent from both         : %s"
          % (sorted(set(SYS) - covered) or "none"))
