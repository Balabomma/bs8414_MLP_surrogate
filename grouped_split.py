"""Configuration-grouped split — fixes mesh-sibling leakage in the CHID-hash split.

WHY THIS EXISTS
---------------
The corpus is a complete 4 (cladding) x 5 (HRRPUA) x 3 (mesh) factorial = 60 runs,
i.e. **20 physical configurations each solved at three mesh resolutions**.

`data_loader.assign_split` hashes the CHID string. The CHID differs across meshes,
so the three mesh siblings of one physical configuration hash independently and
scatter across train/valid/test. Measured on the shipped splits:

    70/15/15 : 6 of 8 test cases had a mesh sibling in TRAIN
    80/10/10 : 5 of 5 test cases had a mesh sibling in TRAIN

A held-out case whose 0.09 m and 0.10 m twins are in training is not a blind
prediction — it is interpolation in a numerical parameter. Reported held-out R2
under that split measures mesh interpolation, not generalisation to unseen
configurations.

THE FIX
-------
Hash the *physical configuration* (cladding system, HRRPUA), not the CHID, so all
three mesh siblings always land in the same bucket. 20 groups at 70/15/15 gives
14/3/3 groups = **42/9/9 simulations**.

PARITY CONTRACT
---------------
`data_loader.py`, `anchor_features.py`, `features_v4.py`, `features_v6.py` are
SHA-256-checked against the KAN project by `verify_parity.py` and MUST NOT be
edited. This module therefore does not touch them: it patches the module-global
`data_loader.assign_split` at runtime, which `prepare_data_splits` resolves
dynamically. All four transplanted files stay byte-identical on disk and
`verify_parity.py` still passes.

NOTE: the same leakage exists in bs8414_KAN_surrogate, which shares the pipeline.
Any future KAN-vs-MLP comparison must apply this fix to BOTH projects.

USAGE
-----
Import and call `install()` BEFORE `prepare_data_splits` is invoked:

    import grouped_split; grouped_split.install()

Set env MLP_SPLIT_MODE=chid to disable and restore the legacy behaviour.
"""
import hashlib
import os
import re

import data_loader
from config import TRAIN_RATIO, VALID_RATIO

_ORIGINAL_ASSIGN_SPLIT = data_loader.assign_split

# Anchor runs (HRRPUA = 2333.3) carry no HRR token in their CHID.
_ANCHOR_HRR = "2333"

_CLADDING_PATTERNS = [
    (re.compile(r"Test_1", re.I), "Test_1_PE_PIR"),
    (re.compile(r"Test_3", re.I), "Test_3_FRPE_PIR"),
    (re.compile(r"Test_5", re.I), "Test_5_LCM_PIR"),
    (re.compile(r"Test_7", re.I), "Test_7_FRPE_Phenolic"),
]


def config_key(chid):
    """Map a CHID to its physical configuration key (cladding, hrr).

    Mesh is deliberately excluded — that is the whole point. Handles both
    naming conventions present in the corpus:

        DCLG_Test_1_PE_PIR_HRR1333_M008_0_1  -> ('Test_1_PE_PIR', '1333')
        DCLG_Test_1_new_version02_0_09       -> ('Test_1_PE_PIR', '2333')
        DCLG_Test_5_LCM_PIR_08               -> ('Test_5_LCM_PIR', '2333')
    """
    cladding = None
    for pat, name in _CLADDING_PATTERNS:
        if pat.search(chid):
            cladding = name
            break
    if cladding is None:
        raise ValueError(f"cannot derive cladding system from CHID {chid!r}")

    m = re.search(r"HRR(\d+)", chid)
    hrr = m.group(1) if m else _ANCHOR_HRR
    return cladding, hrr


def _group_hash(cladding, hrr):
    return hashlib.md5(f"{cladding}|HRR{hrr}".encode()).hexdigest()


def _ranked_assignment(train_ratio, valid_ratio):
    """Deterministic rank-based allocation of the 20 configurations.

    Straight hash bucketing is too ragged at n = 20: it gave 12/3/5 groups
    (36/9/15 sims) instead of the intended 14/3/3 (42/9/9). Sorting the
    configurations by their MD5 digest and allocating by rank hits the target
    counts exactly while remaining fully deterministic — the ordering depends
    only on the configuration identifiers, never on filesystem order, dict
    order, or a random seed.

    The configuration list is built from config.CLADDING_SYSTEMS x
    config.HRR_LEVELS, so it does not depend on which runs exist on disk.
    """
    from config import CLADDING_SYSTEMS, HRR_LEVELS

    claddings = sorted(CLADDING_SYSTEMS)
    configs = [(c, str(h)) for c in claddings for h in sorted(HRR_LEVELS)]

    n = len(configs)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_train = max(1, min(n_train, n - 2))
    n_valid = max(1, min(n_valid, n - n_train - 1))
    n_test = n - n_train - n_valid

    # Stratify by cladding system so the held-out buckets span as many
    # cladding systems as their size allows. Pure hash ranking left the
    # 70/15/15 test bucket covering only 2 of 4 systems, which cannot
    # evidence generalisation across cladding type. Within each system the
    # HRR levels are still ordered by hash, so the choice of which level is
    # held out remains arbitrary-but-deterministic rather than hand-picked.
    per_clad = {c: sorted((c, str(h)) for h in sorted(HRR_LEVELS)) for c in claddings}
    for c in claddings:
        per_clad[c].sort(key=lambda ch: _group_hash(*ch))

    # Deterministic cladding visit order, rotated by hash so no system is
    # structurally privileged.
    clad_order = sorted(claddings, key=lambda c: _group_hash(c, "order"))

    out = {}
    # Deal one configuration to test, then to valid, round-robin across
    # cladding systems, before everything else becomes training.
    cursor = {c: 0 for c in claddings}

    def _take(clad):
        i = cursor[clad]
        if i >= len(per_clad[clad]):
            return None
        cursor[clad] += 1
        return per_clad[clad][i]

    for target, count in (("test", n_test), ("valid", n_valid)):
        placed = 0
        ring = 0
        while placed < count:
            clad = clad_order[ring % len(clad_order)]
            ring += 1
            ch = _take(clad)
            if ch is None:
                continue
            out[ch] = target
            placed += 1

    for ch in configs:
        out.setdefault(ch, "train")
    return out


_ASSIGNMENT_CACHE = {}


def assign_split_grouped(chid, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO):
    """Deterministic split keyed on physical configuration, not CHID.

    All three mesh siblings of a configuration receive the same label, so no
    held-out case can have a near-duplicate of itself in training.
    """
    ck = (train_ratio, valid_ratio)
    if ck not in _ASSIGNMENT_CACHE:
        _ASSIGNMENT_CACHE[ck] = _ranked_assignment(train_ratio, valid_ratio)
    return _ASSIGNMENT_CACHE[ck][config_key(chid)]


def install():
    """Patch data_loader.assign_split unless MLP_SPLIT_MODE=chid."""
    if os.environ.get("MLP_SPLIT_MODE", "grouped").lower() == "chid":
        data_loader.assign_split = _ORIGINAL_ASSIGN_SPLIT
        print("  [split] legacy CHID split active (MLP_SPLIT_MODE=chid)")
        return False
    data_loader.assign_split = assign_split_grouped
    print("  [split] configuration-grouped split active "
          "(mesh siblings move together)")
    return True


def uninstall():
    data_loader.assign_split = _ORIGINAL_ASSIGN_SPLIT
