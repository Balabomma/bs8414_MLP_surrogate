"""The ten corpus simulations whose cladding core does not burn.

Every M009 deck of the three combustible-core systems omits the &REAC that
consumes the core fuel. The core MATL still pyrolyses -- Core_FRPE releases
SPEC_ID(1,1)='ETHYLENE' -- and ETHYLENE is still declared as a tracked &SPEC,
but with no reaction consuming it the released fuel never combusts. The core
absorbs its heat of reaction and contributes no heat release.

Diagnosis, in the M008/M009 sibling pair for Test_3_FRPE_PIR_HRR1667:

    M008   &REAC FUEL='ETHYLENE' present; ETHYLENE not declared as &SPEC
    M009   &REAC FUEL='ETHYLENE' absent;  ETHYLENE declared as passive &SPEC

Corroborated by the peak temperatures. Test 5, whose mineral core is
non-combustible by design and whose decks are therefore unaffected, falls
monotonically as the mesh coarsens (1300 / 1167 / 549 degC). All three affected
systems instead place M009 below BOTH neighbours -- Test 3 at 1252 / 963 / 1068,
Test 7 at 1185 / 863 / 958 -- which a discretisation trend does not produce.

Two consequences, and they differ in scope:

  physics   all ten runs have a non-combusting core
  features  five of them also mis-parse their material vector, because the
            property extractor falls through to the insulation reaction when the
            core reaction is absent. Test 1's fall-through happens to land on the
            same values, so its feature vectors are clean and its physics is not.

Set MLP_DROP_DEFECTIVE=1 to exclude them. The default is off, so existing
models and their stored metrics stay reproducible.

No FDS output is deleted or modified by this module; the exclusion is applied in
the loader only.
"""
import os

# Every M009 deck of Tests 1, 3 and 7. Test 5 is absent by design.
DEFECTIVE = (
    "DCLG_Test_1_PE_PIR_HRR1333_M009_0_1",
    "DCLG_Test_1_PE_PIR_HRR1667_M009_0_1",
    "DCLG_Test_1_PE_PIR_HRR2000_M009_0_1",
    "DCLG_Test_3_FRPE_PIR_HRR1333_M009_0_1",
    "DCLG_Test_3_FRPE_PIR_HRR1667_M009_0_1",
    "DCLG_Test_3_FRPE_PIR_HRR2000_M009_0_1",
    "DCLG_Test_3_FRPE_PIR_HRR2100_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR1333_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR1667_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR2000_M009_0_1",
)

# The subset that also carries a mis-parsed material vector.
MISPARSED = (
    "DCLG_Test_3_FRPE_PIR_HRR1333_M009_0_1",
    "DCLG_Test_3_FRPE_PIR_HRR1667_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR1333_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR1667_M009_0_1",
    "DCLG_Test_7_FRPE_Phenolic_HRR2000_M009_0_1",
)


def enabled():
    return os.environ.get("MLP_DROP_DEFECTIVE", "0") == "1"


def keep_index(meta):
    """Indices of meta entries to retain, or None when the filter is off."""
    if not enabled():
        return None
    bad = set(DEFECTIVE)
    return [i for i, m in enumerate(meta) if m["chid"] not in bad]
