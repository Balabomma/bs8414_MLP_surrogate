"""Restrict the prediction target to the 16 external thermocouples.

The corpus carries 24 channels in three groups of eight:

    External_LV1_*     8   level 1, external face      indices 0-7
    External_LV2_*     8   level 2, external face      indices 8-15
    Insulation_LV2_*   8   level 2, within insulation  indices 16-23

Setting MLP_SENSORS=external drops the insulation group and trains on the 16
external channels only. Selection is by NAME, never by position, so a change in
column order in the source CSVs cannot silently substitute the wrong channels.

What this costs, stated so it is not discovered later: the BR 135 *internal*
fire spread criterion is assessed on the insulation Level 2 thermocouples, so a
16-channel model cannot evaluate it at all. The *external* criterion, which is
what Sections 3.3 and 6.4 report, is assessed on External_LV2 and is retained in
full. Level 1 externals are retained because they set the start time of the
assessment window.

Default is off, so every existing 24-channel model stays reproducible.
"""
import os

import numpy as np

EXTERNAL_PREFIXES = ("External_LV1", "External_LV2")


def enabled():
    return os.environ.get("MLP_SENSORS", "all").lower() == "external"


def n_sensors():
    return 16 if enabled() else 24


def columns(sensor_names):
    """Indices of the external channels, in their original order."""
    idx = [i for i, n in enumerate(sensor_names)
           if str(n).startswith(EXTERNAL_PREFIXES)]
    if len(idx) != 16:
        raise ValueError(
            "expected 16 external channels, matched %d in %r"
            % (len(idx), [str(n) for n in sensor_names]))
    return idx


def apply(outputs, masks, sensor_names):
    """Slice targets to the external channels. No-op unless enabled."""
    if not enabled():
        return outputs, masks, sensor_names
    idx = columns(sensor_names)
    kept = [sensor_names[i] for i in idx]
    print("  [sensors] external only: %d of %d channels retained "
          "(insulation Level 2 dropped)" % (len(idx), len(sensor_names)))
    masks_out = masks
    if masks is not None and np.asarray(masks).ndim == 3:
        masks_out = np.asarray(masks)[:, :, idx]
    return np.asarray(outputs)[:, :, idx], masks_out, kept
