# train.py — the three lines the campaign needs

Three of the four experiments cannot be driven from outside `train.py`, for the
reason set out in `campaign_hooks.py`: `train.py` binds `build_dataset` and
`prepare_data_splits` at import time, so there is no runtime hook that reaches
them. Only the LOCO split works without touching the file (it goes through
`data_loader.assign_split`, which is resolved at call time — the same hook
`grouped_split.py` uses).

Every change below is **guarded by an environment variable that defaults to the
current behaviour**. With no campaign variables set, a patched `train.py`
reproduces the existing replicates exactly. Nothing parity-locked is touched:
`data_loader.py`, `anchor_features.py`, `features_v4.py` and `features_v6.py`
stay byte-identical, so `verify_parity.py` still passes.

Apply with a text editor and re-run `verify_parity.py` before starting.

---

## 1. Imports — after the existing `import grouped_split` (line ~50)

```python
import grouped_split          # configuration-grouped split (mesh-sibling leakage fix)
import campaign_split         # ADD — LOCO folds  (no-op unless MLP_CAMPAIGN set)
import campaign_hooks         # ADD — scaler scope / mesh filter (no-op unless set)
campaign_split.install()      # ADD
```

## 2. Dataset construction — lines 349–351

Current:

```python
    params, outputs, masks, meta, sensor_names = build_dataset()
    _, _, _, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)
```

Patched:

```python
    params, outputs, masks, meta, sensor_names = build_dataset()
    params, outputs, masks, meta = campaign_hooks.filter_mesh(              # ADD
        params, outputs, masks, meta)                                       # ADD
    _, _, _, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)
    scaler = campaign_hooks.refit_scaler(outputs, split_info, scaler)       # ADD
```

The refit must sit **after** `prepare_data_splits` (it needs `train_idx`) and
**before** line ~366, where the scaler statistics are written to
`output_scaler_mean.npy` / `output_scaler_scale.npy`. Everything downstream —
`train_scaled`, `valid_scaled`, the ambient clamp at line ~400, `PhysicsLoss`,
and `make_model` — derives from this one object, so replacing it is sufficient
and nothing else needs to change.

## 3. Candidate seed — line 413

Current:

```python
        seed = 42 + m_idx * 100
```

Patched:

```python
        seed = int(os.environ.get("MLP_SEED_BASE", "42")) + m_idx * 100
```

`os` is already imported (line 38). With `MLP_SEED_BASE` unset the expression is
identical to the current one, so this cannot perturb the existing replicates.

---

## Sanity check before committing GPU time

```powershell
cd D:\VS_projects\bs8414_MLP_surrogate
$env:PYTHONPATH = "D:\VS_projects\bs8414_MLP_surrogate;<campaign-dir>"
.\venv\Scripts\python.exe .\verify_parity.py          # must still pass
.\venv\Scripts\python.exe -c "import campaign_split as c; [print(i, c.config_order()[i]) for i in range(20)]"
```

Then a five-minute smoke test that trains nothing:

```powershell
$env:MLP_CAMPAIGN = "loco"; $env:MLP_LOCO_FOLD = "0"
.\venv\Scripts\python.exe -c @"
import os, grouped_split, campaign_split, data_loader
grouped_split.install(); campaign_split.install()
from data_loader import build_dataset, prepare_data_splits
p,o,m,meta,_ = build_dataset()
_,_,_,_,si,_ = prepare_data_splits(p,o,m,meta)
print('train/valid/test =', len(si['train_idx']), len(si['valid_idx']), len(si['test_idx']))
"@
```

Expect `48 9 3` for a LOCO fold. If you see anything else, stop.
