# bs8414_MLP_surrogate — MLP ablation control for the KAN surrogate

Thermocouple surrogate for the BS 8414-1 facade fire test. This project is a
**control experiment**, not an independent model line: its only job is to answer
whether the **KAN B-spline edge activations** in `bs8414_KAN_surrogate` earn their
place, by running the identical experiment with conventional **MLP** blocks and
nothing else changed.

**Two corpora run side by side, and they are not comparable to each other** —
different targets, different splits, different files. Nothing in the 60-sim
pipeline was modified when the Part1 pipeline was added.

| | 60-sim ablation | Part1 ablation (since 2026-08-18) |
|---|---|---|
| Files | `config.py`, `data_loader.py`, `model.py`, `train.py`, `evaluate.py` | the `*_part1.py` set |
| Data | `bs8414_KAN_surrogate/data/training_data` (read directly, no copy) | `D:\Bs8414_05052026\Part1\_completed` |
| Target | **24 TCs** × 181 steps, 3 groups | **16 external TCs** × 181 steps, 2 groups **+ 5-channel HRR budget** |
| Split | 70/15/15 CHID-hash → **40 / 12 / 8** on the clean 60 | hash → **141 / 20 / 23** of 184 usable |
| Model | `MLPAttentionLSTM`, 853,099 params (KAN: 852,795, +0.04 %) | `Part1MLPAttentionLSTM`, 851,012 params (KAN: 850,765, +0.03 %) |
| Parity proof | `verify_parity.py` | `verify_parity_part1.py` |
| Checkpoints | `models_mlp*`, `models_mlp_70_15_15*`, … | `models_part1_*` |
| Streamlit | — | `app_part1.py` |

---

## The one variable

Every `KANLinear` (learnable B-splines on edges, RBF basis, 8 knots, SiLU
residual) → **`MLPLinear` = `LayerNorm(W₂·GELU(W₁x + b₁) + b₂)`**.

**Capacity is matched, not reduced.** Each MLP block's hidden width is sized so
its parameter count equals the KAN block it replaces:
`h = round((o·i·(K+1) + i) / (i + o + 1))`. `--plain` (or `match_params=False`)
gives the capacity-free variant (675,960 on the 60-sim corpus) if the "same
width" ablation is wanted too.

**Disclosed second difference:** the KAN objective's
`LAMBDA_KAN_REG · kan_regularization(model)` (spline L2 + knot roughness) has no
MLP counterpart and is **dropped rather than faked** — hence `LAMBDA_REG = 0.0`
in `model_part1.py`. AdamW weight decay is identical.

`MLPLinear` is imported from this project's `model.py` by `model_part1.py`, not
re-declared, so the block under test on Part1 is the same one the 60-sim ablation
used.

---

## What is held fixed

### 60-sim ablation

| Component | Source |
|---|---|
| Dataset | `bs8414_KAN_surrogate/data/training_data` — read directly, no copy (`config.SIMS_DIR`) |
| Split | deterministic 70/15/15 CHID-hash → **40 / 12 / 8** on the clean 60-sim set |
| Input vector | 39 features: cladding id, HRR, mesh, 13 material, 20 extended-FDS/derived, D*/dx, physics t_ig, anchor confidence + ignition-anchor scalar |
| Backbone | TimeEncoding → MultiScaleConv(3/9/27) → 2-layer BiLSTM(96) → 4-head self-attention → 3 grouped decoders + skip |
| Causal channels | Δ(t−t_anchor), ignition bump, q(t), Q(t), q(t)^(2/3) |
| Output constraint | hard ambient clamp at 18 °C (per-sensor, scaled space) |
| Loss | peak-weighted MSE × tail-downweight (τ=1.8) + init + smooth + relative + growth + decay + energy |
| Optimisation | AdamW 2e-3 / wd 2e-4, cosine warm restarts, EMA 0.999, clip 1.0, ≤1500 epochs, patience 200 |
| Ensembling | 12 candidates → greedy ≤5 by pooled valid R², inverse-val-loss weights |
| Evaluation | the KAN project's frozen contract, transplanted verbatim (`evaluate.py`) |

### Part1 ablation

Byte-identical to the KAN project (`verify_parity_part1.py` checks SHA-256 *and*
the arrays built inside each venv): `config_part1.py`, `data_loader_part1.py`,
`train_part1.py`, `evaluate_part1.py`, `physics_part1.py`, `explain_part1.py`,
`causal_part1.py`. Plus, inside `model_part1.py`:

- conditioning — cladding(12) + insulation(5) + **geometry(8)** embeddings + 13
  material features, identical embedding widths;
- backbone — sinusoidal TimeEncoding, `input_proj`, MultiScaleConv (k=3/9/27),
  2-layer BiLSTM(96), 4-head self-attention + LayerNorm, param→output skips;
- two thermocouple group decoders + the HRR head, same shapes and dropout;
- hard ambient / zero-HRR output clamps.

`MODEL_NAME = "MLP-Attention-LSTM (Part1, KAN ablation control)"`.

**Geometry is a single 8-way embedding** over observed flag combinations (bit 0
`noair`, bit 1 `nogap`, bit 2 `nocb`), so it learns arbitrary interactions
between the three modifiers but **cannot extrapolate to a combination absent from
training**.

---

## Layout

```
config.py               60-sim config; DATA_DIR points at the KAN project
config_part1.py         Part1 corpus contract                       (shared file)
data_loader.py          transplanted verbatim from the KAN project (SHA-verified)
data_loader_part1.py    Part1 CHID/material/split logic             (shared file)
part1_dataset.py        Part1 loader helper: 185 configurations, one mesh, one source
anchor_features.py      transplanted verbatim
features_v4.py          transplanted verbatim
features_v6.py          transplanted verbatim
sensor_subset.py        restrict the target to the 16 external thermocouples
hrr_targets.py          total-HRR trajectories, supervise the mediating variable
model.py                MLP-Attention-LSTM — MLPLinear, the one 60-sim change
model_part1.py          Part1MLPAttentionLSTM — the one Part1 change
model_causal.py         causal-structure variant: temperature driven by PREDICTED total HRR
physics_part1.py        physics gates + optional closure/geometry penalties  (shared)
train.py                v9 recipe, self-contained
train_part1.py          Part1 trainer                               (shared file)
evaluate.py             frozen 60-sim eval contract  -> models_*/outputs/
evaluate_part1.py       frozen Part1 eval contract                  (shared file)
validate_physics.py     4 physics-sanity checks, same thresholds as the KAN project
killer_excluded.py      pooled R² with the LES-chaotic Test_1 HRR1333 family excluded
verify_parity.py        PROVES 60-sim data/split/feature identity with the KAN project
verify_parity_part1.py  the same proof for the Part1 shared layer
bestofn_driver.py       replicates 2 and 3 (retrain-variance population)
compare_with_kan.py     head-to-head table -> comparison_kan_vs_mlp.md
compare_splits.py       70/15/15 vs 80/10/10
grouped_split.py        configuration-grouped split — fixes mesh-sibling leakage
stratified_split.py     system-stratified configuration-grouped split
campaign_split.py       leave-one-configuration-out and friends (review-response campaign)
campaign_hooks.py       two env-guarded dataset hooks for that campaign
collect_campaign.py / collect_leakctrl.py / collect_regsweep.py   campaign aggregation
baselines.py            classical baselines on the grouped split
grid_convergence.py     GCI for the validated FDS configurations
defective_runs.py       the ten corpus sims whose cladding core does not burn
threshold_and_monotonicity.py   threshold/timing metrics + monotonicity probe
tierA_centreline.py     Tier A on centreline-conforming TC positions
(measured data + origin_figures.py        NOT DISTRIBUTED — see below)
shap_attribution.py     additive-feature attribution (Cremades et al. 2025)
explain_part1.py / causal_part1.py     SHAP and causal explainability  (shared files)
dump_per_sensor*.py / dump_ts_part1.py per-sensor and per-case dumps
metrics_full_part1.py / time_inference_part1.py
plotting.py / make_figures.py            figure sets
app_common_part1.py / app_part1.py / run_app.ps1    the Streamlit app
app_assets/             part1_materials.json + selected_model.json
run_*.ps1 / rerun_*.sh  campaign and replicate drivers
```

`*.pre-sensors`, `*.pre-part1`, `*.pre-campaign-backup`, `*.pre-causal` and
`*.claude-overwrote-2026-08-21` files are before-state snapshots of deliberate
edits. Leave them alone.

---

## Training

Always from this directory, in **this project's own venv**, on the NVIDIA GPU.

```powershell
cd D:\VS_projects\bs8414_MLP_surrogate
.\venv\Scripts\activate
nvidia-smi                                  # confirm the 4090 is free
```

### 60-sim ablation

```powershell
python verify_parity.py                                  # must print PARITY PROVEN
python -u train.py --model-dir models_mlp   > train_mlp_run1.log 2> train_mlp_run1.err.log
python evaluate.py         --model-dir models_mlp
python validate_physics.py --model-dir models_mlp
python killer_excluded.py  --model-dir models_mlp

python -u bestofn_driver.py                              # replicates r2, r3
python compare_with_kan.py --bench
```

### Part1 ablation

```powershell
python verify_parity_part1.py                            # shared layer identical
python -u train_part1.py --members 3 --seed 61 --model-dir models_part1_bal_s61_r4 `
       > train_part1_bal_s61_r4.log 2> train_part1_bal_s61_r4.err.log
python evaluate_part1.py --model-dir models_part1_bal_s61_r4
```

`train_part1.py` options — identical to the KAN project's copy:

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | `models_part1` | output directory — **must not already hold a run** |
| `--members` | 1 | ensemble members; the standard protocol is **3**, seeded `seed + m_idx` |
| `--seed` | 42 | base seed; the balanced design uses 42 / 45 / 48 / 52 / 55 / 61 |
| `--split` | `hash` | `hash` or `system` — **different experiments**, not comparable |
| `--epochs` | 500 | max epochs (early stopping, patience 60) |
| `--force` | off | overwrite a directory that already holds checkpoints |

**Two safety features exist because of real defects.** The trainer refuses to
write into a directory that already holds checkpoints (three retrains were lost
to silent overwrites), and every log carries a `[sentinel] |dW| = ...` line after
epoch 0 proving the weights moved. That sentinel exists because a refactor once
pulled `zero_grad/backward/step` inside a `if train and LAMBDA_REG:` guard — the
KAN was unaffected (`LAMBDA_REG = 2e-3`), but **this project's `LAMBDA_REG` is
0.0**, so the MLP took zero optimiser steps and reported flat loss with "best @
epoch 0", which reads exactly like instant convergence. That run is preserved as
`models_part1_mlp_r1_BROKEN_no_optimiser_step/`.

Roughly **2.5 min per member** on the 4090, so a 3-member ensemble is ~8 min.

Environment knobs: `PART1_SPLIT`, `PART1_LAMBDA_HRR` (default 0.3),
`PART1_LAMBDA_CLOSURE` / `PART1_LAMBDA_GEOM` (default 0.0), `PART1_SIMS_DIR`.
The review-response campaign adds its own env-guarded hooks — see
`campaign_hooks.py` and the `run_*.ps1` drivers.

### Reading the run directories

| Pattern | What it is |
|---|---|
| `models_mlp`, `models_mlp_70_15_15*`, `models_mlp_80_10_10*` | 60-sim ablation, by split ratio; `_grouped`, `_stratified` variants use the leakage-fixed splits |
| `models_mlp_reg_A0..A4` | regularisation sweep (dropout, weight decay, hidden width) |
| `models_mlp_leakctrl*` | leakage-control experiment (and its swapped controls) |
| `models_mlp_loco_f00..f19` | leave-one-configuration-out folds |
| `models_mlp_causal16*`, `models_mlp_ext16*`, `models_mlp_c16drop*` | causal-structure and 16-sensor variants |
| `models_mlp_singleres_M009` | single-resolution control |
| `models_part1_bal_s{42,45,48,52,55,61}_r{1,2,3}` | the **balanced 12+ design**: 3 independent retrains at each base seed, separating within-seed (cudnn non-determinism, ~0.018) from between-seed variance (~0.062) |
| `models_part1_mlp_r1..r5`, `models_part1_r1`, `models_part1_final_r1` | Part1 main sequence |
| `models_part1_fix184_*`, `*_c184`, `models_part1_184_r1` | runs on the corrected 184-configuration corpus |
| `models_part1_sys_mlp_seed*` | `PART1_SPLIT=system` — **not comparable** to hash runs |
| `models_part1_mlp_r1_BROKEN_no_optimiser_step` | preserved defect; excluded everywhere, including the app |

---

## Evaluation

```powershell
python evaluate_part1.py     --model-dir models_part1_bal_s61_r2
python metrics_full_part1.py --model-dir models_part1_bal_s61_r2 --split test
python compare_splits.py
python dump_per_sensor_ext.py
python dump_ts_part1.py
python time_inference_part1.py
python explain_part1.py            # SHAP attribution
python causal_part1.py             # interventional / causal explainability
python shap_attribution.py         # 60-sim SHAP
```

`evaluate_part1.py` is the **frozen evaluation contract** — fixed before any
candidate model existed and byte-identical across every Part1 sensor project. Per
split it reports pooled and per-group TC R²/RMSE in °C, R²/RMSE per HRR channel
in kW, a **per-geometry breakdown**, and physics sanity gates. Results land in
`evaluation_part1.json` in the run directory, which is what the root-level
`select_best_model.py` and `collect_model_comparison.py` read.

### Reading the result

**Compare populations, never single runs.** The KAN champion on the 60-sim corpus
is a 3-replicate population (`bs8414_KAN_surrogate/bestofn_v9_summary.txt`):
valid R² 0.8297 ± 0.0040, test R² 0.8472 ± 0.0257, combined 0.8384 ± 0.0121. Test
R² alone swings ~0.05 between identical retrains, so a single MLP run proves
nothing. `compare_with_kan.py` compares 3-vs-3 means and reports any delta inside
the **±0.02 R² band as inconclusive**, never as a win. Physics sanity
(`validate_physics.py`) must pass before any accuracy claim counts — a better R²
with broken physics is reported as broken.

On **Part1 (hash split)** the best available run is `models_part1_bal_s61_r2`:
combined valid+test TC R² **0.8096** (valid 0.8063 / test 0.8130, test RMSE
61.8 °C, HRR R² 0.951), chosen from 35 candidates with a **0.0094** margin —
inside the noise band, so best-available rather than significantly best. The
balanced design showed between-seed variance (~0.062) is **3.4×** the within-seed
spread (~0.018), which is why runs are reported by seed cell and why an
unbalanced, seed-42-heavy pool has a composition-dependent mean.

Against the Part1 KAN's 0.8526, the gap is 0.043 — outside ±0.02 — but state it
with the design: balanced 3×n replicate populations on the hash split, physics
gates passing on both. Cross-architecture numbers for every arm are collected in
`..\model_comparison.csv`.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_MLP_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # alongside the KAN app for a side-by-side
```

`run_app.ps1` activates this venv, picks `app_part1.py`, exports the material
table if it is missing, prints GPU status, then starts Streamlit. Manual
equivalent: `.\venv\Scripts\activate ; streamlit run app_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it predicts the 16
external thermocouples and the 5-channel HRR budget over the 0–1800 s / 10 s
grid, auto-predicting on every change in under a second. Tabs: per-group TC curves
with an optional ±1 sd ensemble band and a peak table; the HRR budget with its
closure residual; a BR 135 external screen; and a data tab with CSV export and the
exact 16-d input vector.

`app_part1.py` is **byte-identical** to the KAN project's copy and binds only to
`Part1Surrogate` / `MODEL_NAME` from `model_part1.py` — the one file that differs.
Run both apps on different ports and any difference you see is attributable to the
architecture, which is the whole point of this project.

The run selector ranks directories by the recorded `combined_tc_r2`;
**★ selected** is `models_part1_bal_s61_r2`, whose weights are the only ones kept
out of `.gitignore` so a fresh clone can predict without retraining. Currently 39
runs offered, 1 hidden (the BROKEN one).

**Prediction only** — the app never reads `D:\Bs8414_05052026`. Runtime inputs are
the checkpoint plus `app_assets/part1_materials.json`, written once by the
root-level `export_app_assets.py`, which refuses to write unless a
cladding/insulation id provably fixes its material block.

**Part1 enforced from the checkpoint, not the filename**: a run is offered only if
it carries 16 `sensor_names` *and* an HRR head, so a 60-sim checkpoint (24 TCs, no
HRR head) cannot load into a 16-channel model and silently mislabel channels.
Directories containing `BROKEN` are excluded outright. Anything hidden is listed in
the sidebar with the reason.

What the app refuses to claim:

- **Geometry cannot extrapolate** — an 8-way embedding over *observed* flag
  combinations; an absent build-up gets a warning banner, not a plausible curve.
- **BR 135 external only** — Part1 instruments the external face, so the internal
  fire-spread criterion cannot be assessed. A surrogate reading, not a test result.
- **The ensemble band is member disagreement**, not a calibrated interval.
- **`Q_TOTAL` is a residual budget term** near numerical noise, labelled as such.

`app_common_part1.py`, `app_part1.py` and `run_app.ps1` are byte-identical across
the projects that hold them — never hand-edit one copy; edit and re-copy. Full app
contract: `..\APPS.md`.

---

### Repository layout

Run logs, analysis records and before-state snapshots are grouped so the project
root holds only what you run:

```
<project>/
  README.md            this file
  *.py                 all modules and entry points — flat, at the root
  models_*/            checkpoints + per-run provenance JSON
  app_assets/          part1_materials.json, selected_model.json
  docs/                results records and analyses (PART1_RESULTS.md, analysis_*.md, ...)
  logs/                paired .log / .err.log run logs — the provenance of every number
  archive/             before-state snapshots of deliberate edits (*.pre-*, *.bak)
```

**Python stays at the project root, deliberately.** Every module imports flat
(`from config_part1 import ...`) and `config.py` / `config_part1.py` derive
`PROJECT_DIR`, `MODEL_DIR`, `OUTPUT_DIR` and `SLICE_DIR` from `__file__` — moving
them into a `src/` package would silently repoint model and slice paths, and
those files must stay byte-identical across all eleven surrogate projects for
`verify_parity_part1.py` to pass. New run logs still land at the root; move them
into `logs/` when you tidy.

`CLAUDE.md` is git-ignored: it is the working brief for agent sessions, not part
of the published artefact.

---

## Third-party measured data is not distributed

Some figures in the wider study compare measured DCLG thermocouple traces
against the surrogate. Those measurements were supplied by a third party and
are **not this project's data to publish**, so neither the data nor the scripts
that read it are in this repository: `measured_traces.py`,
`measured_1003_1029.py` and `origin_figures.py` are git-ignored and absent.

Nothing in the pipeline depends on them. The surrogate is trained and scored
against **FDS output**, not against measurements, so training, evaluation, the
physics gates, the parity checkers and the Streamlit app all run on a fresh
clone exactly as documented above.

To reproduce those particular comparison figures, obtain the measurements from
their owner and supply your own plotting script.

---

## Non-negotiables

See `CLAUDE.md` for the full list; the short version:

1. **Never change the data pipeline.** `data_loader.py`, `anchor_features.py`,
   `features_v4.py`, `features_v6.py` are byte-identical transplants and
   `verify_parity.py` checks their SHA-256 plus the resulting arrays. If the KAN
   project changes one, **re-copy it** and re-run the checker — never hand-edit.
   The same rule governs every shared `*_part1.py` file and
   `verify_parity_part1.py`.
2. **Never change the data root.** `config.SIMS_DIR` points at the KAN project's
   `data/training_data`; `config_part1.SIMS_DIR` at the Part1 batch.
3. **One variable only.** Anything that changes the input, the split, the
   backbone, the loss or the evaluation makes this a different experiment rather
   than a control.
4. **Populations, not single runs**; **±0.02 R² is inconclusive**; **physics gates
   before accuracy claims**; **always state the split** (`hash` or `system`).

## Related

`..\CLAUDE.md` (project map, Part1 contract) · `..\APPS.md` (app and deployment
contract) · `CLAUDE.md` (this project's working rules) ·
`bs8414_KAN_surrogate` (the arm under test) · `bs8414_surrogate_model`
(Attention-LSTM V3 baseline) · `bs8414_samba_mlp_surrogate` (MLP-Samba sensor arm).
