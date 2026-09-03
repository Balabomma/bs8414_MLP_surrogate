"""Train the MLP-Attention-LSTM surrogate — KAN-ablation control.

The training recipe is the KAN champion's v9 recipe (`bs8414_KAN_surrogate/
train_v9.py` = `train_v8.main(tau=1.8)`), transplanted here so this project is
self-contained. Every hyperparameter, augmentation, loss term, scheduler, EMA
setting and the greedy-ensemble protocol is byte-for-byte the same value as the
run that produced `models_v9`:

  - 39-feature physics-causal input vector (features_v6) + LOO ignition anchors
    with per-cladding calibrated jitter (anchor_features)
  - 4x augmentation: clean / noise(0.03) / same-cladding anchor-compatible mixup /
    amplitude scaling (0.85-1.15)
  - PhysicsLoss: peak-weighted MSE with an adaptive tail-downweight
    tw = 1/(1+(|e|/tau)^2), tau=1.8; + t=0 anchor, smoothness, hot-region relative
    error, growth-phase monotonicity, decay-phase constraint, HRR-energy correlation
  - AdamW(2e-3, wd 2e-4), CosineAnnealingWarmRestarts(T_0=75, T_mult=2), EMA 0.999,
    grad-clip 1.0, up to 1500 epochs, patience 200
  - 12 candidate members -> greedy ensemble (max 5) by pooled valid R2,
    inverse-val-loss weights; all candidates saved for post-hoc re-selection
  - deterministic 70/15/15 CHID-hash split on the shared KAN dataset

THE ONE VARIABLE: the model class (MLP edge blocks instead of KAN B-spline edge
blocks, parameter-matched — see model.py).

THE ONE UNAVOIDABLE SECOND DIFFERENCE: the KAN recipe adds
`LAMBDA_KAN_REG * kan_regularization(model)` (spline L2 + adjacent-knot
roughness) to the training objective. It has no MLP counterpart — there are no
splines — so it is dropped rather than faked. AdamW weight decay (2e-4) is
identical in both. Disclosed here and in the analysis, not swept under the rug.

Usage:
    python -u train.py                        # -> models_mlp/
    python -u train.py --model-dir models_mlp_r2
    python -u train.py --plain                # capacity-free variant (h = out)
"""
import argparse
import copy
import os
import time as time_mod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, PROJECT_DIR
from data_loader import build_dataset, prepare_data_splits
import grouped_split          # configuration-grouped split (mesh-sibling leakage fix)
grouped_split.install()
import stratified_split; stratified_split.install()   # no-op unless MLP_SPLIT_MODE=stratified
import campaign_split         # review campaign: LOCO folds (no-op unless MLP_CAMPAIGN set)
import campaign_hooks         # review campaign: scaler scope / mesh filter (no-op unless set)
campaign_split.install()
from features_v6 import build_params_v6, N_V6_FEATURES
from anchor_features import build_bank, anchors_for, predict_anchor
from model import MLPAttentionLSTM, count_parameters

# Causal-structure variant (MLP_CAUSAL=1): temperature is driven by a PREDICTED
# total heat release, q_tot = q_crib + softplus(cladding_head), instead of the
# prescribed crib source alone. See model_causal.py for why.
USE_CAUSAL = os.environ.get("MLP_CAUSAL", "0") == "1"
LAMBDA_HRR = float(os.environ.get("MLP_LAMBDA_HRR", "0.05"))
HRR_SCALE_MW = 10.0          # normaliser so the aux term is O(1) against the MSE
_HRR_LOOKUP = {}             # (cladding, hrr_norm, mesh_norm) -> (T,) MW

# ── hyperparameters (identical to bs8414_KAN_surrogate/train_v3.py) ───────
# Env-guarded overrides for the regularisation sweep. Every default below is
# the original value, so a run with none of these variables set reproduces the
# existing models byte-for-byte. The sweep exists because train R2 = 0.97
# against test R2 = 0.72 is an overfit, not a capacity limit.
def _envf(name, default):
    return type(default)(os.environ.get(name, default))

HIDDEN_SIZE = _envf("MLP_HIDDEN", 96)
EMBEDDING_DIM = 24
N_HEADS = 4
DROPOUT = _envf("MLP_DROPOUT", 0.15)
LEARNING_RATE = 2e-3
WEIGHT_DECAY = _envf("MLP_WD", 2e-4)
BATCH_SIZE = 8
NUM_EPOCHS = 1500
PATIENCE = 200
NUM_KNOTS = 8              # only used to size the param-matched MLP blocks

N_CANDIDATES = 12
N_KEEP = 5

LAMBDA_PHYSICS_INIT = 0.1
LAMBDA_SMOOTH = 0.005
LAMBDA_PEAK_WEIGHT = 1.0
LAMBDA_REL = 0.5
ANCHOR_JITTER = 0.02       # global jitter std (mixup branch), 0.02*1800 = 36 s
MIXUP_ANCHOR_TOL = 0.08    # only mix sims whose anchors differ < 144 s
EMA_DECAY = 0.999

LAMBDA_GROWTH = 0.05
LAMBDA_DECAY = 0.02
LAMBDA_ENERGY = 0.01
GROWTH_END_IDX = 60        # 600 s
DECAY_START_IDX = 156      # 1560 s

TAU_TAIL = 1.8             # == train_v9.TAU_V9

N_CONTINUOUS = 2 + 13 + N_V6_FEATURES  # 38
MODEL_DIR = os.path.join(PROJECT_DIR, "models_mlp")


# ──────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────

class AugDataset(Dataset):
    """4x augmentation; mixup restricted to same cladding AND compatible anchor.

    Mirrors AugDatasetV3 + AugDatasetV6 from the KAN project: non-mixup branches
    jitter the anchor with the per-cladding calibrated std, the mixup branch
    jitters the blended anchor with the global ANCHOR_JITTER.
    """

    def __init__(self, params, outputs, masks, time_array, anchors, jitter_std):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        self.anchors = torch.FloatTensor(anchors)
        self.jitter_std = torch.FloatTensor(jitter_std)
        clad = params[:, 0].astype(int)
        self.pool = {}
        for i in range(len(params)):
            ok = np.where((clad == clad[i]) &
                          (np.abs(anchors - anchors[i]) < MIXUP_ANCHOR_TOL))[0]
            self.pool[i] = ok if len(ok) > 0 else np.array([i])

    def __len__(self):
        return len(self.params) * 4

    def _jit(self, a):
        return a + torch.randn(1).item() * ANCHOR_JITTER

    def _jit_i(self, a, i):
        return a + torch.randn(1).item() * float(self.jitter_std[i])

    def __getitem__(self, idx):
        n = len(self.params)
        if idx < n:                                   # clean
            i = idx
            return (self.params[i], self.outputs[i], self.masks[i],
                    self.time_array, self._jit_i(self.anchors[i], i))
        elif idx < n * 2:                             # additive noise
            i = idx - n
            return (self.params[i],
                    self.outputs[i] + torch.randn_like(self.outputs[i]) * 0.03,
                    self.masks[i], self.time_array,
                    self._jit_i(self.anchors[i], i))
        elif idx < n * 3:                             # mixup (same cladding)
            i = idx - n * 2
            pool = self.pool[i]
            j = int(pool[torch.randint(0, len(pool), (1,)).item()])
            lam = max(np.random.beta(0.3, 0.3), 0.5)
            return (lam * self.params[i] + (1 - lam) * self.params[j],
                    lam * self.outputs[i] + (1 - lam) * self.outputs[j],
                    torch.ones_like(self.masks[i]), self.time_array,
                    self._jit(lam * self.anchors[i] + (1 - lam) * self.anchors[j]))
        else:                                         # amplitude scaling
            i = idx - n * 3
            s = 0.85 + torch.rand(1).item() * 0.30
            b = self.outputs[i][0:1, :]
            return (self.params[i], b + (self.outputs[i] - b) * s,
                    self.masks[i], self.time_array,
                    self._jit_i(self.anchors[i], i))


class EvalDataset(Dataset):
    def __init__(self, params, outputs, masks, time_array, anchors):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        self.anchors = torch.FloatTensor(anchors)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return (self.params[idx], self.outputs[idx], self.masks[idx],
                self.time_array, self.anchors[idx])


# ──────────────────────────────────────────────────────────────────────
# Loss (== PhysicsLossV8 with tau=1.8, i.e. the v9 objective)
# ──────────────────────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """Peak-weighted MSE with adaptive tail-downweight + physics terms.

    Terms, in order: robust weighted MSE, t=0 anchoring, temporal smoothness,
    hot-region relative error, growth-phase monotonicity, decay-phase constraint,
    HRR-vs-meanT energy correlation.
    """

    def __init__(self, init_temp, scaler_mean, scaler_scale, tau=TAU_TAIL):
        super().__init__()
        self.init_temp = init_temp
        self.register_buffer("s_mean", scaler_mean)
        self.register_buffer("s_scale", scaler_scale)
        self.tau = float(tau)

    def forward(self, pred, target, masks=None, hrr=None):
        if masks is not None:
            me = masks.unsqueeze(-1)
        else:
            me = torch.ones_like(pred[:, :, :1])
        nv = me.sum().clamp(min=1.0)

        err2 = (pred - target) ** 2
        at = torch.abs(target) * me
        mt = at.max().clamp(min=1.0)
        w = 1.0 + LAMBDA_PEAK_WEIGHT * (at / mt)
        with torch.no_grad():
            e = (pred - target).abs()
            tw = 1.0 / (1.0 + (e / self.tau) ** 2)   # detached tail-downweight
        wmse = (err2 * w * tw * me).sum() / nv

        init_l = ((pred[:, 0, :] - self.init_temp.unsqueeze(0)) ** 2).mean()

        diff = pred[:, 1:, :] - pred[:, :-1, :]
        if masks is not None:
            dm = (masks[:, 1:] * masks[:, :-1]).unsqueeze(-1)
            sm = (diff ** 2 * dm).sum() / dm.sum().clamp(min=1.0)
        else:
            sm = (diff ** 2).mean()

        targ_degC = target * self.s_scale + self.s_mean
        hot = (targ_degC > 100.0).float() * me
        abs_degC = (pred - target).abs() * self.s_scale
        rel = (abs_degC / targ_degC.clamp(min=100.0) * hot).sum() / hot.sum().clamp(min=1.0)

        base = (wmse + LAMBDA_PHYSICS_INIT * init_l + LAMBDA_SMOOTH * sm
                + LAMBDA_REL * rel)

        dg = pred[:, 1:GROWTH_END_IDX, :] - pred[:, :GROWTH_END_IDX - 1, :]
        growth = F.relu(-dg).pow(2).mean()
        dd = pred[:, DECAY_START_IDX + 1:, :] - pred[:, DECAY_START_IDX:-1, :]
        decay = F.relu(dd).pow(2).mean()
        energy = pred.new_tensor(0.0)
        if hrr is not None and hrr.numel() > 2:
            mean_t = pred.mean(dim=(1, 2))
            h = hrr - hrr.mean()
            mm = mean_t - mean_t.mean()
            denom = h.std() * mm.std()
            if denom > 1e-6:
                energy = 1.0 - (h * mm).mean() / denom

        total = (base + LAMBDA_GROWTH * growth + LAMBDA_DECAY * decay
                 + LAMBDA_ENERGY * energy)
        return total, {}


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].clone()
        return sd


# ──────────────────────────────────────────────────────────────────────
# Train / eval helpers
# ──────────────────────────────────────────────────────────────────────

def make_model(device, mean, scale, match_params=True):
    cls = MLPAttentionLSTM
    if USE_CAUSAL:
        from model_causal import CausalMLPAttentionLSTM as cls
    m = cls(
        n_timesteps=N_TIMESTEPS, n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
        embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS, dropout=DROPOUT,
        num_knots=NUM_KNOTS, n_continuous=N_CONTINUOUS, match_params=match_params,
    ).to(device)
    m.set_ambient(mean, scale)
    return m


def _hrr_batch(p, q_tot):
    """Total-HRR target for each row of the batch, keyed on its design vector.

    Mixup-augmented rows interpolate the design vector and therefore match no
    key; they return have=0 and contribute nothing to the auxiliary term. That
    is intended -- there is no measured total HRR for a synthetic blend."""
    tgt = torch.zeros_like(q_tot)
    have = torch.zeros(q_tot.shape[0], device=q_tot.device)
    for i in range(p.shape[0]):
        k = (round(float(p[i, 0])), round(float(p[i, 1]), 4), round(float(p[i, 2]), 4))
        v = _HRR_LOOKUP.get(k)
        if v is not None:
            tgt[i] = torch.as_tensor(v, device=q_tot.device, dtype=q_tot.dtype)
            have[i] = 1.0
    return tgt, have


def train_epoch(model, loader, opt, criterion, device, ema):
    model.train()
    total, n = 0, 0
    for p, t, m, ta, a in loader:
        p, t, m, a = p.to(device), t.to(device), m.to(device), a.to(device)
        ta = ta[0].to(device)
        opt.zero_grad()
        out = model(p, ta, a)
        pred = out[0]
        loss, _ = criterion(pred, t, m, hrr=p[:, 1])
        if USE_CAUSAL and len(out) > 2 and _HRR_LOOKUP:
            tgt, have = _hrr_batch(p, out[2])
            if have.any():
                d = (out[2] - tgt) / HRR_SCALE_MW
                loss = loss + LAMBDA_HRR * ((d ** 2).mean(dim=1) * have).sum() / have.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        total += loss.item()
        n += 1
    return total / n


def validate(model, loader, criterion, device):
    model.eval()
    total, n = 0, 0
    with torch.no_grad():
        for p, t, m, ta, a in loader:
            p, t, m, a = p.to(device), t.to(device), m.to(device), a.to(device)
            ta = ta[0].to(device)
            total += criterion(model(p, ta, a)[0], t, m, hrr=p[:, 1])[0].item()
            n += 1
    return total / n


def member_preds_degC(model, params_np, anchors_np, time_array, scaler, device):
    model.eval()
    with torch.no_grad():
        p = torch.FloatTensor(params_np).to(device)
        a = torch.FloatTensor(anchors_np).to(device)
        ta = torch.FloatTensor(time_array).to(device)
        out = model(p, ta, a)[0].cpu().numpy()
    return (out.reshape(-1, N_SENSORS) * scaler.scale_ + scaler.mean_).reshape(out.shape)


def pooled_r2(pred_stack, weights, actual):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    ens = np.tensordot(w, pred_stack, axes=(0, 0))
    return r2_score(actual.flatten(), ens.flatten())


def cladding_loo_resid(bank):
    """Mean leave-one-out anchor-regression residual per cladding id."""
    out = {}
    for c in range(4):
        pos = np.where(bank["clad"] == c)[0]
        resid = []
        for p in pos:
            pred = predict_anchor(c, float(bank["hrr"][p]), float(bank["mesh"][p]),
                                  bank, exclude_idx=int(p))
            resid.append(abs(pred - bank["anchor"][p]))
        out[c] = float(np.mean(resid)) if resid else 0.05
    return out


# ──────────────────────────────────────────────────────────────────────

def main(model_dir=MODEL_DIR, tau=TAU_TAIL, match_params=True):
    print("=" * 70)
    print("  BS8414 MLP-Attention-LSTM Surrogate - Training (KAN ablation control)")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    MODEL_DIR = model_dir
    os.makedirs(MODEL_DIR, exist_ok=True)

    params, outputs, masks, meta, sensor_names = build_dataset()
    if USE_CAUSAL:
        import hrr_targets
        _hrr_arr, _hrr_have = hrr_targets.build(meta)
        for _i, _m in enumerate(meta):
            if _hrr_have[_i]:
                _HRR_LOOKUP[(round(float(params[_i, 0])), round(float(params[_i, 1]), 4),
                             round(float(params[_i, 2]), 4))] = _hrr_arr[_i]
        print("  [hrr] lookup keys: %d" % len(_HRR_LOOKUP))
    import sensor_subset
    outputs, masks, sensor_names = sensor_subset.apply(outputs, masks, sensor_names)
    params, outputs, masks, meta = campaign_hooks.filter_mesh(
        params, outputs, masks, meta)
    _, _, _, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)
    scaler = campaign_hooks.refit_scaler(outputs, split_info, scaler)

    train_idx = split_info["train_idx"]
    valid_idx = split_info["valid_idx"]
    test_idx = split_info["test_idx"]

    bank = build_bank(params, outputs, train_idx)
    np.savez(os.path.join(MODEL_DIR, "anchor_bank.npz"), **bank)

    params_v6 = build_params_v6(params, meta, bank)
    print(f"\n  Input vector: {params_v6.shape[1]} features "
          f"({len(params)} sims: {len(train_idx)}/{len(valid_idx)}/{len(test_idx)})")
    print(f"  Robust data loss: tail-downweight TAU={tau} (scaled); ambient clamp on")
    print(f"  MLP blocks param-matched to KAN: {match_params}")

    np.save(os.path.join(MODEL_DIR, "output_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "output_scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "sensor_names.npy"), sensor_names)

    train_anchors = anchors_for(params[train_idx], bank,
                                loo_bank_pos=np.arange(len(train_idx)))
    valid_anchors = anchors_for(params[valid_idx], bank)
    test_anchors = anchors_for(params[test_idx], bank)

    loo = cladding_loo_resid(bank)
    jitter_std = np.array([max(0.02, loo[int(params[i, 0])])
                           for i in train_idx], dtype=np.float32)

    train_scaled = scaler.transform(
        outputs[train_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)
    valid_scaled = scaler.transform(
        outputs[valid_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)

    aug = AugDataset(params_v6[train_idx], train_scaled, masks[train_idx],
                     time_array, train_anchors, jitter_std)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        EvalDataset(params_v6[valid_idx], valid_scaled, masks[valid_idx],
                    time_array, valid_anchors),
        batch_size=BATCH_SIZE)

    valid_actual = outputs[valid_idx]
    test_actual = outputs[test_idx]

    print(f"  Train: {len(train_idx)} real -> {len(aug)} augmented")

    init_temp = torch.FloatTensor(
        scaler.transform(np.full((1, N_SENSORS), 18.0, dtype=np.float32))[0]
    ).to(device)
    criterion = PhysicsLoss(
        init_temp,
        torch.FloatTensor(scaler.mean_), torch.FloatTensor(scaler.scale_),
        tau=tau,
    ).to(device)

    print(f"\n--- Training {N_CANDIDATES} candidate members ---")
    members, val_losses, valid_pred_stack = [], [], []
    t0 = time_mod.time()

    for m_idx in range(N_CANDIDATES):
        seed = int(os.environ.get("MLP_SEED_BASE", "42")) + m_idx * 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = make_model(device, scaler.mean_, scaler.scale_, match_params)
        if m_idx == 0:
            print(f"  Params: {count_parameters(model):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=75, T_mult=2, eta_min=1e-5)
        ema = EMA(model, decay=EMA_DECAY)
        eval_model = make_model(device, scaler.mean_, scaler.scale_, match_params)

        best_v, best_s, pat = float("inf"), None, 0
        for ep in range(1, NUM_EPOCHS + 1):
            train_epoch(model, train_loader, opt, criterion, device, ema)
            eval_model.load_state_dict(ema.apply_to(model))
            vl = validate(eval_model, valid_loader, criterion, device)
            sched.step()
            if vl < best_v:
                best_v = vl
                best_s = copy.deepcopy(eval_model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        model.load_state_dict(best_s)
        members.append(model)
        val_losses.append(best_v)
        vp = member_preds_degC(model, params_v6[valid_idx], valid_anchors,
                               time_array, scaler, device)
        valid_pred_stack.append(vp)
        vr2 = r2_score(valid_actual.flatten(), vp.flatten())
        print(f"  [M{m_idx+1:2d}] valid loss={best_v:.4f}  valid R2={vr2:.4f}  "
              f"({time_mod.time()-t0:.0f}s elapsed)", flush=True)

    valid_pred_stack = np.stack(valid_pred_stack)

    torch.save({
        "model_states": [m.state_dict() for m in members],
        "n_models": len(members),
        "val_losses": val_losses,
    }, os.path.join(MODEL_DIR, "all_candidates.pt"))

    print(f"\n--- Greedy ensemble selection (max {N_KEEP} members, by valid R2) ---")
    selected, best_r2 = [], -np.inf
    while len(selected) < N_KEEP:
        best_cand, best_cand_r2 = None, best_r2
        for c in range(N_CANDIDATES):
            if c in selected:
                continue
            trial = selected + [c]
            r2 = pooled_r2(valid_pred_stack[trial],
                           [1.0 / val_losses[t] for t in trial], valid_actual)
            if r2 > best_cand_r2:
                best_cand, best_cand_r2 = c, r2
        if best_cand is None:
            break
        selected.append(best_cand)
        best_r2 = best_cand_r2
        print(f"  + M{best_cand+1}  -> valid R2={best_r2:.4f}")

    weights = np.array([1.0 / val_losses[i] for i in selected])
    weights = weights / weights.sum()
    ensemble = [members[i] for i in selected]
    print(f"  Selected: {[f'M{i+1}' for i in selected]}  "
          f"weights={[f'{w:.3f}' for w in weights]}")
    print(f"  Time: {time_mod.time()-t0:.0f}s")

    torch.save({
        "model_states": [m.state_dict() for m in ensemble],
        "n_models": len(ensemble),
        "ensemble_weights": weights.tolist(),
        "selected_candidates": [i + 1 for i in selected],
        "candidate_val_losses": val_losses,
        "n_continuous": N_CONTINUOUS,
        "match_params": match_params,
    }, os.path.join(MODEL_DIR, "best_model.pt"))

    for name, sp, sa, an in [("Valid", params_v6[valid_idx], valid_actual, valid_anchors),
                             ("Test", params_v6[test_idx], test_actual, test_anchors)]:
        stack = np.stack([member_preds_degC(m, sp, an, time_array, scaler, device)
                          for m in ensemble])
        ens = np.tensordot(weights, stack, axes=(0, 0))
        r2 = r2_score(sa.flatten(), ens.flatten())
        rmse = np.sqrt(mean_squared_error(sa.flatten(), ens.flatten()))
        hot = sa.flatten() > 100
        mape = np.mean(np.abs((sa.flatten()[hot] - ens.flatten()[hot])
                              / sa.flatten()[hot])) * 100
        sub = (ens.flatten() < 17.0).sum()
        print(f"\n  {name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAPE(T>100)={mape:.1f}%  "
              f"sub-ambient pts={sub}")
        for k in range(len(sa)):
            print(f"    sim {k}: R2={r2_score(sa[k].flatten(), ens[k].flatten()):.3f}")

    print(f"\n  Saved -> {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--tau", type=float, default=TAU_TAIL)
    ap.add_argument("--plain", action="store_true",
                    help="capacity-free MLP (hidden = out) instead of param-matched")
    args = ap.parse_args()
    d = args.model_dir
    if not os.path.isabs(d):
        d = os.path.join(PROJECT_DIR, d)
    main(model_dir=d, tau=args.tau, match_params=not args.plain)
