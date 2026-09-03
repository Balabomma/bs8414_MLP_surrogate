"""Classical baselines on the configuration-grouped split (manuscript Sec 6.6).

Without a baseline an R2 value cannot be judged. Four cheap predictors are
evaluated on exactly the same held-out configurations, with the same masking
and the same metric definitions as the network:

  B0  global training-mean curve   -- the null model; per-sensor mean over time
  B1  per-cladding training mean   -- exploits the categorical factor only
  B2  nearest neighbour in HRR     -- within cladding, nearest source level
  B3  POD + ridge regression       -- k modes of the training response,
                                     ridge on (cladding one-hot, hrr, dstar/dx)

Metrics mirror evaluate.py: pooled R2 against the grand mean, mean per-channel
R2 against each channel's own mean, RMSE and MAE in degC over masked points.
"""
import os
import numpy as np

os.environ.setdefault("MLP_SPLIT", "70_15_15")
import grouped_split
grouped_split.install()
from data_loader import build_dataset, prepare_data_splits  # noqa: E402
from config import MESH_SIZES  # noqa: E402

RIDGE = 1e-3
N_MODES = 10


def metrics(actual, pred, mask):
    m = mask.astype(bool)
    a = np.concatenate([actual[i][m[i]].flatten() for i in range(len(actual))])
    p = np.concatenate([pred[i][m[i]].flatten() for i in range(len(pred))])
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2_pool = 1 - ss_res / ss_tot
    # per-channel
    chans = []
    n_s = actual[0].shape[1]
    for s in range(n_s):
        av = np.concatenate([actual[i][m[i], s] for i in range(len(actual))])
        pv = np.concatenate([pred[i][m[i], s] for i in range(len(pred))])
        if av.size < 2 or av.var() == 0:
            continue
        chans.append(1 - ((av - pv) ** 2).sum() / ((av - av.mean()) ** 2).sum())
    rmse = float(np.sqrt(((a - p) ** 2).mean()))
    mae = float(np.abs(a - p).mean())
    return r2_pool, float(np.mean(chans)), rmse, mae


def main():
    params, outputs, masks, meta, names = build_dataset()
    _, _, _, _, si, _ = prepare_data_splits(params, outputs, masks, meta)
    tr, te = si["train_idx"], si["test_idx"]
    print(f"\n  train {len(tr)} sims | held-out {len(te)} sims\n")

    clad = params[:, 0].astype(int)
    hrr = params[:, 1].astype(float)
    Ytr = outputs[tr]

    preds = {}

    # B0 global training mean curve
    preds["B0 global training-mean curve"] = np.repeat(
        Ytr.mean(axis=0)[None, ...], len(te), axis=0)

    # B1 per-cladding training mean
    p1 = np.zeros((len(te),) + outputs.shape[1:])
    for k, i in enumerate(te):
        same = [j for j in tr if clad[j] == clad[i]]
        p1[k] = outputs[same].mean(axis=0) if same else Ytr.mean(axis=0)
    preds["B1 per-cladding training mean"] = p1

    # B2 nearest neighbour in HRR within cladding
    p2 = np.zeros_like(p1)
    for k, i in enumerate(te):
        same = [j for j in tr if clad[j] == clad[i]]
        if not same:
            p2[k] = Ytr.mean(axis=0); continue
        j = min(same, key=lambda j: abs(hrr[j] - hrr[i]))
        p2[k] = outputs[j]
    preds["B2 nearest neighbour in HRR"] = p2

    # B3 POD + ridge on (cladding one-hot, hrr, mesh)
    F = Ytr.reshape(len(tr), -1)
    mu = F.mean(axis=0)
    U, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
    k = min(N_MODES, Vt.shape[0])
    basis = Vt[:k]
    Ctr = (F - mu) @ basis.T
    energy = float((S[:k] ** 2).sum() / (S ** 2).sum())

    def feats(idx):
        n = len(idx)
        X = np.zeros((n, 4 + 2))
        for a, i in enumerate(idx):
            X[a, int(clad[i])] = 1.0
            X[a, 4] = hrr[i]
            X[a, 5] = params[i, 2]
        return X

    Xtr, Xte = feats(tr), feats(te)
    A = Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ctr)
    Cte = Xte @ W
    p3 = (Cte @ basis + mu).reshape((len(te),) + outputs.shape[1:])
    preds[f"B3 POD({k} modes, {energy*100:.1f}% energy) + ridge"] = p3

    print(f"{'baseline':<44}{'R2_chan':>9}{'R2_pool':>9}{'RMSE':>9}{'MAE':>9}")
    print("-" * 80)
    for name, P in preds.items():
        r2p, r2c, rmse, mae = metrics(outputs[te], P, masks[te])
        print(f"{name:<44}{r2c:>9.3f}{r2p:>9.3f}{rmse:>9.1f}{mae:>9.1f}")
    print("-" * 80)
    print(f"{'MLP-Attention-LSTM (Sec 6.1)':<44}{0.559:>9.3f}{0.7135:>9.3f}"
          f"{130.2:>9.1f}{60.0:>9.1f}")


if __name__ == "__main__":
    main()
