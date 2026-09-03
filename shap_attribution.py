"""Additive-feature attribution for the surrogate, following Cremades, Hoyas &
Vinuesa (2024), Int. J. Heat Fluid Flow 110:109205.

Three additive-feature-attribution estimators are run on the same wrapped model
and the same held-out set, and all three are reported:

    deep SHAP     DeepExplainer (DeepLIFT rescale rule, exact-additive)
    gradient SHAP GradientExplainer (expected gradients over the background)
    integrated gradients   single-baseline path integral, 128 steps

Deep SHAP is attempted first because it is what was asked for. It carries a
documented architectural restriction -- the DeepLIFT rescale rule is defined for
feed-forward and convolutional stacks -- and this network contains an LSTM,
multi-head attention and LayerNorm. The script therefore reports which layer
blocks it rather than silently degrading, and each estimator gets a deep copy of
the network so that hooks left by a failed attempt cannot contaminate the next.

Agreement between estimators is reported as a Spearman rank correlation, and
each estimator is repeated over several disjoint background draws so that the
spread of the ranking is visible. Attribution variance is a known failure mode
of these methods and no single draw is reported on its own.

    python shap_attribution.py --model-dir models_mlp_70_15_15_grouped
"""
import argparse
import copy
import os
import warnings

import numpy as np
import torch

os.environ.setdefault("MLP_SPLIT", "70_15_15")
import grouped_split                                      # noqa: E402
grouped_split.install()
from data_loader import build_dataset, prepare_data_splits  # noqa: E402
from features_v4 import V4_FEATURES                       # noqa: E402
from features_v6 import build_params_v6                   # noqa: E402
from anchor_features import anchors_for                   # noqa: E402
from evaluate import load_ensemble                        # noqa: E402

from config import MATERIAL_FEATURES                    # noqa: E402

BASE16 = ["cladding_id", "hrr_norm", "mesh_norm"] + list(MATERIAL_FEATURES)
V6_EXTRA = ["dstar_over_dx", "t_ignition_phys", "anchor_confidence"]
FEATURES = BASE16 + list(V4_FEATURES) + V6_EXTRA

# Families assigned by name, not by substring, because two of the names are
# misleading: dstar_over_dx is a resolution index despite sitting in the v6
# block, and hrr_mesh is a source-resolution interaction rather than either
# alone. It is reported separately for that reason.
FAMILIES = [
    ("cladding system", ["cladding_id", "anchor_confidence"]),
    ("source strength", ["hrr_norm", "hrr_inv", "hrr_mw"]),
    ("mesh resolution", ["mesh_norm", "dstar_over_dx"]),
    ("source x resolution", ["hrr_mesh"]),
    ("material properties", None),          # everything else
    ("derived thermal", ["t_ignition_phys"]),
]


def families():
    named = {n for _, ns in FAMILIES if ns for n in ns}
    out = []
    for label, ns in FAMILIES:
        if ns is None:
            idx = [i for i, n in enumerate(FEATURES) if n not in named]
        else:
            idx = [FEATURES.index(n) for n in ns]
        out.append((label, idx))
    assert sum(len(i) for _, i in out) == len(FEATURES)
    return out


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


class Wrap(torch.nn.Module):
    """params -> scalar per sample: mean predicted temperature in degC over the
    24 x 181 field.

    The anchor is pinned at the training-mean value for every sample, so the
    attribution is to the design vector alone rather than to a quantity derived
    from it. forward() returns (prediction, attention_weights), so only the
    first element is taken. The ambient floor is applied as a maximum, so the
    gradient vanishes wherever a channel is pinned at ambient -- early-time
    behaviour is therefore under-weighted by the gradient-based estimators.
    """

    def __init__(self, net, t, anchor, mean, scale):
        super().__init__()
        self.net, self.t = net, t
        self.register_buffer("anchor", anchor)
        self.register_buffer("mu", mean)
        self.register_buffer("sd", scale)

    def forward(self, p):
        out = self.net(p, self.t, self.anchor.expand(p.shape[0]))[0]
        return (out * self.sd + self.mu).mean(dim=(1, 2)).unsqueeze(1)


def run_kernel(make_w, bg, x):
    """Model-agnostic KernelSHAP.

    Needed because the two structural blind spots of the gradient estimators
    both fall on features this paper argues about. cladding_id enters through an
    embedding lookup, so its gradient is identically zero however important the
    system is; and the plume term q^(2/3) has an unbounded derivative at q = 0,
    which the source ramp passes through, so hrr_mw returns NaN. KernelSHAP
    perturbs inputs and re-evaluates, so neither applies.
    """
    import shap
    w = make_w()
    dev = x.device
    cud = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = True     # forward-only path, no RNN backward

    def f(z):
        out = []
        with torch.no_grad():
            for i in range(0, len(z), 256):
                b = torch.tensor(z[i:i + 256], dtype=torch.float32, device=dev)
                out.append(w(b).cpu().numpy())
        return np.concatenate(out)

    try:
        ex = shap.KernelExplainer(f, bg.cpu().numpy())
        phi = ex.shap_values(x.cpu().numpy(), nsamples=2 * len(FEATURES) + 2048,
                             silent=True)
    finally:
        torch.backends.cudnn.enabled = cud
    return np.asarray(phi)


def run_deep(make_w, bg, x):
    import shap
    from shap.explainers._deep import deep_pytorch as dp
    # LayerNorm is not in shap's PyTorch op table; without an entry it falls
    # through to the elementwise rescale rule, which is not valid for a layer
    # that normalises across the feature axis. Register it as linear so the
    # attempt fails (or succeeds) on the recurrent layers rather than on this.
    dp.op_handler["LayerNorm"] = dp.linear_1d
    w = make_w()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ex = shap.DeepExplainer(w, bg)
        return np.asarray(ex.shap_values(x, check_additivity=True))


def run_gradient(make_w, bg, x):
    import shap
    ex = shap.GradientExplainer(make_w(), bg)
    return np.asarray(ex.shap_values(x, nsamples=400))


def run_ig(make_w, bg, x, steps=128):
    w = make_w()
    base = bg.mean(dim=0, keepdim=True)
    total = torch.zeros_like(x)
    for s in range(1, steps + 1):
        xi = (base + (x - base) * (s / steps)).clone().requires_grad_(True)
        gr, = torch.autograd.grad(w(xi).sum(), xi)
        total += gr
    return ((x - base) * total / steps).detach().cpu().numpy()


def reduce(phi):
    phi = np.asarray(phi)
    while phi.ndim > 2:
        phi = phi[..., 0]
    return np.abs(phi).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_mlp_70_15_15_grouped")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--background", type=int, default=10)
    ap.add_argument("--member", type=int, default=0,
                    help="ensemble member to attribute (0-based)")
    ap.add_argument("--only", default=None,
                    help="comma-separated estimators to run")
    a = ap.parse_args()

    # cuDNN refuses RNN backward outside training mode. Attribution needs
    # gradients through the LSTM with dropout off, so the fused kernel is
    # disabled and the native implementation used instead; forward values are
    # unchanged, only the kernel differs.
    torch.backends.cudnn.enabled = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, weights, mean, scale, sensor_names, bank = load_ensemble(a.model_dir)
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, si, time_array = prepare_data_splits(params, outputs, masks, meta)
    pv6 = build_params_v6(params, meta, bank)
    anchors = anchors_for(params, bank)
    tr, te = np.asarray(si["train_idx"]), np.asarray(si["test_idx"])
    assert pv6.shape[1] == len(FEATURES), (pv6.shape[1], len(FEATURES))

    t = torch.tensor(time_array, dtype=torch.float32, device=dev)
    mu = torch.tensor(mean, dtype=torch.float32, device=dev)
    sd = torch.tensor(scale, dtype=torch.float32, device=dev)
    anch = torch.tensor(float(anchors[tr].mean()), dtype=torch.float32, device=dev)
    a.member = min(a.member, len(models) - 1)
    base_net = models[a.member].eval()
    x = torch.tensor(pv6[te], dtype=torch.float32, device=dev)

    def make_w():
        return Wrap(copy.deepcopy(base_net).eval(), t, anch, mu, sd).to(dev).eval()

    print("=" * 74)
    print("  ADDITIVE-FEATURE ATTRIBUTION -- %s" % a.model_dir)
    print("=" * 74)
    print("  ensemble member %d of %d (weight %.3f); %d features; %d held-out sims"
          % (a.member + 1, len(models), float(weights[a.member]),
             pv6.shape[1], len(te)))
    print("  target: mean predicted temperature (degC) over 24 sensors x %d steps"
          % len(time_array))
    print("  %d background draws of %d training simulations each\n"
          % (a.draws, a.background))

    rng = np.random.default_rng(0)
    methods = [("deep SHAP", run_deep, a.draws), ("kernel SHAP", run_kernel, 4),
               ("gradient SHAP", run_gradient, a.draws),
               ("integrated gradients", run_ig, a.draws)]
    if a.only:
        keep = {k.strip() for k in a.only.split(",")}
        methods = [m for m in methods if m[0] in keep]
    results, notes = {}, {}
    for label, fn, ndraw in methods:
        per_draw = []
        for d in range(ndraw):
            bg_idx = rng.choice(tr, size=a.background, replace=False)
            bg = torch.tensor(pv6[bg_idx], dtype=torch.float32, device=dev)
            try:
                per_draw.append(reduce(fn(make_w, bg, x)))
            except Exception as e:
                msg = str(e).strip().splitlines()[0][:150]
                notes[label] = "%s: %s" % (type(e).__name__, msg)
                break
        if per_draw:
            results[label] = np.stack(per_draw)
            print("  [ok]   %-22s %d/%d draws" % (label, len(per_draw), ndraw))
        else:
            print("  [FAIL] %-22s %s" % (label, notes[label]))

    if not results:
        print("\n  no estimator completed")
        return 1

    for primary in ("deep SHAP", "kernel SHAP", "gradient SHAP",
                    "integrated gradients"):
        if primary in results:
            break
    imp = results[primary].mean(axis=0)
    sd_imp = results[primary].std(axis=0)
    tot = np.nansum(imp)
    print("\n  reporting: %s (mean over draws)" % primary)

    print("\n  top 12 features by mean |phi|   (phi in degC)")
    print("  %-30s%10s%9s%9s" % ("feature", "mean|phi|", "sd", "share"))
    print("  " + "-" * 58)
    for k in np.argsort(np.where(np.isnan(imp), -np.inf, -imp))[:12]:
        print("  %-30s%10.3f%9.3f%8.1f %%"
              % (FEATURES[k], imp[k], sd_imp[k], 100 * imp[k] / tot))

    print("\n  by feature family")
    print("  %-32s%5s%11s%9s" % ("family", "n", "sum|phi|", "share"))
    print("  " + "-" * 58)
    for label, idx in families():
        if idx:
            v = np.nansum(imp[idx])
            nn = int(np.isnan(imp[idx]).sum())
            print("  %-32s%5d%11.3f%8.1f %%%s"
                  % (label, len(idx), v, 100 * v / tot,
                     "   (%d undefined)" % nn if nn else ""))

    fin = imp[~np.isnan(imp)]
    cum = np.sort(fin)[::-1].cumsum() / tot
    print("\n  effective dimension: %d of %d features carry 90 %% of attributed"
          % (int(np.searchsorted(cum, 0.90) + 1), len(FEATURES)))
    print("  importance; %d carry 50 %%; %d of %d features attract < 0.1 %% each"
          % (int(np.searchsorted(cum, 0.50) + 1),
             int((fin / tot < 0.001).sum()), len(fin)))

    print("\n  stability -- Spearman rank correlation of the 39-feature ranking")
    if len(results[primary]) > 1:
        rs = [spearman(results[primary][i], results[primary][j])
              for i in range(len(results[primary]))
              for j in range(i + 1, len(results[primary]))]
        print("    between background draws (%s): %.3f  [%.3f, %.3f]"
              % (primary, float(np.mean(rs)), min(rs), max(rs)))
    keys = list(results)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            print("    %s vs %s: %.3f" % (keys[i], keys[j],
                  spearman(results[keys[i]].mean(0), results[keys[j]].mean(0))))
    for label, msg in notes.items():
        print("\n  %s did not run -- %s" % (label, msg))

    out_npz = os.path.join(a.model_dir, "shap_attribution_m%d.npz" % a.member)
    np.savez(out_npz,
             features=np.array(FEATURES), primary=primary,
             **{k.replace(" ", "_"): v for k, v in results.items()})
    print("\n  saved %s" % os.path.join(a.model_dir, "shap_attribution.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
