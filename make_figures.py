"""Figure set for the Engineering with Computers manuscript.

Springer conventions: single column 84 mm, double column 174 mm, max height
234 mm; sans-serif lettering 8-12 pt; >=600 dpi for combination art.

Palette: Okabe-Ito subset validated for colour-vision deficiency
(min CVD dE = 8.6, min adjacent normal-vision dE = 25.8). Every series also
carries a line style and marker so the figures survive greyscale printing --
identity is never colour-alone.

Usage: python make_figures.py --model-dir models_mlp_70_15_15_grouped
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MLP_SPLIT", "70_15_15")
import grouped_split
grouped_split.install()
from data_loader import build_dataset, prepare_data_splits           # noqa: E402
from features_v6 import build_params_v6                              # noqa: E402
from anchor_features import anchors_for                              # noqa: E402
from evaluate import load_ensemble, predict, inverse_scale           # noqa: E402
from config import T_END, N_TIMESTEPS                                # noqa: E402

MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "black": "#000000"}
GREY, GRID = "#4D4D4D", "#D9D9D9"
OUT = "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.7,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CH_CRIT = "External_LV2_main02(1029)"
CH_TRIG = "External_LV1_main02(1003)"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"   wrote {OUT}/{name}.png|pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_mlp_70_15_15_grouped")
    a = ap.parse_args()

    models, w, mean, scale, names, bank = load_ensemble(a.model_dir)
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, si, time_array = prepare_data_splits(params, outputs, masks, meta)
    pv6 = build_params_v6(params, meta, bank)
    te = si["test_idx"]
    t = np.linspace(0, T_END, N_TIMESTEPS)
    pred = inverse_scale(predict(models, w, pv6[te], anchors_for(params[te], bank), time_array),
                         mean, scale)
    ic, it = names.index(CH_CRIT), names.index(CH_TRIG)

    def short(chid):
        c, h = grouped_split.config_key(chid)
        nm = {"Test_1_PE_PIR": "PE", "Test_3_FRPE_PIR": "FR-PE",
              "Test_5_LCM_PIR": "A2", "Test_7_FRPE_Phenolic": "FR-PE/ph"}[c]
        mm = "0.08" if "M008" in chid or chid.endswith(("_0_08", "_08")) else \
             "0.09" if "M009" in chid or chid.endswith("_0_09") else "0.10"
        return f"{nm}, {h}, {mm} m"

    # ---- Fig 1: held-out time histories at the criterion channel ----
    n = len(te)
    fig, axes = plt.subplots(3, 3, figsize=(W2, W2 * 0.62), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off"); continue
        i = te[k]
        ax.plot(t, outputs[i][:, ic], color=C["black"], ls="-", lw=1.0, label="FDS")
        ax.plot(t, pred[k][:, ic], color=C["verm"], ls="--", lw=1.0, label="Surrogate")
        crit = outputs[i][0, ic] + 600
        ax.axhline(crit, color=GREY, ls=":", lw=0.7)
        ax.set_title(short(meta[i]["chid"]), fontsize=7, pad=2)
        ax.set_xlim(0, T_END); ax.tick_params(length=2)
        if k % 3 == 0: ax.set_ylabel("Temperature (°C)")
        if k >= 6: ax.set_xlabel("Time (s)")
    axes.ravel()[0].legend(frameon=False, loc="upper right", handlelength=1.6,
                           borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig1_heldout_histories_DG1_1029")

    # ---- Fig 2: parity, by sensor group ----
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.95))
    groups = [("External L1", [j for j, s in enumerate(names) if s.startswith("External_LV1")], C["blue"], "o"),
              ("External L2", [j for j, s in enumerate(names) if s.startswith("External_LV2")], C["verm"], "s"),
              ("Insulation",  [j for j, s in enumerate(names) if s.startswith("Insulation")],  C["green"], "^")]
    for lbl, idxs, col, mk in groups:
        A = np.concatenate([outputs[te[k]][:, idxs].ravel() for k in range(n)])
        P = np.concatenate([pred[k][:, idxs].ravel() for k in range(n)])
        sel = np.random.default_rng(0).choice(len(A), size=min(4000, len(A)), replace=False)
        ax.scatter(A[sel], P[sel], s=1.2, c=col, marker=mk, alpha=0.35,
                   linewidths=0, label=lbl, rasterized=True)
    lim = [0, max(outputs[te].max(), pred.max()) * 1.02]
    ax.plot(lim, lim, color=C["black"], lw=0.8, ls="-")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("FDS temperature (°C)"); ax.set_ylabel("Surrogate (°C)")
    lg = ax.legend(frameon=False, loc="lower right", markerscale=6, handletextpad=0.4)
    for h in lg.legend_handles: h.set_alpha(1)
    fig.tight_layout(pad=0.3)
    save(fig, "fig2_parity_by_group")

    # ---- Fig 3: per-channel R2, sorted ----
    r2 = []
    m = masks[te].astype(bool)
    for s in range(len(names)):
        av = np.concatenate([outputs[te[k]][m[k], s] for k in range(n)])
        pv = np.concatenate([pred[k][m[k], s] for k in range(n)])
        r2.append(1 - ((av - pv) ** 2).sum() / ((av - av.mean()) ** 2).sum())
    r2 = np.array(r2); order = np.argsort(r2)
    cols = [C["green"] if names[j].startswith("Insulation") else
            (C["verm"] if names[j].startswith("External_LV2") else C["blue"]) for j in order]
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.36))
    ax.bar(range(len(r2)), r2[order], color=cols, width=0.72, linewidth=0)
    ax.axhline(0, color=C["black"], lw=0.6)
    ax.axhline(np.mean(r2), color=GREY, ls="--", lw=0.7)
    ax.text(0.5, np.mean(r2) + 0.03, f"mean {np.mean(r2):.2f}", fontsize=6.5, color=GREY)
    for j, lab in [(list(order).index(it), "DG1_1003"), (list(order).index(ic), "DG1_1029")]:
        ax.annotate(lab, (j, r2[order][j]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6.5, color=C["black"])
    ax.set_xticks([]); ax.set_ylabel("Held-out $R^2$")
    ax.set_xlabel("Thermocouple channel (sorted)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="External L1"),
                       Patch(color=C["verm"], label="External L2"),
                       Patch(color=C["green"], label="Insulation")],
              frameon=False, loc="upper left", ncol=3, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig3_per_channel_r2")

    # ---- Fig 4: baseline comparison ----
    labels = ["Training-\nmean", "Per-clad\nmean", "NN in\nHRR", "POD+\nridge", "MLP-Att-\nLSTM"]
    vals = [0.309, 0.371, 0.061, 0.466, 0.559]
    cols4 = [GREY] * 4 + [C["blue"]]
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    ax.bar(labels, vals, color=cols4, width=0.68, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5,
                color=C["black"] if i == 4 else GREY)
    ax.set_ylabel("Mean per-channel $R^2$"); ax.set_ylim(0, 0.68)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "fig4_baselines")

    print("\n  per-channel R2 recomputed for Fig 3: "
          f"mean={np.mean(r2):.4f} min={r2.min():.4f} max={r2.max():.4f}")


if __name__ == "__main__":
    main()
