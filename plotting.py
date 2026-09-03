"""Publication-style plot helpers — same figures the KAN project produces.

Transplanted from `bs8414_KAN_surrogate/evaluate_kan.py` (identical figure
geometry, styling, statistics boxes and CSV side-outputs) with the model label
changed to MLP, so KAN and MLP figures can be placed side by side without any
presentation confound.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, T_END

MODEL_LABEL = "MLP"

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.linewidth": 1.2,
    "axes.labelsize": 13, "axes.titlesize": 13, "axes.labelweight": "bold",
    "xtick.major.size": 6, "xtick.major.width": 1.2,
    "xtick.minor.size": 3, "xtick.minor.width": 0.8,
    "xtick.direction": "in", "xtick.top": True,
    "ytick.major.size": 6, "ytick.major.width": 1.2,
    "ytick.minor.size": 3, "ytick.minor.width": 0.8,
    "ytick.direction": "in", "ytick.right": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.framealpha": 1.0,
    "legend.fontsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def inverse_scale(scaled, mean, scale):
    shape = scaled.shape
    return (scaled.reshape(-1, N_SENSORS) * scale + mean).reshape(shape)


def plot_scatter(actual, predicted, output_dir, set_label, ratio_label):
    a, p = actual.flatten(), predicted.flatten()
    r2 = r2_score(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a, p, alpha=0.15, s=3, c="#1f77b4", edgecolors="none", rasterized=True)
    lims = [min(a.min(), p.min()) - 20, max(a.max(), p.max()) + 20]
    ax.plot(lims, lims, "k-", linewidth=1.5, label="y = x")
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(x_line, x_line * 0.9, x_line * 1.1, alpha=0.08, color="gray",
                    label="+/-10%")
    ax.set_xlabel("FDS Temperature (°C)")
    ax.set_ylabel(f"{MODEL_LABEL}-Predicted Temperature (°C)")
    ax.set_title(f"{MODEL_LABEL} Ensemble - {set_label} ({ratio_label})")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black")
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.1f} °C\nN = {len(a):,}",
            transform=ax.transAxes, fontsize=11, va="top", bbox=props)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower right", fancybox=False); ax.grid(False)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"scatter_{set_label}.png")
    plt.savefig(fname); plt.close()
    return fname


def plot_timeseries(actual, predicted, time_s, sensor_idx, sensor_label,
                    sims_meta, output_dir, set_label, mask):
    paths = []
    for i in range(actual.shape[0]):
        meta = sims_meta[i]
        chid_short = f"{meta['cladding']}_HRR{meta['hrr']}_{meta['mesh']}"
        m = mask[i].astype(bool)
        t = time_s[m]
        a = actual[i, m, sensor_idx]
        p = predicted[i, m, sensor_idx]
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.plot(t, a, color="#1f77b4", linewidth=2.0, label="FDS Simulation")
        ax.plot(t, p, color="#d62728", linewidth=2.0, linestyle="--",
                label=f"{MODEL_LABEL}-Attention-LSTM (Ensemble)")
        err = np.abs(a - p)
        ax.fill_between(t, p - err, p + err, alpha=0.1, color="#d62728")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
        hrr_mw = meta["hrr"] * 1.5 / 1000
        ax.set_title(f"{sensor_label}\n{meta['cladding']} | "
                     f"HRRPUA={meta['hrr']} kW/m² ({hrr_mw:.2f} MW) | Mesh={meta['mesh']}")
        ax.set_xlim(0, T_END)
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        r2 = r2_score(a, p); rmse = np.sqrt(mean_squared_error(a, p))
        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
        ax.text(0.97, 0.05, f"R²={r2:.3f}\nRMSE={rmse:.1f}°C",
                transform=ax.transAxes, fontsize=10, ha="right", bbox=props)
        ax.legend(loc="best", fancybox=False); ax.grid(False)
        plt.tight_layout()
        safe = sensor_label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        fname = os.path.join(output_dir, f"ts_{safe}_{set_label}_{chid_short}.png")
        plt.savefig(fname); plt.close()
        pd.DataFrame({
            "Time_s": t, "FDS_Actual_degC": a,
            f"{MODEL_LABEL}_Predicted_degC": p, "Abs_Error_degC": err,
        }).to_csv(fname.replace(".png", ".csv"), index=False)
        paths.append(fname)
    return paths


def plot_per_sensor_bar(sensor_results, output_dir, ratio_label):
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [s["sensor"] for s in sensor_results]
    r2s = np.array([s["r2"] for s in sensor_results])
    colors = ["#2ca02c" if r > 0.9 else ("#ff7f0e" if r > 0.8 else "#d62728")
              for r in r2s]
    ax.bar(range(len(names)), r2s, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.9, color="#2ca02c", linestyle="--", linewidth=1, label="R² = 0.9")
    ax.axhline(0.8, color="#ff7f0e", linestyle="--", linewidth=1, label="R² = 0.8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Test R²"); ax.set_ylim(min(0, r2s.min() - 0.05), 1.05)
    ax.set_title(f"{MODEL_LABEL} per-sensor test R² ({ratio_label})  -  "
                 f"avg={r2s.mean():.3f}")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = os.path.join(output_dir, "per_sensor_R2.png")
    plt.savefig(fname); plt.close()
    return fname


def plot_error_hist(actual, predicted, output_dir, set_label):
    err = (predicted - actual).flatten()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(err, bins=60, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction error (°C)  [pred - FDS]")
    ax.set_ylabel("Count")
    ax.set_title(f"{MODEL_LABEL} error distribution - {set_label}")
    me, sd = err.mean(), err.std()
    p95 = np.percentile(np.abs(err), 95)
    props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
    ax.text(0.97, 0.95, f"mean={me:.1f}°C\nstd={sd:.1f}°C\n|err| P95={p95:.1f}°C",
            transform=ax.transAxes, fontsize=10, ha="right", va="top", bbox=props)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    fname = os.path.join(output_dir, f"error_hist_{set_label}.png")
    plt.savefig(fname); plt.close()
    return fname
