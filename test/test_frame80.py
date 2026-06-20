"""
Chapter 6 - Frame 80 DOA estimation on real radar data.
BPSK-domain and Gauss-domain models tested separately, each with MUSIC/ESPRIT references.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GRADUATION_DIR = Path(__file__).resolve().parents[1]
_PROJECT_BASE = _GRADUATION_DIR.parent

for _p in [_PROJECT_BASE, _GRADUATION_DIR, _GRADUATION_DIR / "external"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.real_test import (
    estimate_one_frame,
    get_music_peak_and_spectrum,
    load_models,
    get_bpsk_specs,
    get_gauss_specs,
)
from utils.radar_utils import load_and_reshape, process_radar_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FRAME_IDX = 80
NUM_RX = 8
REAL_DATA_PATH = _GRADUATION_DIR / "data" / "raw" / "cropped.bin"

PLOT_DIR_REAL = _GRADUATION_DIR / "result" / "plot" / "real"
DATA_DIR_REAL = _GRADUATION_DIR / "result" / "test_data" / "real"
for _d in [PLOT_DIR_REAL, DATA_DIR_REAL]:
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["font.size"] = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_real_data():
    if not REAL_DATA_PATH.exists():
        print(f"找不到文件: {REAL_DATA_PATH}")
        return None
    raw_data = load_and_reshape(str(REAL_DATA_PATH), conjugate=True)
    print(f"成功加载数据: {REAL_DATA_PATH}")
    return raw_data


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_results_csv(records, out_path):
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {out_path}")
    return df


def save_frame80_results(results, mus_fbss, out_name):
    records = []
    for name, angle in results.items():
        delta = (
            angle - mus_fbss
            if not np.isnan(angle) and not np.isnan(mus_fbss)
            else np.nan
        )
        records.append({
            "algorithm": name,
            "estimate_angle": angle,
            "delta_to_music_fbss": delta,
            "is_reference": name == "MUSIC (FBSS)",
        })
    return save_results_csv(records, DATA_DIR_REAL / out_name)


# ---------------------------------------------------------------------------
# Traditional + DL results
# ---------------------------------------------------------------------------
def run_group(X_80, R_80, specs, group_name, device):
    """Load models for one group and estimate one frame."""
    print(f"\n{'='*60}")
    print(f"  加载 {group_name} 组模型")
    print(f"{'='*60}")
    loaded = load_models(specs, device=device)
    print(f"  已加载 {len(loaded)} 个模型, 开始推理...\n")
    return estimate_one_frame(X_80, R_80, loaded, device, num_rx=NUM_RX)


# ---------------------------------------------------------------------------
# Plot one group (standalone figure)
# ---------------------------------------------------------------------------
def plot_group_standalone(th_fbss, sp_fbss_db, algorithms, group_label, xlim, out_path, legend_elements):
    """Generate a standalone figure for one model group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(th_fbss, sp_fbss_db, color="#1f77b4", linewidth=1.0, alpha=0.45)
    ax.set_xlim(xlim)
    ax.set_ylim(-2.1, 0.8)
    ax.set_xlabel("Angle (degrees)", fontsize=14)
    ax.set_ylabel("Display offset", fontsize=14)

    offset = (0.18, 0.08)
    for alg in algorithms:
        if alg["angle"] is not None and not np.isnan(alg["angle"]):
            ax.scatter(
                alg["angle"], alg["y"],
                marker=alg["marker"], s=alg["size"],
                color=alg["color"], zorder=5, linewidth=1.2,
            )
            bbox = dict(boxstyle="round,pad=0.25", fc="white", ec=alg["color"], lw=0.8, alpha=0.9)
            ax.text(
                alg["angle"] + offset[0], alg["y"] + offset[1],
                f"{alg['name']}: {alg['angle']:.1f}",
                color=alg["color"], fontsize=12, ha="left", va="bottom",
                bbox=bbox, zorder=6,
            )

    # MUSIC(FBSS) baseline
    mus_fbss = next((a["angle"] for a in algorithms if a["name"] == "MUSIC (FBSS)"), np.nan)
    mus_raw = next((a["angle"] for a in algorithms if a["name"] == "MUSIC (Raw)"), np.nan)
    if not np.isnan(mus_fbss):
        ax.axvline(x=mus_fbss, color="blue", linestyle="--", linewidth=0.8, alpha=0.3)
        base_text_x = mus_raw - 0.8 if not np.isnan(mus_raw) else mus_fbss - 0.8
        ax.text(
            base_text_x, 0.55, "Base Line", color="blue",
            fontsize=15, fontweight="bold", ha="center", va="bottom",
        )

    ax.legend(handles=legend_elements, loc="upper left", ncol=2,
              fancybox=True, shadow=True, fontsize=10, edgecolor="black")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xticks(np.arange(-86, -65, 2))
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"{group_label} 图已保存: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw_data = load_real_data()
    if raw_data is None:
        return

    X_all, R_all = process_radar_data(raw_data, is_simulation=False)
    if X_all is None or R_all is None:
        print("雷达数据预处理失败。")
        return

    R_80 = R_all[TARGET_FRAME_IDX]
    X_80 = X_all[TARGET_FRAME_IDX]

    # --- Run both groups ---
    results_bpsk = run_group(X_80, R_80, get_bpsk_specs(), "BPSK Domain", device)
    results_gauss = run_group(X_80, R_80, get_gauss_specs(), "Gauss Domain", device)

    # Get MUSIC(FBSS) spectrum for plotting
    mus_fbss, th_fbss, sp_fbss = get_music_peak_and_spectrum(R_80, num_rx=NUM_RX, use_fbss=True)
    results_bpsk["MUSIC (FBSS)"] = mus_fbss
    results_gauss["MUSIC (FBSS)"] = mus_fbss
    spec_fbss_db = 10 * np.log10(np.abs(sp_fbss) / (np.max(np.abs(sp_fbss)) + 1e-12) + 1e-12)

    # --- Save CSVs ---
    save_frame80_results(results_bpsk, mus_fbss, "real_frame80_bpsk_results.csv")
    save_frame80_results(results_gauss, mus_fbss, "real_frame80_gauss_results.csv")

    # --- Traditional algorithms (shared) ---
    trad_algos = [
        {"name": "MUSIC (Raw)",   "angle": results_bpsk.get("MUSIC (Raw)", np.nan),   "marker": "^", "color": "green",  "size": 80,  "y": -0.22},
        {"name": "MUSIC (FBSS)",  "angle": mus_fbss,                                   "marker": "^", "color": "blue",   "size": 80,  "y": 0.05},
        {"name": "ESPRIT (Raw)",  "angle": results_bpsk.get("ESPRIT (Raw)", np.nan),   "marker": "x", "color": "indigo", "size": 70,  "y": -1.90},
        {"name": "ESPRIT (FBSS)", "angle": results_bpsk.get("ESPRIT (FBSS)", np.nan),  "marker": "x", "color": "black",  "size": 70,  "y": -1.70},
    ]

    # --- Auto-rank DL models by distance to MUSIC(FBSS) ---
    def rank_algos(dl_algos, mus_ref):
        """Sort DL models by abs delta to reference, assign y from top (closest) to bottom."""
        ranked = []
        for a in dl_algos:
            if a["angle"] is not None and not np.isnan(a["angle"]):
                ranked.append((abs(a["angle"] - mus_ref), a))
        ranked.sort(key=lambda x: x[0])
        y_positions = [-0.45, -0.80, -1.15, -1.50, -1.85][:len(ranked)]
        result = []
        for (_, a), y in zip(ranked, y_positions):
            a["y"] = y
            result.append(a)
        return result

    bpsk_dl = [
        {"name": "TransIQ",             "angle": results_bpsk.get("TransIQ", np.nan),             "marker": "o", "color": "red",          "size": 100},
        {"name": "MLP",                  "angle": results_bpsk.get("MLP", np.nan),                  "marker": "v", "color": "brown",        "size": 90},
        {"name": "IQ-ResNet",     "angle": results_bpsk.get("IQ-ResNet", np.nan),     "marker": "*", "color": "deepskyblue",  "size": 100},
        {"name": "CNN-ClassifyIQ",       "angle": results_bpsk.get("CNN-ClassifyIQ", np.nan),       "marker": "s", "color": "darkorange",   "size": 90},
    ]
    bpsk_algos = rank_algos(bpsk_dl, mus_fbss)

    gauss_dl = [
        {"name": "REG-CNN",          "angle": results_gauss.get("REG-CNN", np.nan),           "marker": "o", "color": "orange",       "size": 100},
        {"name": "SPE-CNN",          "angle": results_gauss.get("SPE-CNN", np.nan),           "marker": "o", "color": "gold",         "size": 100},
        {"name": "ViT",              "angle": results_gauss.get("ViT", np.nan),               "marker": "o", "color": "red",          "size": 100},
        {"name": "IQ-ResNet","angle": results_gauss.get("IQ-ResNet", np.nan), "marker": "P", "color": "m",            "size": 100},
        {"name": "Learning-SPICE",   "angle": results_gauss.get("Learning-SPICE", np.nan),    "marker": "D", "color": "purple",       "size": 100},
    ]
    gauss_algos = rank_algos(gauss_dl, mus_fbss)

    # --- Legend elements per group ---
    bpsk_legend = [
        Line2D([0], [0], marker="^", color="w", markeredgecolor="green",       markerfacecolor="green",       markersize=9, label="MUSIC (Raw)"),
        Line2D([0], [0], marker="^", color="w", markeredgecolor="blue",        markerfacecolor="blue",        markersize=9, label="MUSIC (FBSS)"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="indigo",      markeredgewidth=2,             markersize=9, label="ESPRIT (Raw)"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="black",       markeredgewidth=2,             markersize=9, label="ESPRIT (FBSS)"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="red",         markerfacecolor="red",         markersize=9, label="TransIQ"),
        Line2D([0], [0], marker="v", color="w", markeredgecolor="brown",       markerfacecolor="brown",       markersize=9, label="MLP"),
        Line2D([0], [0], marker="*", color="w", markeredgecolor="deepskyblue", markerfacecolor="deepskyblue", markersize=9, label="IQ-ResNet"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor="darkorange",  markerfacecolor="darkorange",  markersize=9, label="CNN-ClassifyIQ"),
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="MUSIC Spectrum"),
    ]
    gauss_legend = [
        Line2D([0], [0], marker="^", color="w", markeredgecolor="green",  markerfacecolor="green",  markersize=9, label="MUSIC (Raw)"),
        Line2D([0], [0], marker="^", color="w", markeredgecolor="blue",   markerfacecolor="blue",   markersize=9, label="MUSIC (FBSS)"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="indigo", markeredgewidth=2,        markersize=9, label="ESPRIT (Raw)"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="black",  markeredgewidth=2,        markersize=9, label="ESPRIT (FBSS)"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="orange",  markerfacecolor="orange",  markersize=9, label="REG-CNN"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="gold",    markerfacecolor="gold",    markersize=9, label="SPE-CNN"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="red",     markerfacecolor="red",     markersize=9, label="ViT"),
        Line2D([0], [0], marker="P", color="w", markeredgecolor="m",       markerfacecolor="m",       markersize=9, label="IQ-ResNet"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="purple",  markerfacecolor="purple",  markersize=9, label="Learning-SPICE"),
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="MUSIC Spectrum"),
    ]

    xlim = (-86, -66)

    # --- BPSK figure ---
    plot_group_standalone(th_fbss, spec_fbss_db, trad_algos + bpsk_algos,
                          "BPSK Domain", xlim,
                          PLOT_DIR_REAL / "test_frame80_BPSK.png", bpsk_legend)

    # --- Gauss figure ---
    plot_group_standalone(th_fbss, spec_fbss_db, trad_algos + gauss_algos,
                          "Gauss Domain", xlim,
                          PLOT_DIR_REAL / "test_frame80_Gauss.png", gauss_legend)


if __name__ == "__main__":
    main()
