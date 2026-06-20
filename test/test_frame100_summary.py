"""
Chapter 6 - 100-frame stability evaluation on real radar data.
Horizontal bar chart (mean + std error bars) per domain, CSV summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GRADUATION_DIR = Path(__file__).resolve().parents[1]
_PROJECT_BASE = _GRADUATION_DIR.parent

for _p in [_PROJECT_BASE, _GRADUATION_DIR, _GRADUATION_DIR / "external"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.real_test import estimate_one_frame, load_models, get_bpsk_specs, get_gauss_specs
from utils.radar_utils import load_and_reshape, process_radar_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RX = 8
REAL_DATA_PATH = _GRADUATION_DIR / "data" / "raw" / "cropped.bin"

PLOT_DIR_REAL = _GRADUATION_DIR / "result" / "plot" / "real"
DATA_DIR_REAL = _GRADUATION_DIR / "result" / "test_data" / "real"
PLOT_DIR_REAL.mkdir(parents=True, exist_ok=True)
DATA_DIR_REAL.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- colors aligned with test_frame80 ----
ALGO_COLORS = {
    "MUSIC (Raw)":       "#2ca02c",  # green
    "MUSIC (FBSS)":      "#1f77b4",  # blue
    "ESPRIT (Raw)":      "#4b0082",  # indigo
    "ESPRIT (FBSS)":     "#333333",  # black
    "TransIQ":           "#d62728",  # red
    "MLP":               "#8b4513",  # brown
    "IQ-ResNet (BPSK)":  "#00bfff",  # deepskyblue
    "CNN-ClassifyIQ":    "#ff8c00",  # darkorange
    "REG-CNN":           "#ff7f0e",  # orange
    "SPE-CNN":           "#ffd700",  # gold
    "ViT":               "#d62728",  # red
    "IQ-ResNet (Gauss)": "#e83e8c",  # magenta
    "Learning-SPICE":    "#9467bd",  # purple
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_real_data():
    if not REAL_DATA_PATH.exists():
        print(f"找不到文件: {REAL_DATA_PATH}")
        return None
    return load_and_reshape(str(REAL_DATA_PATH), conjugate=True)


def run_100_frames(X_all, R_all, specs, group_name, device):
    print(f"\n{'='*60}")
    print(f"  {group_name} — 100帧稳定性测试")
    print(f"{'='*60}")
    loaded = load_models(specs, device=device)
    print(f"  已加载 {len(loaded)} 个模型\n")

    total = min(len(X_all), len(R_all))
    all_records = []
    for i in range(total):
        print(f"  [{group_name}] 第 {i+1}/{total} 帧...")
        results = estimate_one_frame(X_all[i], R_all[i], loaded, device, num_rx=NUM_RX)
        mus_ref = results.get("MUSIC (FBSS)", np.nan)
        for name, angle in results.items():
            delta = angle - mus_ref if not np.isnan(angle) and not np.isnan(mus_ref) else np.nan
            all_records.append({
                "frame_idx": i,
                "algorithm": name,
                "estimate_angle": angle,
                "music_fbss_angle": mus_ref,
                "delta_to_music_fbss": delta,
                "abs_delta_to_music_fbss": abs(delta) if not np.isnan(delta) else np.nan,
                "group": group_name,
            })
    return all_records


def build_summary(df):
    rows = []
    for alg, sub in df.groupby("algorithm"):
        valid = sub.dropna(subset=["estimate_angle"])
        valid_d = sub.dropna(subset=["abs_delta_to_music_fbss"])
        rows.append({
            "algorithm": alg,
            "valid_frames": int(len(valid)),
            "mean_angle": float(valid["estimate_angle"].mean()) if len(valid) else np.nan,
            "std_angle": float(valid["estimate_angle"].std(ddof=0)) if len(valid) else np.nan,
            "mean_abs_delta": float(valid_d["abs_delta_to_music_fbss"].mean()) if len(valid_d) else np.nan,
            "std_abs_delta": float(valid_d["abs_delta_to_music_fbss"].std(ddof=0)) if len(valid_d) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def plot_bar_chart(summary_df, dl_names, trad_names, out_path):
    df = summary_df[summary_df["algorithm"].isin(dl_names + trad_names)].copy()
    df = df.dropna(subset=["mean_abs_delta"])
    df = df.sort_values("mean_abs_delta", ascending=True)

    labels = df["algorithm"].tolist()
    values = df["mean_abs_delta"].tolist()
    errors = df["std_abs_delta"].tolist()
    colors = [ALGO_COLORS.get(lab, "#999999") for lab in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, xerr=errors, color=colors, edgecolor="white",
                   height=0.6, capsize=3, error_kw={"linewidth": 1.2})

    ax.set_xlabel("100-Frame Mean |Δ| and Std to MUSIC (FBSS) (°)", fontsize=15)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=11)

    for bar, val, err in zip(bars, values, errors):
        ax.text(bar.get_width() + err + 0.03, bar.get_y() + bar.get_height()/2,
                f"{val:.3f} ± {err:.3f}°", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"柱状图已保存: {out_path}")


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

    records_bpsk = run_100_frames(X_all, R_all, get_bpsk_specs(), "BPSK Domain", device)
    records_gauss = run_100_frames(X_all, R_all, get_gauss_specs(), "Gauss Domain", device)

    df_all = pd.DataFrame(records_bpsk + records_gauss)
    df_all.to_csv(DATA_DIR_REAL / "real_all_frames_algorithm_results.csv",
                  index=False, encoding="utf-8-sig")

    summary_bpsk = build_summary(df_all[df_all["group"] == "BPSK Domain"])
    summary_gauss = build_summary(df_all[df_all["group"] == "Gauss Domain"])
    summary_all = build_summary(df_all)

    summary_all.to_csv(DATA_DIR_REAL / "real_all_frames_stability_summary.csv",
                       index=False, encoding="utf-8-sig")

    trad_names = ["MUSIC (Raw)", "MUSIC (FBSS)", "ESPRIT (Raw)", "ESPRIT (FBSS)"]
    bpsk_dl = ["TransIQ", "MLP", "IQ-ResNet (BPSK)", "CNN-ClassifyIQ"]
    gauss_dl = ["REG-CNN", "SPE-CNN", "ViT", "IQ-ResNet (Gauss)", "Learning-SPICE"]

    plot_bar_chart(summary_bpsk, bpsk_dl, trad_names,
                   PLOT_DIR_REAL / "real_all_frames_stability_BPSK.png")
    plot_bar_chart(summary_gauss, gauss_dl, trad_names,
                   PLOT_DIR_REAL / "real_all_frames_stability_Gauss.png")


if __name__ == "__main__":
    main()
