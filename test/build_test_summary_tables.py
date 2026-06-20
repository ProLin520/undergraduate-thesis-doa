import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Graduation.utils.metrics_utils import BPSK_DATA_DIR, GAUSS_DATA_DIR, save_csv

OUTPUT_COLUMNS = ["Scenario", "Condition", "Metric", "BestModel", "BestValue", "SecondBestModel", "SecondBestValue", "ConclusionNote"]
BPSK_MODEL_COLUMNS = ["IQ-ResNet", "ViT", "CNN (Regression)", "CNN (Classify)", "MLP", "MUSIC", "iq", "cnn_c", "mlp", "cnn_reg", "vit_base", "vit_transfer", "music"]
GAUSS_MODEL_COLUMNS = ["ViT", "IQ-ResNet", "SPE-CNN", "REG-CNN", "Learning-SPICE", "MUSIC"]

"""
bpsk_single_angle_summary_rho0.csv
bpsk_single_angle_summary_rho1.csv
bpsk_two_sep5_summary.csv
bpsk_two_sep4_snr_key_points.csv
bpsk_two_sep_rho1_delta_key_points.csv
bpsk_seven_cross_metrics.csv

gauss_single_transfer_rmse.csv
gauss_three_random_snr_key_points_rho0.csv
gauss_three_random_snr_key_points_rho1.csv
gauss_three_delta_success_thresholds.csv
gauss_seven_spacing_recall_thresholds_rho0.csv
gauss_seven_spacing_recall_thresholds_rho1.csv
gauss_seven_shifted_center_key_points_rho0.csv
gauss_seven_shifted_center_key_points_rho1.csv
"""

def warn(message):
    print(f"warning: {message}")

def read_csv_any(base_dir, *relative_paths):
    for relative_path in relative_paths:
        path = Path(base_dir) / relative_path
        if path.exists():
            return pd.read_csv(path), path
    warn("missing CSV: " + " or ".join(str(Path(base_dir) / p) for p in relative_paths))
    return None, None


def rank_values(values, higher_is_better=False):
    clean = []
    for model, value in values:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(value) and np.isfinite(value):
            clean.append((str(model), float(value)))
    clean.sort(key=lambda item: item[1], reverse=higher_is_better)
    return clean


def make_row(scenario, condition, metric, ranked, note):
    best = ranked[0] if len(ranked) > 0 else ("", np.nan)
    second = ranked[1] if len(ranked) > 1 else ("", np.nan)
    return {"Scenario": scenario, "Condition": condition, "Metric": metric, "BestModel": best[0], "BestValue": best[1], "SecondBestModel": second[0], "SecondBestValue": second[1], "ConclusionNote": note}


def max_snr_row(df, snr_columns=("snr", "SNR")):
    if df is None or df.empty:
        return None, None
    for column in snr_columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.dropna().empty:
                return None, None
            idx = values.idxmax()
            return df.loc[idx], values.loc[idx]
    return None, None


def nearest_row(df, column, target):
    if df is None or df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    if values.dropna().empty:
        return None
    return df.loc[(values - target).abs().idxmin()]


def add_wide_row(records, scenario, condition, metric, row_or_df, model_columns, higher_is_better=False, note=""):
    if row_or_df is None:
        warn(f"skip {scenario}: no data")
        return
    if isinstance(row_or_df, pd.Series):
        values = [(column, row_or_df[column]) for column in model_columns if column in row_or_df.index]
    else:
        values = [(column, pd.to_numeric(row_or_df[column], errors="coerce").mean()) for column in model_columns if column in row_or_df.columns]
    ranked = rank_values(values, higher_is_better=higher_is_better)
    if not ranked:
        warn(f"skip {scenario}: no usable model values")
        return
    records.append(make_row(scenario, condition, metric, ranked, note))


def add_long_row(records, scenario, condition, metric, df, model_col, value_col, higher_is_better=False, note=""):
    if df is None or df.empty or model_col not in df.columns or value_col not in df.columns:
        warn(f"skip {scenario}: missing {model_col}/{value_col}")
        return
    ranked = rank_values([(row[model_col], row[value_col]) for _, row in df.iterrows()], higher_is_better=higher_is_better)
    if not ranked:
        warn(f"skip {scenario}: no usable model values")
        return
    records.append(make_row(scenario, condition, metric, ranked, note))


def add_max_snr_wide_row(records, scenario, df, model_columns, rho_label, note_prefix=""):
    row, max_snr = max_snr_row(df)
    if row is None:
        warn(f"skip {scenario} {rho_label}: no SNR column")
        return
    note = f"{note_prefix}BestValue是最大SNR={max_snr:g} dB时最优模型的RMSE。"
    add_wide_row(records, scenario, f"{rho_label}, max SNR={max_snr:g} dB", "RMSE at max SNR lower is better", row, model_columns, False, note)


def add_max_snr_long_row(records, scenario, df, rho_label, model_col, value_col, note_prefix=""):
    row, max_snr = max_snr_row(df)
    if row is None:
        warn(f"skip {scenario} {rho_label}: no SNR column")
        return
    subset = df[pd.to_numeric(df["snr" if "snr" in df.columns else "SNR"], errors="coerce") == max_snr]
    note = f"{note_prefix}BestValue是最大SNR={max_snr:g} dB时最优模型的RMSE。"
    add_long_row(records, scenario, f"{rho_label}, max SNR={max_snr:g} dB", "RMSE at max SNR lower is better", subset, model_col, value_col, False, note)


def build_bpsk_table():
    records = []

    single_rho1, _ = read_csv_any(BPSK_DATA_DIR, "bpsk_single_angle_summary_rho1.csv", "SingleSource/bpsk_single_angle_summary_rho1.csv")
    if single_rho1 is not None:
        add_long_row(records, "BPSK单信源", "rho=1, |angle|>=75 edge region", "edge RMSE lower is better", single_rho1, "model", "mean_rmse_edge_abs_ge_75", False, "边缘角区平均RMSE最小的模型。")
        add_max_snr_long_row(records, "BPSK单信源RMSE汇总", single_rho1, "rho=1", "model", "mean_rmse_all")

    single_rho0, _ = read_csv_any(BPSK_DATA_DIR, "bpsk_single_angle_summary_rho0.csv", "SingleSource/bpsk_single_angle_summary_rho0.csv")
    if single_rho0 is not None:
        add_max_snr_long_row(records, "BPSK单信源RMSE汇总", single_rho0, "rho=0", "model", "mean_rmse_all")

    sep5, _ = read_csv_any(BPSK_DATA_DIR, "bpsk_two_sep5_summary.csv", "TwoSource/bpsk_two_sep5_summary.csv")
    if sep5 is not None:
        add_long_row(records, "BPSK双信源sep=5°", "rho=0, snr=0, delta_theta=5", "overall RMSE lower is better", sep5, "model", "rmse_all", False, "二信源固定5°间隔下整体RMSE最小。")

    sep4_snr, _ = read_csv_any(BPSK_DATA_DIR, "bpsk_two_sep4_snr_rmse.csv", "TwoSource/bpsk_two_sep4_snr_rmse.csv")
    if sep4_snr is not None:
        add_max_snr_wide_row(records, "BPSK双信源SNR扫描", sep4_snr, BPSK_MODEL_COLUMNS, "rho=0, DeltaTheta=4°")
        add_max_snr_wide_row(records, "BPSK双信源RMSE汇总", sep4_snr, BPSK_MODEL_COLUMNS, "rho=0")

    delta_rho1, _ = read_csv_any(BPSK_DATA_DIR, "bpsk_two_sep_rho1_delta_rmse.csv", "TwoSource/bpsk_two_sep_rho1_delta_rmse.csv")
    if delta_rho1 is not None:
        row = nearest_row(delta_rho1, "DeltaTheta", 5)
        add_wide_row(records, "BPSK双信源角间隔扫描", "rho=1, nearest DeltaTheta=5°", "RMSE lower is better", row, BPSK_MODEL_COLUMNS, False, "取DeltaTheta最接近5°的一行比较。")
        add_wide_row(records, "BPSK双信源RMSE汇总", "rho=1, fixed SNR, nearest DeltaTheta=5°", "RMSE lower is better", row, BPSK_MODEL_COLUMNS, False, "该CSV没有SNR轴，BestValue取DeltaTheta最接近5°的固定条件RMSE。")

    seven, _ = read_csv_any(BPSK_DATA_DIR, "SevenSource/bpsk_seven_cross_metrics.csv", "bpsk_seven_cross_metrics.csv")
    if seven is not None:
        subset = seven[(seven["train_type"] == "random_train") & (seven["test_type"] == "random_test")] if {"train_type", "test_type"}.issubset(seven.columns) else pd.DataFrame()
        add_long_row(records, "BPSK七信源random-random", "random_train + random_test", "Recall@2° higher is better", subset, "model", "recall_at_2", True, "Recall@2°是逐信源统计，预测角与真实角绝对误差小于2°即召回成功。")
        subset = seven[(seven["train_type"] == "sector_train") & (seven["test_type"] == "random_test")] if {"train_type", "test_type"}.issubset(seven.columns) else pd.DataFrame()
        add_long_row(records, "BPSK七信源sector-random", "sector_train + random_test", "FullSuccess@2° higher is better", subset, "model", "full_success_at_2", True, "FullSuccess@2°要求一个样本7个信源全部误差小于2°。")
        if "rho" in seven.columns:
            for rho_value, group in seven.groupby("rho"):
                add_long_row(records, "BPSK七信源RMSE汇总", f"rho={rho_value:g}, fixed seven-source cross test", "RMSE lower is better", group, "model", "rmse", False, "该CSV没有SNR轴，BestValue取当前七信源交叉测试条件下最优RMSE。")

    save_csv(pd.DataFrame(records, columns=OUTPUT_COLUMNS), BPSK_DATA_DIR / "table_bpsk_main_results.csv")


def build_gauss_table():
    records = []

    transfer, _ = read_csv_any(GAUSS_DATA_DIR, "gauss_single_transfer_rmse.csv", "SingleSource/gauss_single_transfer_rmse.csv")
    if transfer is not None and "improvement_percent" in transfer.columns:
        value = float(pd.to_numeric(transfer["improvement_percent"], errors="coerce").mean())
        records.append(make_row("高斯单信源迁移学习", "rho=1, averaged over SNR", "mean improvement percent higher is better", [("Transfer Model", value)], "平均提升百分比=(base_on_rho1-transfer_on_rho1)/base_on_rho1*100。"))
        row, max_snr = max_snr_row(transfer)
        if row is not None:
            add_wide_row(records, "高斯单信源RMSE汇总", f"rho=0, max SNR={max_snr:g} dB", "RMSE at max SNR lower is better", pd.Series({"Base Model": row["base_on_rho0_rmse"]}), ["Base Model"], False, f"BestValue是最大SNR={max_snr:g} dB时Base模型RMSE。")
            add_wide_row(records, "高斯单信源RMSE汇总", f"rho=1, max SNR={max_snr:g} dB", "RMSE at max SNR lower is better", pd.Series({"Base Model": row["base_on_rho1_rmse"], "Transfer Model": row["transfer_on_rho1_rmse"]}), ["Base Model", "Transfer Model"], False, f"BestValue是最大SNR={max_snr:g} dB时最优模型的RMSE。")

    for rho_tag in ["0.0", "1.0"]:
        three, _ = read_csv_any(GAUSS_DATA_DIR, f"ThreeSource/gauss_three_random_snr_rmse_rho{rho_tag}.csv", f"gauss_three_random_snr_rmse_rho{rho_tag}.csv")
        if three is not None:
            row = nearest_row(three, "snr", 0)
            add_wide_row(records, "高斯三信源随机输入", f"rho={rho_tag}, nearest SNR=0dB", "RMSE lower is better", row, GAUSS_MODEL_COLUMNS, False, "取SNR最接近0dB的一行比较。")
            add_max_snr_wide_row(records, "高斯三信源RMSE汇总", three, GAUSS_MODEL_COLUMNS, f"rho={float(rho_tag):g}")

    thresholds, _ = read_csv_any(GAUSS_DATA_DIR, "ThreeSource/gauss_three_rho1_delta_success_thresholds.csv")
    add_long_row(records, "高斯三信源delta成功率", "rho=1, success threshold=0.9", "min delta lower is better", thresholds, "model", "min_delta", False, "达到0.9成功率所需delta越小，分辨能力越强。")

    for rho_tag in ["0.0", "1.0"]:
        seven, _ = read_csv_any(GAUSS_DATA_DIR, f"SevenSource/gauss_seven_random_snr_rmse_rho{rho_tag}.csv", f"gauss_seven_random_snr_rmse_rho{rho_tag}.csv")
        if seven is not None:
            row = nearest_row(seven, "snr", 5)
            add_wide_row(records, "高斯七信源随机输入", f"rho={rho_tag}, nearest SNR=5dB", "RMSE lower is better", row, GAUSS_MODEL_COLUMNS, False, "取SNR最接近5dB的一行比较。")
            add_max_snr_wide_row(records, "高斯七信源RMSE汇总", seven, GAUSS_MODEL_COLUMNS, f"rho={float(rho_tag):g}")

    for rho_tag in ["0.0", "1.0"]:
        spacing, _ = read_csv_any(GAUSS_DATA_DIR, f"SevenSource/gauss_seven_spacing_recall_thresholds_rho{rho_tag}.csv")
        add_long_row(records, "高斯七信源spacing group", f"rho={rho_tag}, Recall threshold=0.9", "min d lower is better", spacing, "model", "min_d", False, "Recall达到0.9所需d越小，密集信源分辨能力越强。")

    stability_values = {}
    for rho_tag in ["0.0", "1.0"]:
        shifted, _ = read_csv_any(GAUSS_DATA_DIR, f"SevenSource/gauss_seven_shifted_center_key_points_rho{rho_tag}.csv")
        if shifted is not None and {"center", "metric", "model", "value"}.issubset(shifted.columns):
            subset = shifted[(shifted["metric"] == "full_success") & (shifted["center"].abs() == 24)]
            for model, group in subset.groupby("model"):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                if not values.empty:
                    stability_values[f"{model} (rho={rho_tag})"] = float(values.mean() - values.std(ddof=0))
    add_wide_row(records, "高斯七信源shifted-center", "center=±24, d=8/12 key points", "FullSuccess stability score higher is better", pd.Series(stability_values), list(stability_values.keys()), True, "BestValue=mean(full_success)-std(full_success)，兼顾成功率和边缘中心偏移稳定性。")

    save_csv(pd.DataFrame(records, columns=OUTPUT_COLUMNS), GAUSS_DATA_DIR / "table_gauss_main_results.csv")


def main():
    build_bpsk_table()
    build_gauss_table()


if __name__ == "__main__":
    main()
