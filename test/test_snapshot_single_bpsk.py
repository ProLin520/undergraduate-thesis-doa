import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = r"D:\Python\Project\doa_estimation\Graduation"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import convert_to_complex, music_algorithm, scm_embeding
from dl_models.vit_model import VisionTransformer
from Graduation.utils.metrics_utils import BPSK_DATA_DIR, first_x_reach_threshold, save_csv


def format_rho_tag(rho):
    rho_value = float(rho)
    return f"rho{int(rho_value)}" if rho_value.is_integer() else f"rho{str(rho).replace('.', 'p')}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snr_fixed = 0
    rho = 1.0
    snapshots_to_test = [100, 200, 500, 1000, 2000, 4000]

    # The snapshot experiment averages 100 samples for each DOA.
    # Use -90..90 for all 181 DOAs; switch to -85..85 only when reproducing plots that intentionally avoid edge angles.
    test_angles_range = np.arange(-90, 91)

    save_dir = os.path.join(PROJECT_ROOT, "result", "plot", "bpsk", f"M_8_K_1_rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    print("==================================================")
    print(f"Running RMSE vs Snapshots (SNR={snr_fixed}dB, rho={rho})")
    print(f"Angles evaluated: {test_angles_range[0]} to {test_angles_range[-1]} degrees")
    print(f"Results will be saved to: {save_dir}")
    print("==================================================")

    iq_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\SingleSource\IQ_ResNet_SingleSource_rho{rho}.pth"
    vit_weight = (
        rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_1_base\weight_base_bestIQ_rho{rho}.pth"
        if np.isclose(rho, 0.0)
        else rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_1_base_transfer\weight_transfer_bestIQ_rho{rho}.pth"
    )
    cnn_cls_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SingleSource\Model_CNN_ClassificationIQ_8ULA_K1_rho{rho}.h5"
    cnn_reg_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SingleSource\Model_CNN_RegressionIQ_8ULA_K1_rho{rho}.h5"
    mlp_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\SingleSource\Model_MLP_ClassificationIQ_8ULA_K1_rho{rho}.h5"
    test_base_dir = rf"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Single_Source\Test_4000_Rho{rho}"
    data_path = os.path.join(test_base_dir, f"test_data_snr{snr_fixed}.npy")
    label_path = os.path.join(test_base_dir, f"test_labels_snr{snr_fixed}.npy")

    for path in [iq_weight, vit_weight, cnn_cls_weight, cnn_reg_weight, mlp_weight, data_path, label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    iq_model = IQ_ResNet(num_classes=181).to(device)
    iq_model.load_state_dict(torch.load(iq_weight, map_location=device))
    iq_model.eval()

    embedding_dim = 768
    vit_model = VisionTransformer(
        embed_layer=scm_embeding(8, embedding_dim),
        embed_dim=embedding_dim,
        out_dims=181,
        drop_ratio=0,
        attn_drop_ratio=0,
    ).to(device)
    vit_model.load_state_dict(torch.load(vit_weight, map_location=device))
    vit_model.eval()

    model_cnn_cls = tf.keras.models.load_model(cnn_cls_weight, compile=False)
    model_cnn_reg = tf.keras.models.load_model(cnn_reg_weight, compile=False)
    model_mlp_cls = tf.keras.models.load_model(mlp_weight, compile=False)
    print("Models loaded.")

    test_data_full = np.load(data_path)
    test_labels_full = np.load(label_path)
    true_angles_all = test_labels_full if test_labels_full.ndim == 1 else np.argmax(test_labels_full, axis=1) - 90

    # 统一初始化顺序：iq -> vit -> cnn_c -> cnn_r -> mlp -> music
    results_rmse = {key: [] for key in ["iq", "vit", "cnn_c", "cnn_r", "mlp", "music"]}
    csv_raw_records = []
    csv_rmse_records = []
    metric_records = []
    all_abs_err = {key: [] for key in results_rmse}

    for snap in snapshots_to_test:
        sq_errs = {key: [] for key in results_rmse}
        print(f"\nEvaluating snapshots={snap}")

        test_data_snap = test_data_full[:, :, :snap]

        for target_angle in tqdm(test_angles_range, leave=False):
            idx = np.where(true_angles_all == target_angle)[0]
            if len(idx) == 0:
                continue

            x_batch = test_data_snap[idx]
            batch_size, d_iq, snapshots = x_batch.shape
            sensors = d_iq // 2

            x_tensor = torch.tensor(x_batch, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                outputs = iq_model(x_tensor)
                grid_torch = torch.arange(-90, 91, device=device).float()
                pred_iq = torch.sum(torch.softmax(outputs, dim=1) * grid_torch, dim=1).cpu().numpy()

            x_cnn = np.zeros((batch_size, sensors, sensors, 3), dtype=np.float32)
            x_vit = np.zeros((batch_size, 2, sensors, sensors), dtype=np.float32)
            music_preds = []

            for sample_idx in range(batch_size):
                x_complex = convert_to_complex(x_batch[sample_idx])
                cov = (x_complex @ x_complex.conj().T) / snapshots
                music_preds.append(music_algorithm(cov))

                max_cov = np.max(np.abs(cov))
                x_cnn[sample_idx, :, :, 0] = np.real(cov)
                x_cnn[sample_idx, :, :, 1] = np.imag(cov)
                x_cnn[sample_idx, :, :, 2] = np.angle(cov) / np.pi
                if max_cov > 0:
                    x_cnn[sample_idx, :, :, 0] /= max_cov
                    x_cnn[sample_idx, :, :, 1] /= max_cov

                x_vit[sample_idx, 0, :, :] = np.real(cov)
                x_vit[sample_idx, 1, :, :] = np.imag(cov)
                max_vit = np.max(np.abs(x_vit[sample_idx]))
                if max_vit > 0:
                    x_vit[sample_idx] /= max_vit

            grid_np = np.arange(-90, 91)
            pred_cnn_cls = np.sum(model_cnn_cls.predict(x_cnn, verbose=0) * grid_np, axis=1)
            pred_mlp = np.sum(model_mlp_cls.predict(x_cnn, verbose=0) * grid_np, axis=1)
            pred_cnn_reg = model_cnn_reg.predict(x_cnn, verbose=0).flatten() * 90.0

            with torch.no_grad():
                x_vit_tensor = torch.tensor(x_vit, dtype=torch.float32).to(device)
                outputs_vit = vit_model(x_vit_tensor)
                pred_vit = torch.sum(torch.softmax(outputs_vit, dim=1) * grid_torch, dim=1).cpu().numpy()

            err_music = np.abs(target_angle - np.array(music_preds))
            err_iq = np.abs(target_angle - pred_iq)
            err_cnn_cls = np.abs(target_angle - pred_cnn_cls)
            err_cnn_reg = np.abs(target_angle - pred_cnn_reg)
            err_mlp = np.abs(target_angle - pred_mlp)
            err_vit = np.abs(target_angle - pred_vit)

            sq_errs["music"].extend(err_music ** 2)
            sq_errs["iq"].extend(err_iq ** 2)
            sq_errs["cnn_c"].extend(err_cnn_cls ** 2)
            sq_errs["cnn_r"].extend(err_cnn_reg ** 2)
            sq_errs["mlp"].extend(err_mlp ** 2)
            sq_errs["vit"].extend(err_vit ** 2)

            all_abs_err["music"].extend(err_music)
            all_abs_err["iq"].extend(err_iq)
            all_abs_err["cnn_c"].extend(err_cnn_cls)
            all_abs_err["cnn_r"].extend(err_cnn_reg)
            all_abs_err["mlp"].extend(err_mlp)
            all_abs_err["vit"].extend(err_vit)

            # 统一 CSV 保存列名顺序
            for sample_idx in range(batch_size):
                csv_raw_records.append({
                    "Snapshots": snap,
                    "True_Angle": target_angle,
                    "IQ_ResNet": pred_iq[sample_idx],
                    "ViT": pred_vit[sample_idx],
                    "CNN_C": pred_cnn_cls[sample_idx],
                    "CNN_R": pred_cnn_reg[sample_idx],
                    "MLP": pred_mlp[sample_idx],
                    "MUSIC": music_preds[sample_idx],
                })

        for key in results_rmse:
            results_rmse[key].append(np.sqrt(np.mean(sq_errs[key])))
            metric_records.append(
                {"rho": rho, "snr": snr_fixed, "model": key, "snapshots": snap, "rmse": results_rmse[key][-1]})

        # 统一输出顺序
        csv_rmse_records.append({
            "Snapshots": snap,
            "IQ_ResNet": results_rmse["iq"][-1],
            "ViT": results_rmse["vit"][-1],
            "CNN_C": results_rmse["cnn_c"][-1],
            "CNN_R": results_rmse["cnn_r"][-1],
            "MLP": results_rmse["mlp"][-1],
            "MUSIC": results_rmse["music"][-1],
        })

        # 统一控制台打印顺序
        print(
            f"  Result: IQ-ResNet={results_rmse['iq'][-1]:.2f} | "
            f"ViT={results_rmse['vit'][-1]:.2f} | "
            f"CNN-C={results_rmse['cnn_c'][-1]:.2f} | "
            f"CNN-R={results_rmse['cnn_r'][-1]:.2f} | "
            f"MLP={results_rmse['mlp'][-1]:.2f} | "
            f"MUSIC={results_rmse['music'][-1]:.2f}"
        )

    pd.DataFrame(csv_raw_records).to_csv(os.path.join(save_dir, "Raw_Errors.csv"), index=False)
    pd.DataFrame(csv_rmse_records).to_csv(os.path.join(save_dir, "RMSE_Results.csv"), index=False)

    threshold_records = []
    for key in results_rmse:
        first_snapshots = first_x_reach_threshold(snapshots_to_test, [-rmse for rmse in results_rmse[key]], -1.0)
        threshold_records.append(
            {"rho": rho, "snr": snr_fixed, "model": key, "threshold_rmse": 1.0, "first_snapshots": first_snapshots})

    rho_tag = format_rho_tag(rho)
    save_csv(metric_records, BPSK_DATA_DIR / "SingleSource" / f"bpsk_single_snapshot_metrics_{rho_tag}.csv")
    save_csv(threshold_records, BPSK_DATA_DIR / "SingleSource" / f"bpsk_single_snapshot_thresholds_{rho_tag}.csv")

    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0})

    # 统一图表 Legend 命名格式
    styles = {
        "iq": ("md-", "IQ-ResNet"),
        "vit": ("ro-", "ViT"),
        "cnn_c": ("g>-", "CNN (Classify)"),
        "cnn_r": ("yX-", "CNN (Regression)"),
        "mlp": ("c^-", "MLP"),
        "music": ("k*-", "MUSIC"),
    }

    if np.isclose(rho, 1.0):
        fig = plt.figure(figsize=(9, 6))
        bottom_ax = fig.add_axes([0.12, 0.12, 0.78, 0.55])
        top_ax = fig.add_axes([0.12, 0.69, 0.78, 0.25], sharex=bottom_ax)
        plt.setp(top_ax.get_xticklabels(), visible=False)

        bottom_ax.set_ylim(0, 2.5)
        bottom_ax.set_yticks(np.arange(0, 2.75, 0.5))
        top_ax.set_ylim(42, 52)
        top_ax.set_yticks(np.arange(42, 52, 5))

        # 统一画线顺序: iq, vit, cnn_c, cnn_r, mlp
        for key in ["iq", "vit", "cnn_c", "cnn_r", "mlp"]:
            fmt, label = styles[key]
            bottom_ax.plot(snapshots_to_test, results_rmse[key], fmt, label=label, alpha=0.85)

        fmt, label = styles["music"]
        top_ax.plot(snapshots_to_test, results_rmse["music"], fmt, label=label, alpha=0.85)

        for ax in [bottom_ax, top_ax]:
            ax.grid(True, which="major", linestyle="-", alpha=0.3)
            ax.grid(True, which="minor", linestyle=":", alpha=0.2)

        bottom_ax.set_xscale("log")
        bottom_ax.set_xticks(snapshots_to_test)
        bottom_ax.set_xticklabels([str(s) for s in snapshots_to_test])
        bottom_ax.set_xlabel("Number of Snapshots", fontsize=14)
        fig.text(0.06, 0.5, "RMSE (Degrees)", va="center", ha="center", rotation="vertical", fontsize=14)

        handles, labels = [], []
        for ax in [bottom_ax, top_ax]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique = dict(zip(labels, handles))
        bottom_ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=10, framealpha=0.95,
                         edgecolor="0.8")

        bottom_pos = bottom_ax.get_position()
        top_pos = top_ax.get_position()
        x_left = bottom_pos.x0 - 0.02
        y_bottom = bottom_pos.y0 + bottom_pos.height
        y_top = top_pos.y0
        for offset in [0, 0.01]:
            fig.add_artist(
                plt.Line2D([x_left + offset, x_left + offset + 0.015], [y_bottom, y_top], transform=fig.transFigure,
                           color="black", lw=1.5))
            fig.add_artist(plt.Line2D([x_left + offset + 0.005, x_left + offset + 0.02], [y_bottom, y_top],
                                      transform=fig.transFigure, color="black", lw=1.5))

        plt.savefig(os.path.join(save_dir, "RMSE_vs_Snapshots.png"), dpi=300, bbox_inches="tight")
    else:
        plt.figure(figsize=(9, 6))
        # 统一画线顺序
        for key in ["iq", "vit", "cnn_c", "cnn_r", "mlp", "music"]:
            fmt, label = styles[key]
            plt.plot(snapshots_to_test, results_rmse[key], fmt, label=label, alpha=0.85)
        plt.xscale("log")
        plt.xticks(snapshots_to_test, labels=[str(s) for s in snapshots_to_test])
        plt.xlabel("Number of Snapshots", fontsize=14)
        plt.ylabel("RMSE (Degrees)", fontsize=14)
        plt.grid(True, which="major", linestyle="-", alpha=0.3)
        plt.grid(True, which="minor", linestyle=":", alpha=0.2)
        plt.legend(fontsize=10, loc="upper right", framealpha=0.95, edgecolor="0.8")
        plt.savefig(os.path.join(save_dir, "RMSE_vs_Snapshots.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(9, 6))

    def plot_cdf(err_arr, color, label):
        sorted_err = np.sort(err_arr)
        plt.plot(sorted_err, np.arange(len(sorted_err)) / (len(sorted_err) - 1), color=color, linestyle="-",
                 label=label, linewidth=2)

    # 统一 CDF 绘图及 Legend 顺序
    plot_cdf(all_abs_err["iq"], "m", "IQ-ResNet")
    plot_cdf(all_abs_err["vit"], "r", "ViT")
    plot_cdf(all_abs_err["cnn_c"], "g", "CNN (Classify)")
    plot_cdf(all_abs_err["cnn_r"], "y", "CNN (Regression)")
    plot_cdf(all_abs_err["mlp"], "c", "MLP")
    plot_cdf(all_abs_err["music"], "k", "MUSIC")

    plt.xlabel("Absolute Error (Degrees)", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    plt.xlim([0, 15])
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", fontsize=11, edgecolor="0.8")
    plt.savefig(os.path.join(save_dir, "CDF_Comparison.png"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()