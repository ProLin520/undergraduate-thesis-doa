import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import convert_to_complex, music_algorithm, scm_embeding
from dl_models.vit_model import VisionTransformer
from Graduation.utils.metrics_utils import BPSK_DATA_DIR, error_quantiles, save_csv


def parse_labels(test_labels):
    if test_labels.ndim == 2:
        return np.argmax(test_labels, axis=1) - 90
    if test_labels.ndim == 1:
        return test_labels if np.max(test_labels) <= 90 else test_labels - 90
    raise ValueError(f"Unsupported label shape: {test_labels.shape}")


def format_rho_tag(rho):
    rho_value = float(rho)
    return f"rho{int(rho_value)}" if rho_value.is_integer() else f"rho{str(rho).replace('.', 'p')}"


def evaluate_models(args, device=None):
    rho = args.rho
    snr_to_test = args.snr
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(PROJECT_ROOT, "data", "IQ_Data", "Single_Source", f"Test_Rho{rho}")
    data_path = os.path.join(data_dir, f"test_data_snr{snr_to_test}.npy")
    label_path = os.path.join(data_dir, f"test_labels_snr{snr_to_test}.npy")

    test_data = np.load(data_path)
    true_angles_all = parse_labels(np.load(label_path))

    iq_weight = os.path.join(PROJECT_ROOT, "result", "IQ_ResNet", "SingleSource", f"IQ_ResNet_SingleSource_rho{rho}.pth")
    cnn_cls_weight = os.path.join(PROJECT_ROOT, "result", "CNN", "SingleSource", f"Model_CNN_ClassificationIQ_8ULA_K1_rho{rho}.h5")
    cnn_reg_weight = os.path.join(PROJECT_ROOT, "result", "CNN", "SingleSource", f"Model_CNN_RegressionIQ_8ULA_K1_rho{rho}.h5")
    mlp_weight = os.path.join(PROJECT_ROOT, "result", "MLP", "SingleSource", f"Model_MLP_ClassificationIQ_8ULA_K1_rho{rho}.h5")
    vit_reg_base_weight = os.path.join(
        PROJECT_ROOT, "result", "vit", "vit_M_8_k_1_base", f"weight_base_bestIQ_reg_rho{rho}.pth"
    )
    vit_reg_transfer_weight = os.path.join(
        PROJECT_ROOT, "result", "vit", "vit_M_8_k_1_base_transfer", f"weight_transfer_bestIQ_reg_rho{rho}.pth"
    )

    model_iq = IQ_ResNet(num_classes=181).to(device)
    model_iq.load_state_dict(torch.load(iq_weight, map_location=device))
    model_iq.eval()

    model_cnn_cls = tf.keras.models.load_model(cnn_cls_weight, compile=False)
    model_cnn_reg = tf.keras.models.load_model(cnn_reg_weight, compile=False)
    model_mlp_cls = tf.keras.models.load_model(mlp_weight, compile=False)

    embedding_dim = 768
    vit_models = {}

    def load_reg_vit(weight_path):
        model = VisionTransformer(
            embed_layer=scm_embeding(8, embedding_dim),
            embed_dim=embedding_dim,
            out_dims=1,
            drop_ratio=0,
            attn_drop_ratio=0,
        ).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        return model

    if os.path.exists(vit_reg_base_weight):
        vit_models["vit_reg_base"] = load_reg_vit(vit_reg_base_weight)
    if os.path.exists(vit_reg_transfer_weight):
        vit_models["vit_reg_transfer"] = load_reg_vit(vit_reg_transfer_weight)
    if not vit_models:
        raise FileNotFoundError(
            "Regression ViT weight not found. Checked:\n"
            f"  base: {vit_reg_base_weight}\n"
            f"  transfer: {vit_reg_transfer_weight}"
        )

    test_angles_range = np.arange(-85, 86)
    rmses = {k: [] for k in ["iq", "cnn_c", "cnn_r", "mlp", "music", *vit_models.keys()]}
    valid_angles = []
    per_angle_records = []
    summary_errors = {k: [] for k in rmses.keys()}
    summary_rmse_records = {k: [] for k in rmses.keys()}

    def append_prediction_stats(model_name, target_angle, predictions):
        true_values = np.full_like(np.asarray(predictions, dtype=float), target_angle, dtype=float)
        stats = error_quantiles(np.asarray(predictions, dtype=float), true_values)
        per_angle_records.append({"rho": rho, "snr": snr_to_test, "model": model_name, "angle": int(target_angle), "n_samples": int(len(true_values)), "rmse": stats["rmse"], "mae": stats["mae"]})
        summary_errors[model_name].extend(np.abs(np.asarray(predictions, dtype=float) - true_values).tolist())
        summary_rmse_records[model_name].append({"angle": int(target_angle), "rmse": stats["rmse"]})

    def append_music_stats(target_angle, sq_errors):
        abs_errors = np.sqrt(np.asarray(sq_errors, dtype=float))
        rmse = float(np.sqrt(np.mean(np.asarray(sq_errors, dtype=float))))
        per_angle_records.append({"rho": rho, "snr": snr_to_test, "model": "music", "angle": int(target_angle), "n_samples": int(len(abs_errors)), "rmse": rmse, "mae": float(np.mean(abs_errors))})
        summary_errors["music"].extend(abs_errors.tolist())
        summary_rmse_records["music"].append({"angle": int(target_angle), "rmse": rmse})

    for target_angle in test_angles_range:
        idx = np.where(true_angles_all == target_angle)[0]
        if len(idx) == 0:
            continue

        valid_angles.append(target_angle)
        x_batch = test_data[idx]
        batch_size, d_iq, snapshots = x_batch.shape
        sensors = d_iq // 2

        x_tensor = torch.tensor(x_batch, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model_iq(x_tensor)
            pred_iq = torch.argmax(outputs, dim=1).cpu().numpy() - 90
        rmses["iq"].append(np.sqrt(np.mean((target_angle - pred_iq) ** 2)))
        append_prediction_stats("iq", target_angle, pred_iq)

        x_cnn = np.zeros((batch_size, sensors, sensors, 3), dtype=np.float32)
        x_vit = np.zeros((batch_size, 2, sensors, sensors), dtype=np.float32)
        sq_err_music = []

        for sample_idx in range(batch_size):
            x_complex = convert_to_complex(x_batch[sample_idx])
            r_cov = (x_complex @ x_complex.conj().T) / snapshots

            sq_err_music.append((target_angle - music_algorithm(r_cov)) ** 2)

            max_cov = np.max(np.abs(r_cov))
            x_cnn[sample_idx, :, :, 0] = np.real(r_cov)
            x_cnn[sample_idx, :, :, 1] = np.imag(r_cov)
            x_cnn[sample_idx, :, :, 2] = np.angle(r_cov) / np.pi
            if max_cov > 0:
                x_cnn[sample_idx, :, :, 0] /= max_cov
                x_cnn[sample_idx, :, :, 1] /= max_cov

            x_vit[sample_idx, 0, :, :] = np.real(r_cov)
            x_vit[sample_idx, 1, :, :] = np.imag(r_cov)
            max_vit = np.max(np.abs(x_vit[sample_idx]))
            if max_vit > 0:
                x_vit[sample_idx] /= max_vit

        rmses["music"].append(np.sqrt(np.mean(sq_err_music)))
        append_music_stats(target_angle, sq_err_music)

        probs_cnn_cls = model_cnn_cls.predict(x_cnn, verbose=0)
        probs_mlp = model_mlp_cls.predict(x_cnn, verbose=0)
        pred_cnn_cls = np.argmax(probs_cnn_cls, axis=1) - 90
        pred_mlp = np.argmax(probs_mlp, axis=1) - 90
        pred_cnn_reg = model_cnn_reg.predict(x_cnn, verbose=0).flatten() * 90.0

        rmses["cnn_c"].append(np.sqrt(np.mean((target_angle - pred_cnn_cls) ** 2)))
        rmses["mlp"].append(np.sqrt(np.mean((target_angle - pred_mlp) ** 2)))
        rmses["cnn_r"].append(np.sqrt(np.mean((target_angle - pred_cnn_reg) ** 2)))
        append_prediction_stats("cnn_c", target_angle, pred_cnn_cls)
        append_prediction_stats("mlp", target_angle, pred_mlp)
        append_prediction_stats("cnn_r", target_angle, pred_cnn_reg)

        x_vit_tensor = torch.tensor(x_vit, dtype=torch.float32).to(device)
        with torch.no_grad():
            for model_name, model in vit_models.items():
                pred_vit_reg = model(x_vit_tensor).view(-1).cpu().numpy()
                rmses[model_name].append(np.sqrt(np.mean((target_angle - pred_vit_reg) ** 2)))
                append_prediction_stats(model_name, target_angle, pred_vit_reg)

    summary_records = []
    for model_name in rmses.keys():
        rmse_rows = summary_rmse_records[model_name]
        all_abs_errors = np.asarray(summary_errors[model_name], dtype=float)
        if not rmse_rows or all_abs_errors.size == 0:
            continue
        rmse_values = np.asarray([row["rmse"] for row in rmse_rows], dtype=float)
        angle_values = np.asarray([row["angle"] for row in rmse_rows], dtype=float)
        center_mask = np.abs(angle_values) <= 60
        edge_mask = np.abs(angle_values) >= 75
        max_idx = int(np.argmax(rmse_values))
        summary_records.append({"rho": rho, "snr": snr_to_test, "model": model_name, "mean_rmse_all": float(np.mean(rmse_values)), "mean_rmse_center_abs_le_60": float(np.mean(rmse_values[center_mask])) if np.any(center_mask) else np.nan, "mean_rmse_edge_abs_ge_75": float(np.mean(rmse_values[edge_mask])) if np.any(edge_mask) else np.nan, "max_rmse": float(rmse_values[max_idx]), "max_rmse_angle": int(angle_values[max_idx]), "p90_abs_error": float(np.percentile(all_abs_errors, 90))})

    rho_tag = format_rho_tag(rho)
    save_csv(per_angle_records, BPSK_DATA_DIR / "SingleSource" / f"bpsk_single_angle_regvit_per_angle_{rho_tag}.csv")
    save_csv(summary_records, BPSK_DATA_DIR / "SingleSource" / f"bpsk_single_angle_regvit_summary_{rho_tag}.csv")

    return np.array(valid_angles), rmses


def plot_figure_5(angles, rmses, rho=0.0, snr_to_test=0):
    plt.figure(figsize=(8, 6))

    plt.plot(angles, rmses["iq"], "o", color="purple", markerfacecolor="none", label="IQ-ResNet", markersize=6, alpha=0.8)
    plt.plot(angles, rmses["cnn_c"], "bs", markerfacecolor="none", label="CNN (Classify)", markersize=6, alpha=0.7)
    plt.plot(angles, rmses["cnn_r"], "g^", markerfacecolor="none", label="CNN (Regression)", markersize=6, alpha=0.7)
    plt.plot(angles, rmses["mlp"], "v", color="orange", markerfacecolor="none", label="MLP", markersize=6, alpha=0.7)
    plt.plot(angles, rmses["music"], "k*", markerfacecolor="none", label="MUSIC", markersize=6, alpha=0.7)
    if "vit_reg_base" in rmses:
        plt.plot(angles, rmses["vit_reg_base"], "D", color="#4DB6E8", markerfacecolor="none", label="ViT (Regression)", markersize=6, alpha=0.85)
    if "vit_reg_transfer" in rmses:
        plt.plot(angles, rmses["vit_reg_transfer"], "D", color="#4DB6E8", markerfacecolor="none", label="ViT Reg Transfer", markersize=6, alpha=0.85)

    plt.xlim(-85, 85)
    plt.ylim(-0.05, 5.0)
    plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    plt.xlabel("True angle")
    plt.ylabel("RMSE")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)

    save_dir = os.path.join(PROJECT_ROOT, "result", "plot", "bpsk", f"M_8_K_1_rho{rho}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Comparison_single_source_regvit_rho{rho}.png")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--snr", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    angles, rmses = evaluate_models(args, device=args.device)
    plot_figure_5(angles, rmses, rho=args.rho, snr_to_test=args.snr)
