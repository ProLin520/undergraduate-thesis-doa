import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from tqdm import tqdm

PROJECT_ROOT = r"D:\Python\Project\doa_estimation\Graduation"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import scm_embeding
from dl_models.vit_model import VisionTransformer
from Graduation.utils.metrics_utils import BPSK_DATA_DIR, full_success_at, recall_at, rmse_deg, save_csv

MODEL_NAMES = ["IQ-ResNet", "ViT", "CNN (Classify)", "MLP"]


def get_weight_paths(train_type, rho):
    if train_type == "random_train":
        return {
            "IQ-ResNet": rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\SevenSource\IQ_ResNet_SevenSource_rho{rho}.pth",
            "ViT": rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base\weight_base_SevenSource_IQ_rho{rho}.pth",
            "CNN (Classify)": rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\Model_CNN_ClassifyIQ_SevenSource_rho{rho}.h5",
            "MLP": rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\SevenSource\Model_MLP_ClassifyIQ_SevenSource_rho{rho}.h5",
        }
    if train_type == "sector_train":
        return {
            "IQ-ResNet": rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\SevenSource\IQ_ResNet_SevenSource_Article_rho{rho}.pth",
            "ViT": rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base\weight_base_SevenSource_ArticleIQ_rho{rho}.pth",
            "CNN (Classify)": rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\Model_CNN_ClassifyIQ_SevenSource_Article_rho{rho}.h5",
            "MLP": rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\SevenSource\Model_MLP_ClassifyIQ_SevenSource_Article_rho{rho}.h5",
        }
    raise ValueError(f"Unsupported train_type: {train_type}")


def get_test_data_paths(test_type, rho):
    if test_type == "random_test":
        data_path = rf"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Seven_Source\Seven_Source_Random_test_Rho{rho}"
        return data_path, "random_iq_data.npy", "random_true_angles.npy"
    if test_type == "sector_test":
        data_path = rf"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Seven_Source\Seven_Source_Article_test_Rho{rho}"
        return data_path, "sector_iq_data.npy", "sector_true_angles.npy"
    raise ValueError(f"Unsupported test_type: {test_type}")


def get_figure_name(train_type, test_type):
    if train_type == "random_train" and test_type == "random_test":
        return "Random_Random.png"
    if train_type == "sector_train" and test_type == "random_test":
        return "Random_Sector.png"
    if train_type == "sector_train" and test_type == "sector_test":
        return "Sector_Sector.png"
    if train_type == "random_train" and test_type == "sector_test":
        return "Sector_Random.png"
    raise ValueError(f"Unsupported figure combination: {train_type}, {test_type}")


def remove_existing_records(path, key_columns, new_records):
    if not os.path.exists(path):
        return
    old_df = pd.read_csv(path)
    if old_df.empty:
        return
    new_keys = {tuple(record[column] for column in key_columns) for record in new_records}
    keep_mask = ~old_df.apply(lambda row: tuple(row[column] for column in key_columns) in new_keys, axis=1)
    save_csv(old_df[keep_mask], path)


def format_data_for_models(iq_mat_batch, sensors=8, snapshots=1024):
    batch_size = iq_mat_batch.shape[0]
    x_iq = torch.tensor(iq_mat_batch, dtype=torch.float32).unsqueeze(1)
    x_vit_list, x_cnn_list = [], []

    for idx in range(batch_size):
        iq_mat = iq_mat_batch[idx]
        x_complex = iq_mat[:sensors, :] + 1j * iq_mat[sensors:, :]
        cov = (x_complex @ x_complex.conj().T) / snapshots

        x_vit = np.zeros((2, sensors, sensors), dtype=np.float32)
        x_vit[0, :, :] = np.real(cov)
        x_vit[1, :, :] = np.imag(cov)
        max_vit = np.max(np.abs(x_vit))
        if max_vit > 1e-8:
            x_vit /= max_vit
        x_vit_list.append(x_vit)

        x_cnn = np.zeros((sensors, sensors, 3), dtype=np.float32)
        x_cnn[:, :, 0] = np.real(cov)
        x_cnn[:, :, 1] = np.imag(cov)
        x_cnn[:, :, 2] = np.angle(cov) / np.pi
        max_cov = np.max(np.abs(cov))
        if max_cov > 1e-8:
            x_cnn[:, :, 0] /= max_cov
            x_cnn[:, :, 1] /= max_cov
        x_cnn_list.append(x_cnn)

    return x_iq, torch.tensor(np.array(x_vit_list), dtype=torch.float32), np.array(x_cnn_list, dtype=np.float32)


def load_models(train_type, rho, device):
    weight_paths = get_weight_paths(train_type, rho)
    model_iq = IQ_ResNet(num_classes=181).to(device)
    model_iq.load_state_dict(torch.load(weight_paths["IQ-ResNet"], map_location=device))
    model_iq.eval()
    model_vit = VisionTransformer(embed_layer=scm_embeding(8, 768), embed_dim=768, out_dims=181).to(device)
    model_vit.load_state_dict(torch.load(weight_paths["ViT"], map_location=device))
    model_vit.eval()
    model_cnn = load_model(weight_paths["CNN (Classify)"])
    model_mlp = load_model(weight_paths["MLP"])
    return {"IQ-ResNet": model_iq, "ViT": model_vit, "CNN (Classify)": model_cnn, "MLP": model_mlp}


def predict_top7(models, iq_data, device, desc):
    x_iq, x_vit, x_cnn = format_data_for_models(iq_data)
    results = {model_name: [] for model_name in MODEL_NAMES}
    batch_size = 64

    for start in tqdm(range(0, len(iq_data), batch_size), desc=desc):
        end = min(start + batch_size, len(iq_data))
        batch_iq = x_iq[start:end].to(device)
        batch_vit = x_vit[start:end].to(device)
        with torch.no_grad():
            pred_iq = models["IQ-ResNet"](batch_iq)
            pred_vit = models["ViT"](batch_vit)

        batch_cnn = x_cnn[start:end]
        pred_cnn = models["CNN (Classify)"].predict(batch_cnn, verbose=0)
        pred_mlp = models["MLP"].predict(batch_cnn, verbose=0)

        for idx in range(end - start):
            results["IQ-ResNet"].append(np.sort(torch.topk(pred_iq[idx:idx + 1], 7, dim=1)[1].cpu().numpy()[0] - 90))
            results["ViT"].append(np.sort(torch.topk(pred_vit[idx:idx + 1], 7, dim=1)[1].cpu().numpy()[0] - 90))
            results["CNN (Classify)"].append(np.sort(np.argsort(pred_cnn[idx])[-7:] - 90))
            results["MLP"].append(np.sort(np.argsort(pred_mlp[idx])[-7:] - 90))

    return {model_name: np.array(preds) for model_name, preds in results.items()}


def save_cross_metrics(train_type, test_type, rho, true_labels, results):
    metric_records = []
    sample_records = []
    true_sorted = np.sort(true_labels, axis=1)

    for model_name in MODEL_NAMES:
        pred_sorted = np.sort(results[model_name], axis=1)
        metric_records.append({"train_type": train_type, "test_type": test_type, "model": model_name, "rho": rho, "n_samples": int(len(true_sorted)), "rmse": rmse_deg(pred_sorted, true_sorted), "recall_at_2": recall_at(pred_sorted, true_sorted, 2), "full_success_at_2": full_success_at(pred_sorted, true_sorted, 2)})
        for sample_id in range(min(50, len(true_sorted))):
            sample_records.append({"train_type": train_type, "test_type": test_type, "model": model_name, "rho": rho, "sample_id": sample_id, "true_angles": " ".join(map(str, true_sorted[sample_id].astype(int).tolist())), "pred_angles": " ".join(map(str, pred_sorted[sample_id].astype(int).tolist()))})

    output_dir = BPSK_DATA_DIR / "SevenSource"
    metrics_path = output_dir / "bpsk_seven_cross_metrics.csv"
    sample_path = output_dir / "bpsk_seven_cross_predictions_sample.csv"
    remove_existing_records(metrics_path, ["train_type", "test_type", "model", "rho"], metric_records)
    remove_existing_records(sample_path, ["train_type", "test_type", "model", "rho"], sample_records)
    old_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    old_samples = pd.read_csv(sample_path) if os.path.exists(sample_path) else pd.DataFrame()
    save_csv(pd.concat([old_metrics, pd.DataFrame(metric_records)], ignore_index=True), metrics_path)
    save_csv(pd.concat([old_samples, pd.DataFrame(sample_records)], ignore_index=True), sample_path)


def plot_random_results(true_labels, results, save_fig_path):
    display_true = true_labels
    display_results = results
    max_display_points = 200
    if len(results["IQ-ResNet"]) > max_display_points:
        rng = np.random.default_rng(seed=42)
        sample_idx = np.sort(rng.choice(len(results["IQ-ResNet"]), size=max_display_points, replace=False))
        display_true = true_labels[sample_idx]
        display_results = {model_name: preds[sample_idx] for model_name, preds in results.items()}

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    soft_colors = ["#8C6BB1", "#6BAED6", "#80C7A1", "#F9C55C", "#F79D4E", "#E8836C", "#D95F5F"]
    markers = {"IQ-ResNet": "o", "ViT": "s", "CNN (Classify)": "D", "MLP": "x"}

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[idx]
        ax.set_box_aspect(1)
        preds = display_results[model_name]
        marker_style = markers[model_name]
        ax.plot([-90, 90], [-90, 90], color="black", linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)

        for src_idx in range(7):
            if marker_style == "x":
                ax.scatter(display_true[:, src_idx], preds[:, src_idx], marker=marker_style, color=soft_colors[src_idx], s=12, alpha=0.6, zorder=2, label=rf"$\theta_{src_idx + 1}$")
            else:
                ax.scatter(display_true[:, src_idx], preds[:, src_idx], marker=marker_style, facecolors="none", edgecolors=soft_colors[src_idx], s=12, alpha=0.7, linewidth=1.2, zorder=2, label=rf"$\theta_{src_idx + 1}$")

        ax.set_title(f"{model_name} Random Test", fontsize=14, fontweight="bold")
        ax.set_xlabel("True DOA (Degrees)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Estimated DOA (Degrees)", fontsize=12)
        ax.set_xlim([-95, 95])
        ax.set_ylim([-95, 95])
        ax.set_xticks(np.arange(-80, 81, 40))
        ax.set_yticks(np.arange(-80, 81, 40))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10, ncol=2, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sector_results(true_labels, results, save_fig_path):
    x_axis_vals = true_labels[:, 5]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors = ["#800080", "#0000FF", "#00FFFF", "#008000", "#FFD700", "#FFA500", "#FF0000"]
    markers = {"IQ-ResNet": "o", "ViT": "s", "CNN (Classify)": "D", "MLP": "x"}

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[idx]
        ax.set_box_aspect(1)
        preds = results[model_name]
        marker_style = markers[model_name]

        for src_idx in range(7):
            ax.plot(x_axis_vals, true_labels[:, src_idx], color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1)
            if marker_style == "x":
                ax.scatter(x_axis_vals, preds[:, src_idx], marker=marker_style, color=colors[src_idx], s=25, alpha=0.8, zorder=2, label=rf"$\theta_{src_idx + 1}$")
            else:
                ax.scatter(x_axis_vals, preds[:, src_idx], marker=marker_style, facecolors="none", edgecolors=colors[src_idx], s=25, alpha=0.8, zorder=2, label=rf"$\theta_{src_idx + 1}$")

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("True Angle of Source 6", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Estimated DOA", fontsize=12)
        ax.set_xlim([5, 85])
        ax.set_ylim([-95, 95])
        ax.set_yticks(np.arange(-80, 81, 40))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=10, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_plot(train_type, test_type, rho, device, model_cache):
    if train_type not in model_cache:
        model_cache[train_type] = load_models(train_type, rho, device)

    data_path, iq_file, angle_file = get_test_data_paths(test_type, rho)
    iq_data = np.load(os.path.join(data_path, iq_file))
    true_labels = np.load(os.path.join(data_path, angle_file))
    results = predict_top7(model_cache[train_type], iq_data, device, desc=f"{get_figure_name(train_type, test_type)}")
    save_cross_metrics(train_type, test_type, rho, true_labels, results)

    save_fig_path = os.path.join(PROJECT_ROOT, "result", "plot", "bpsk", f"M_8_K_7_rho{rho}", get_figure_name(train_type, test_type))
    if test_type == "random_test":
        plot_random_results(true_labels, results, save_fig_path)
    else:
        plot_sector_results(true_labels, results, save_fig_path)


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as exc:
            print(exc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rho = 0.0
    model_cache = {}
    evaluate_and_plot("random_train", "random_test", rho, device, model_cache)
    evaluate_and_plot("sector_train", "random_test", rho, device, model_cache)
    evaluate_and_plot("sector_train", "sector_test", rho, device, model_cache)
    evaluate_and_plot("random_train", "sector_test", rho, device, model_cache)


if __name__ == "__main__":
    main()
