import gc
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(exc)

root = Path(__file__).resolve().parents[2]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))

from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import music_batch_fast, scm_embeding
from dl_models.vit_model import VisionTransformer
from Graduation.utils.metrics_utils import BPSK_DATA_DIR, nearest_value, save_csv

PLOT_STYLES = {
    "IQ-ResNet": {"fmt": "md-", "label": "IQ-ResNet"},
    "CNN (Classify)": {"fmt": "g>-", "label": "CNN (Classify)"},
    "CNN (Regression)": {"fmt": "yX-", "label": "CNN (Regression)"},
    "MLP": {"fmt": "c^-", "label": "MLP"},
    "MUSIC": {"fmt": "k*-", "label": "MUSIC"},
    "ViT": {"fmt": "ro-", "label": "ViT"},
}


def load_all_models(device, rho):
    models = {}
    models["IQ-ResNet"] = IQ_ResNet(num_classes=181).to(device)
    iq_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\TwoSource\IQ_ResNet_TwoSource_rho{rho}.pth"
    models["IQ-ResNet"].load_state_dict(torch.load(iq_weight, map_location=device))
    models["IQ-ResNet"].eval()

    vit_weight = (
        rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_2_base\weight_base_Twosource_rho{rho}.pth"
        if np.isclose(rho, 0.0)
        else rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_2_transfer\weight_transfer_TwoSource_rho{rho}.pth"
    )
    models["ViT"] = VisionTransformer(embed_layer=scm_embeding(8, 768), embed_dim=768, out_dims=181).to(device)
    models["ViT"].load_state_dict(torch.load(vit_weight, map_location=device))
    models["ViT"].eval()

    cnn_reg_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\TwoSource\Model_CNN_RegressionIQ_TwoSource_rho{rho}.h5"
    cnn_cls_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\TwoSource\Model_CNN_ClassifyIQ_TwoSource_rho{rho}.h5"
    mlp_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\TwoSource\Model_MLP_ClassifyIQ_TwoSource_rho{rho}.h5"
    models["CNN (Regression)"] = load_model(cnn_reg_weight, compile=False)
    models["CNN (Classify)"] = load_model(cnn_cls_weight, compile=False)
    models["MLP"] = load_model(mlp_weight, compile=False)
    return models


def evaluate_angle_sep_performance(proj_root, device):
    rho = 1.0
    test_dir = rf"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Two_Source\Test_sep_Rho{rho}"
    save_dir = os.path.join(proj_root, "Graduation", "result", "plot", "bpsk", f"M_8_K_2_rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    delta_thetas = [1, 2, 5, 8, 10, 15, 20]
    alg_names = ["IQ-ResNet", "ViT", "CNN (Regression)", "CNN (Classify)", "MLP", "MUSIC"]
    dt_rmse_results = {name: [] for name in alg_names}
    all_abs_err = {name: [] for name in alg_names}
    csv_rmse_records = []
    models = load_all_models(device, rho)

    sensors = 8
    batch_size = 500
    for delta_theta in delta_thetas:
        data_iq = np.load(os.path.join(test_dir, f"test_iq_data_delta{delta_theta}.npy"))
        labels = np.load(os.path.join(test_dir, f"test_labels_delta{delta_theta}.npy"))
        snapshots = data_iq.shape[2]
        preds_dict = {name: [] for name in alg_names}

        for start in tqdm(range(0, len(labels), batch_size), leave=False):
            batch_iq = data_iq[start:start + batch_size].astype(np.float32)
            batch_complex = batch_iq[:, :sensors, :] + 1j * batch_iq[:, sensors:, :]
            cov_batch = batch_complex @ batch_complex.conj().transpose(0, 2, 1) / snapshots

            batch_cnn = np.zeros((len(batch_iq), sensors, sensors, 3), dtype=np.float32)
            batch_cnn[:, :, :, 0], batch_cnn[:, :, :, 1], batch_cnn[:, :, :, 2] = np.real(cov_batch), np.imag(cov_batch), np.angle(cov_batch) / np.pi
            max_cnn = np.max(np.abs(cov_batch), axis=(1, 2), keepdims=True)
            batch_cnn[:, :, :, 0] /= max_cnn + 1e-8
            batch_cnn[:, :, :, 1] /= max_cnn + 1e-8

            batch_vit = np.zeros((len(batch_iq), 2, sensors, sensors), dtype=np.float32)
            batch_vit[:, 0, :, :], batch_vit[:, 1, :, :] = np.real(cov_batch), np.imag(cov_batch)
            batch_vit /= np.max(np.abs(batch_vit), axis=(1, 2, 3), keepdims=True) + 1e-8

            with torch.no_grad():
                out_iq = models["IQ-ResNet"](torch.tensor(batch_iq).to(device))
                _, top2_idx = torch.topk(out_iq, 2, dim=1)
                preds_dict["IQ-ResNet"].append(np.sort(top2_idx.cpu().numpy() - 90, axis=1))

                out_vit = models["ViT"](torch.tensor(batch_vit).to(device))
                _, top2_idx_vit = torch.topk(out_vit, 2, dim=1)
                preds_dict["ViT"].append(np.sort(top2_idx_vit.cpu().numpy() - 90, axis=1))

            out_cnn_reg = models["CNN (Regression)"].predict(batch_cnn, verbose=0)
            preds_dict["CNN (Regression)"].append(np.sort(out_cnn_reg * 90.0, axis=1))

            out_cnn_cls = models["CNN (Classify)"].predict(batch_cnn, verbose=0)
            preds_dict["CNN (Classify)"].append(np.sort(np.argsort(out_cnn_cls, axis=1)[:, -2:] - 90, axis=1))

            out_mlp = models["MLP"].predict(batch_cnn, verbose=0)
            preds_dict["MLP"].append(np.sort(np.argsort(out_mlp, axis=1)[:, -2:] - 90, axis=1))

            preds_dict["MUSIC"].append(music_batch_fast(batch_complex))

        current_row = {"Delta_Theta": delta_theta}
        for name in alg_names:
            all_preds = np.concatenate(preds_dict[name], axis=0)
            diff = np.abs(all_preds - labels)
            diff = np.minimum(diff, 180.0 - diff)
            all_abs_err[name].extend(diff.flatten())
            current_rmse = np.sqrt(np.mean(diff ** 2))
            dt_rmse_results[name].append(current_rmse)
            current_row[name] = current_rmse

        csv_rmse_records.append(current_row)
        del data_iq, labels, preds_dict
        gc.collect()

    pd.DataFrame(csv_rmse_records).to_csv(os.path.join(save_dir, "RMSE_Results_vs_AngleSep.csv"), index=False)
    bpsk_rmse_records = [{"DeltaTheta": row["Delta_Theta"], **{name: row[name] for name in alg_names}} for row in csv_rmse_records]
    delta_values = [row["DeltaTheta"] for row in bpsk_rmse_records]
    bpsk_key_records = []
    for target_delta in [1, 5, 10]:
        nearest_delta = nearest_value(delta_values, target_delta)
        bpsk_key_records.append(next(row for row in bpsk_rmse_records if row["DeltaTheta"] == nearest_delta))
    save_csv(bpsk_rmse_records, BPSK_DATA_DIR / "TwoSource" / "bpsk_two_sep_rho1_delta_rmse.csv")
    save_csv(bpsk_key_records, BPSK_DATA_DIR / "TwoSource" / "bpsk_two_sep_rho1_delta_key_points.csv")

    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0})
    plt.figure(figsize=(8, 6))
    for name in alg_names:
        plt.plot(delta_thetas, np.array(dt_rmse_results[name]), PLOT_STYLES[name]["fmt"], label=PLOT_STYLES[name]["label"], markersize=8, alpha=0.85)
    # plt.title(f"DOA Performance vs. Angle Separation (SNR=5dB, Rho={rho})")
    plt.xlabel(r"Angle Separation $\Delta\theta$ (degrees)")
    plt.ylabel("RMSE")
    plt.xticks(delta_thetas)
    plt.ylim([-0.5, 10])
    plt.legend(fontsize=11, loc="upper right", edgecolor="0.8")
    plt.grid(True, which="major", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "RMSE_vs_AngleSep_Linear.png"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = str(Path(__file__).resolve().parents[2])
    evaluate_angle_sep_performance(project_root, device)
