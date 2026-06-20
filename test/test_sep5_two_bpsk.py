import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import torch
from tensorflow.keras.models import load_model

root = Path(__file__).resolve().parents[2]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))

from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding,music_batch_fast
from dl_models.IQ_ResNet_model import IQ_ResNet
from Graduation.utils.metrics_utils import BPSK_DATA_DIR, save_csv


def save_sep5_metrics(labels, preds_dict, alg_names, rho=0.0, snr=0, delta_theta=5):
    labels_sorted = np.sort(labels, axis=1)
    theta1_values = np.unique(labels_sorted[:, 0])
    per_angle_records = []
    summary_records = []

    for name in alg_names:
        if len(preds_dict[name]) == 0:
            continue
        preds_sorted = np.sort(preds_dict[name], axis=1)
        errors = preds_sorted - labels_sorted
        source1_rmse_by_angle = []
        source2_rmse_by_angle = []

        for theta1 in theta1_values:
            idx = labels_sorted[:, 0] == theta1
            theta2 = theta1 + delta_theta
            rmse_source1 = float(np.sqrt(np.mean(errors[idx, 0] ** 2)))
            rmse_source2 = float(np.sqrt(np.mean(errors[idx, 1] ** 2)))
            mean_pred_source1 = float(np.mean(preds_sorted[idx, 0]))
            mean_pred_source2 = float(np.mean(preds_sorted[idx, 1]))
            source1_rmse_by_angle.append((theta1, theta2, rmse_source1))
            source2_rmse_by_angle.append((theta1, theta2, rmse_source2))
            per_angle_records.append({"theta1": int(theta1), "theta2": int(theta2), "model": name, "rmse_source1": rmse_source1, "rmse_source2": rmse_source2, "mean_pred_source1": mean_pred_source1, "mean_pred_source2": mean_pred_source2})

        max_s1_theta1, max_s1_theta2, max_s1_rmse = max(source1_rmse_by_angle, key=lambda item: item[2])
        max_s2_theta1, max_s2_theta2, max_s2_rmse = max(source2_rmse_by_angle, key=lambda item: item[2])
        summary_records.append({"model": name, "rho": rho, "snr": snr, "delta_theta": delta_theta, "max_rmse_source1": float(max_s1_rmse), "max_rmse_source1_theta1": int(max_s1_theta1), "max_rmse_source1_theta2": int(max_s1_theta2), "max_rmse_source2": float(max_s2_rmse), "max_rmse_source2_theta1": int(max_s2_theta1), "max_rmse_source2_theta2": int(max_s2_theta2)})

    per_angle_df = pd.DataFrame(per_angle_records)
    mid_summary_records = []
    if not per_angle_df.empty:
        mid_df = per_angle_df[(per_angle_df["theta1"] > -80) & (per_angle_df["theta1"] < 80) & (per_angle_df["theta2"] > -80) & (per_angle_df["theta2"] < 80)]
        for model, group in mid_df.groupby("model"):
            source1_idx = group["rmse_source1"].idxmax()
            source2_idx = group["rmse_source2"].idxmax()
            source1_row = group.loc[source1_idx]
            source2_row = group.loc[source2_idx]
            mid_summary_records.append({"model": model, "rho": rho, "snr": snr, "delta_theta": delta_theta, "angle_range": "(-80,80)", "n_angle_pairs": int(len(group)), "mean_rmse_source1": float(group["rmse_source1"].mean()), "mean_rmse_source2": float(group["rmse_source2"].mean()), "mean_rmse_two_sources": float(((group["rmse_source1"] + group["rmse_source2"]) / 2).mean()), "max_rmse_source1": float(source1_row["rmse_source1"]), "max_rmse_source1_theta1": int(source1_row["theta1"]), "max_rmse_source1_theta2": int(source1_row["theta2"]), "max_rmse_source2": float(source2_row["rmse_source2"]), "max_rmse_source2_theta1": int(source2_row["theta1"]), "max_rmse_source2_theta2": int(source2_row["theta2"])})

    save_csv(per_angle_df, BPSK_DATA_DIR / "TwoSource" / "bpsk_two_sep5_per_angle.csv")
    save_csv(summary_records, BPSK_DATA_DIR / "TwoSource" / "bpsk_two_sep5_max_rmse_summary.csv")
    save_csv(mid_summary_records, BPSK_DATA_DIR / "TwoSource" / "bpsk_two_sep5_mid_angle_summary.csv")


def load_all_models(device, proj_root):
    print(">>> 正在加载深度学习模型...")
    models = {}

    iq_path = os.path.join(r"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\TwoSource\IQ_ResNet_TwoSource_rho0.0.pth")
    models['IQ-ResNet'] = IQ_ResNet(num_classes=181).to(device)
    if os.path.exists(iq_path): models['IQ-ResNet'].load_state_dict(torch.load(iq_path, map_location=device))
    models['IQ-ResNet'].eval()

    vit_path = os.path.join(r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_2_base\weight_base_Twosource_rho0.0.pth")
    models['ViT'] = VisionTransformer(embed_layer=scm_embeding(8, 768), embed_dim=768, out_dims=181).to(device)
    if os.path.exists(vit_path): models['ViT'].load_state_dict(torch.load(vit_path, map_location=device))
    models['ViT'].eval()

    cnn_r_path = os.path.join(r"D:\Python\Project\doa_estimation\Graduation\result\CNN\TwoSource\Model_CNN_RegressionIQ_TwoSource_rho0.0.h5")
    models['CNN (Regression)'] = load_model(cnn_r_path, compile=False) if os.path.exists(cnn_r_path) else None

    cnn_c_path = os.path.join(r"D:\Python\Project\doa_estimation\Graduation\result\CNN\TwoSource\Model_CNN_ClassifyIQ_TwoSource_rho0.0.h5")
    models['CNN (Classify)'] = load_model(cnn_c_path) if os.path.exists(cnn_c_path) else None

    mlp_path = os.path.join(r"D:\Python\Project\doa_estimation\Graduation\result\MLP\TwoSource\Model_MLP_ClassifyIQ_TwoSource_rho0.0.h5")
    models['MLP'] = load_model(mlp_path) if os.path.exists(mlp_path) else None

    return models


def evaluate_and_plot(proj_root, device):
    rho = 0.0
    test_dir = os.path.join("D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Two_Source\Test_5_deg_Rho0.0")
    print(">>> 加载离线测试数据...")
    data_iq = np.load(os.path.join(test_dir, 'test_data_5_deg.npy'))
    labels = np.load(os.path.join(test_dir, 'test_labels_5_deg.npy'))

    # 【修改处】统一列表顺序，这也是子图(subplots)渲染的顺序
    alg_names = ['IQ-ResNet', 'ViT', 'CNN (Classify)', 'CNN (Regression)', 'MLP', 'MUSIC']
    preds_dict = {name: [] for name in alg_names}
    models = load_all_models(device, proj_root)

    print(">>> 开始进行模型推理与 MUSIC 算法计算 (全链路批处理极速版)...")
    M = 8
    T = data_iq.shape[2]
    batch_size = 500  # 一次性处理 500 个样本

    for i in tqdm(range(0, len(labels), batch_size)):
        # 1. 批量截取数据
        b_iq = data_iq[i:i + batch_size].astype(np.float32)

        # 2. 批量极速转换 SCM
        b_complex = b_iq[:, :M, :] + 1j * b_iq[:, M:, :]
        R_batch = b_complex @ b_complex.conj().transpose(0, 2, 1) / T

        # 批量构造 CNN 格式 (B, 8, 8, 3)
        b_cnn = np.zeros((len(b_iq), M, M, 3), dtype=np.float32)
        b_cnn[:, :, :, 0] = np.real(R_batch)
        b_cnn[:, :, :, 1] = np.imag(R_batch)
        b_cnn[:, :, :, 2] = np.angle(R_batch) / np.pi
        max_cnn = np.max(np.abs(R_batch), axis=(1, 2), keepdims=True)
        b_cnn[:, :, :, 0] /= (max_cnn + 1e-8)
        b_cnn[:, :, :, 1] /= (max_cnn + 1e-8)

        # 批量构造 ViT 格式 (B, 2, 8, 8)
        b_vit = np.zeros((len(b_iq), 2, M, M), dtype=np.float32)
        b_vit[:, 0, :, :] = np.real(R_batch)
        b_vit[:, 1, :, :] = np.imag(R_batch)
        max_vit = np.max(np.abs(b_vit), axis=(1, 2, 3), keepdims=True)
        b_vit /= (max_vit + 1e-8)

        # 1. IQ-ResNet
        with torch.no_grad():
            out_iq = models['IQ-ResNet'](torch.tensor(b_iq).to(device))
            _, top2_idx = torch.topk(out_iq, 2, dim=1)
            preds_dict['IQ-ResNet'].append(np.sort(top2_idx.cpu().numpy() - 90, axis=1))

        # 2. ViT
        with torch.no_grad():
            out_vit = models['ViT'](torch.tensor(b_vit).to(device))
            _, top2_idx_vit = torch.topk(out_vit, 2, dim=1)
            preds_dict['ViT'].append(np.sort(top2_idx_vit.cpu().numpy() - 90, axis=1))

        # 3. CNN (Regression)
        if models['CNN (Regression)']:
            out_cnnr = models['CNN (Regression)'].predict(b_cnn, verbose=0)
            preds_dict['CNN (Regression)'].append(np.sort(out_cnnr * 90.0, axis=1))

        # 4. CNN (Classify)
        if models['CNN (Classify)']:
            out_cnnc = models['CNN (Classify)'].predict(b_cnn, verbose=0)
            preds_dict['CNN (Classify)'].append(np.sort(np.argsort(out_cnnc, axis=1)[:, -2:] - 90, axis=1))

        # 5. MLP
        if models['MLP']:
            out_mlp = models['MLP'].predict(b_cnn, verbose=0)
            preds_dict['MLP'].append(np.sort(np.argsort(out_mlp, axis=1)[:, -2:] - 90, axis=1))

        # 6. MUSIC (批量计算版)
        preds_dict['MUSIC'].append(music_batch_fast(b_complex))

    # 拼接所有批次结果
    for name in alg_names:
        if len(preds_dict[name]) > 0:
            preds_dict[name] = np.concatenate(preds_dict[name], axis=0)

    save_sep5_metrics(labels, preds_dict, alg_names, rho=0.0, snr=0, delta_theta=5)

    angles_source1 = np.arange(-90, 86)
    results = {name: {'mean1': [], 'mean2': [], 'rmse1': [], 'rmse2': []} for name in alg_names}
    samples_per_angle = 100

    for i, a1 in enumerate(angles_source1):
        start, end = i * samples_per_angle, (i + 1) * samples_per_angle
        a2 = a1 + 5

        for name in alg_names:
            if len(preds_dict[name]) == 0: continue
            pred_s1 = preds_dict[name][start:end, 0]
            pred_s2 = preds_dict[name][start:end, 1]

            results[name]['mean1'].append(np.mean(pred_s1))
            results[name]['mean2'].append(np.mean(pred_s2))
            results[name]['rmse1'].append(np.sqrt(np.mean((pred_s1 - a1) ** 2)))
            results[name]['rmse2'].append(np.sqrt(np.mean((pred_s2 - a2) ** 2)))

    # 分别画两张大图
    plot_2x3_grid(angles_source1, results, alg_names, metric='mean', rho=rho)
    plot_2x3_grid(angles_source1, results, alg_names, metric='rmse', rho=rho)


def plot_2x3_grid(angles, results, alg_names, metric='mean', rho=0.0):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs_flat = axs.flatten()

    for i, name in enumerate(alg_names):
        ax = axs_flat[i]
        if len(results[name]['mean1']) == 0:
            ax.set_title(f"{name} (Not Found)")
            continue

        if metric == 'mean':
            ax.plot(angles, results[name]['mean1'], 'ro', markersize=3, label='Pred $\\theta_1$', alpha=0.6)
            ax.plot(angles, results[name]['mean2'], 'bs', markersize=3, label='Pred $\\theta_2$', alpha=0.6)
            ax.plot(angles, angles, 'k--', label='True $\\theta_1$', alpha=0.5)
            ax.plot(angles, angles + 5, 'k:', label='True $\\theta_2$', alpha=0.5)
            ax.set_ylabel('Predicted Angle')
        else:
            ax.plot(angles, results[name]['rmse1'], 'ro', markersize=3, label='RMSE $\\theta_1$', alpha=0.6)
            ax.plot(angles, results[name]['rmse2'], 'bs', markersize=3, label='RMSE $\\theta_2$', alpha=0.6)
            ax.set_ylim(0, 20)
            ax.set_ylabel('RMSE')

        ax.set_title(name, fontsize=14)
        ax.set_xlabel('True Angle $\\theta_1$ (degrees)')
        ax.legend()
        ax.grid(True)

        save_dir = rf"D:\Python\Project\doa_estimation\Graduation\result\plot\bpsk\M_8_K_2_rho{rho}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"Two Source separate 5 degree {metric.upper()}.png")
        plt.savefig(save_path, dpi=500, bbox_inches='tight')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_path = str(Path(__file__).resolve().parents[2])
    evaluate_and_plot(root_path, dev)