import sys
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 路径对齐 =================
root = Path(__file__).resolve().parents[2]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path = [str(ext_lib), str(root)] + sys.path

from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, nearest_value, save_csv
from data.data_create.file_dataloader import file_array_Dataloader
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from dl_models.IQ_ResNet_model import IQ_ResNet
from models.dl_model.CNN.literature_CNN import std_CNN
from dl_models.CNN_model import CNN_Regression
from dl_models.MLP import LearningSPICE_SP_MLP, scm_to_vec72

from data.data_create.signal_datasets90 import ULA_dataset
from models.subspace_model.music import Music
from dl_models.embeding_layer import get_continuous_angle,music_algorithm


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    snr_list = [-20, -15, -10, -5, 0, 5, 10]
    rho = 0.0
    save_dir = os.path.join(root, 'Graduation', 'result', 'plot', 'gauss', f'M_8_K_1_rho{rho}')
    os.makedirs(save_dir, exist_ok=True)

    # ================= 1. 初始化并严格加载 =================
    print("\n>>> 正在加载模型权重...")
    model_vit = VisionTransformer(embed_layer=scm_embeding(8, 768), embed_dim=768, out_dims=1).to(device)
    model_iq = IQ_ResNet(num_classes=181).to(device)
    model_spe = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    modle_reg = CNN_Regression().to(device)
    model_spice = LearningSPICE_SP_MLP(M=8, out_dim=181).to(device)

    models_info = [
        # ("ViT", model_vit,
        #  os.path.join(root, "Graduation", "result", "vit", "vit_M_8_k_1_base", "weight_base_best_snr-2010.pth")),
        ("ViT", model_vit,
         os.path.join(root, "Graduation", "result", "vit", "vit_M_8_k_1_base_transfer", f"weight_transfer_best_snr-2010_rho{rho}.pth")),
        ("IQ-ResNet", model_iq,
         os.path.join(root, "Graduation", "result", "IQ_ResNet", "SingleSource", f"IQ_ResNet_Gaussian_rho{rho}.pth")),
        ("SPE-CNN", model_spe,
         os.path.join(root, "Graduation", "result", "CNN", "SingleSource", f"SPE_CNN_Gaussian_8ULA_K1_rho{rho}.pth")),
        ("REG-CNN", modle_reg,
         os.path.join(root, "Graduation", "result", "CNN", "SingleSource", f"CNN_Regression_Gaussian_rho{rho}.pth")),
        ("Learning-SPICE", model_spice,
         os.path.join(root, "Graduation", "result", "MLP", "SingleSource", f"LearningSPICE_Gaussian_rho{rho}.pth"))
    ]

    for name, model, path in models_info:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"[正常] {name} 权重已加载")
        else:
            print(f"[严重警告] 未找到 {name} 权重文件！检查路径 -> {path}")
        model.eval()

    # ================= 2. 初始化原生 MUSIC 模型 =================
    ideal_dataset = ULA_dataset(M=8, rho=0.0)

    def get_ideal_steer_vector(doas, in_f=None):
        return ideal_dataset.get_A(doas, in_f, array_imperfection=False)

    music_estimator = Music(get_ideal_steer_vector, start=-90, end=90, step=0.1)

    dataset_base_path = fr"D:\Python\Project\doa_estimation\Graduation\data\ViT\ViT_M_8_K_1\M_8_k_1_test1_rho{rho}"

    csv_raw_records, csv_rmse_records = [], []
    all_err = {k: [] for k in ['vit', 'iq', 'spe', 'reg', 'spice', 'music']}

    for snr in snr_list:
        folder_path = os.path.join(dataset_base_path, f"test_random_input_snr_{snr}.npz")
        if not os.path.exists(folder_path):
            print(f"[警告] 未找到 {folder_path}，跳过该 SNR={snr}")
            continue

        loader = file_array_Dataloader(folder_path, batch_size=128, shuffle=False, load_style='torch', input_type='y_t',
                                       output_type='doa')

        sq_err = {k: [] for k in all_err.keys()}
        rmse_results = {'ViT': [], 'IQ-ResNet': [], 'SPE-CNN': [], 'REG-CNN': [], 'Learning-SPICE': [], 'MUSIC': []}

        print(f"\n>>> 正在进行 6 大算法联合评估 SNR = {snr} dB ...")

        with torch.no_grad():
            for inputs_complex, labels_doa in tqdm(loader, leave=False):
                labels_doa = labels_doa.to(device).view(-1)
                B, M, T = inputs_complex.shape
                inputs_complex = inputs_complex.to(device)

                # ================= 数据预处理 =================
                R_complex = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T

                # 1. 2通道模型
                X_2ch = torch.zeros(B, 2, M, M, device=device)
                X_2ch[:, 0, :, :] = R_complex.real
                X_2ch[:, 1, :, :] = R_complex.imag
                max_val_2ch = torch.max(torch.abs(X_2ch.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1, 1)
                X_2ch = X_2ch / torch.clamp(max_val_2ch, min=1e-12)

                # 2. 3通道模型
                X_spe = torch.zeros(B, 3, M, M, device=device)
                X_spe[:, 0, :, :] = R_complex.real
                X_spe[:, 1, :, :] = R_complex.imag
                X_spe[:, 2, :, :] = R_complex.angle() / torch.pi
                max_spe = torch.max(torch.abs(R_complex.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
                X_spe[:, 0, :, :] /= torch.clamp(max_spe, min=1e-12)
                X_spe[:, 1, :, :] /= torch.clamp(max_spe, min=1e-12)

                # 3. IQ-ResNet
                inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
                rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
                X_iq = inputs_iq / (rms_val + 1e-8)

                # 4. Learning-SPICE
                max_val_spice = torch.max(torch.abs(R_complex.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
                R_noisy_spice = R_complex / torch.clamp(max_val_spice, min=1e-12)
                X_vec72 = scm_to_vec72(R_noisy_spice)

                # ================= 推理 =================
                pred_vit = model_vit(X_2ch).squeeze(-1)
                pred_reg = modle_reg(X_2ch).squeeze(-1)

                out_iq = model_iq(X_iq)
                pred_iq = get_continuous_angle(out_iq, radius=2)

                out_spe = model_spe(X_spe)
                pred_spe = get_continuous_angle(out_spe, radius=2)

                out_spice = model_spice(X_vec72)
                pred_spice = get_continuous_angle(out_spice, radius=2)

                # ================= MUSIC 计算 (仅针对纯传统算法) =================
                pred_music_list = []
                R_np = R_complex.cpu().numpy()

                for i in range(B):
                    # 只有纯正的 MUSIC 算法还需要跑这个循环
                    peak_m = music_algorithm(R_np[i])
                    pred_music_list.append(peak_m)

                pred_music = torch.tensor(pred_music_list, device=device)

                # ================= 记录误差 =================
                preds = {'vit': pred_vit, 'iq': pred_iq, 'spe': pred_spe, 'reg': pred_reg, 'spice': pred_spice,
                         'music': pred_music}

                for key in preds.keys():
                    err = torch.abs(preds[key] - labels_doa)
                    all_err[key].extend(err.cpu().numpy())
                    sq_err[key].append(err ** 2)

                for j in range(B):
                    csv_raw_records.append({
                        'SNR': snr, 'True_Angle': labels_doa[j].item(),
                        'ViT': pred_vit[j].item(), 'IQ_ResNet': pred_iq[j].item(),
                        'SPE_CNN': pred_spe[j].item(), 'CNN_Reg': pred_reg[j].item(),
                        'Learning_SPICE': pred_spice[j].item(), 'MUSIC': pred_music_list[j]  # 🌟 修复了字典里的调用
                    })

        rmse_res = {k: torch.sqrt(torch.mean(torch.cat(sq_err[k]))).item() for k in preds.keys()}
        csv_rmse_records.append({'SNR': snr, **rmse_res})
        print(
            f" -> ViT:{rmse_res['vit']:.2f}° | Reg:{rmse_res['reg']:.2f}° | SPE:{rmse_res['spe']:.2f}° | IQ:{rmse_res['iq']:.2f}° | SPICE:{rmse_res['spice']:.2f}° | MUSIC:{rmse_res['music']:.2f}°")

    # ================= 3. 存储与绘图 =================
    if csv_raw_records:
        pd.DataFrame(csv_raw_records).to_csv(os.path.join(save_dir, "Raw_Errors_CDF.csv"), index=False)
        pd.DataFrame(csv_rmse_records).to_csv(os.path.join(save_dir, "RMSE_Results.csv"), index=False)

        plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.linewidth': 1.0})
        df_rmse = pd.DataFrame(csv_rmse_records)
        metric_snrs = df_rmse["SNR"].tolist()
        rmse_records = [{"rho": rho, "snr": row["SNR"], "ViT": row["vit"], "IQ-ResNet": row["iq"], "SPE-CNN": row["spe"], "REG-CNN": row["reg"], "Learning-SPICE": row["spice"], "MUSIC": row["music"]} for _, row in df_rmse.iterrows()]
        key_snrs = {nearest_value(metric_snrs, target) for target in [-10, 0, 10]}
        key_records = [record for record in rmse_records if record["snr"] in key_snrs]
        save_csv(rmse_records, GAUSS_DATA_DIR / "SingleSource" / f"gauss_single_random_snr_rmse_rho{rho}.csv")
        save_csv(key_records, GAUSS_DATA_DIR / "SingleSource" / f"gauss_single_random_snr_key_points_rho{rho}.csv")

        # 图 1: RMSE 折线图
        plt.figure(figsize=(9, 6))
        plot_items = [
            ('ViT', 'vit', 'o', 'r'),
            ('REG-CNN', 'reg', 'D', 'y'),
            ('SPE-CNN', 'spe', '^', 'g'),
            ('IQ-ResNet', 'iq', 's', 'm'),
            ('Learning-SPICE', 'spice', 'v', 'c'),
            ('MUSIC', 'music', '*', 'k')
        ]
        for name, column, marker, color in plot_items:
            plt.plot(df_rmse['SNR'], df_rmse[column], marker=marker, color=color, linestyle='-', label=name)

        plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE (Deg)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', labelsize=12)
        # plt.title(f'RMSE Comparison in Gaussian  Rho={rho}', fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(save_dir, f"RMSE_Comparison_rho{rho}.png"), dpi=300, bbox_inches='tight')

        # 图 2: CDF 概率图
        plt.figure(figsize=(9, 6))

        def plot_cdf(err_arr, color, ls, label, lw=2):
            s_err = np.sort(err_arr)
            plt.plot(s_err, np.arange(len(s_err)) / (len(s_err) - 1), color=color, linestyle=ls, label=label,
                     linewidth=lw)

        plot_cdf(all_err['vit'], 'r', '-', 'ViT')
        plot_cdf(all_err['reg'], 'y', '-.', 'REG-CNN')
        plot_cdf(all_err['spe'], 'g', '--', 'SPE-CNN')
        plot_cdf(all_err['iq'], 'm', ':', 'IQ-ResNet')
        plot_cdf(all_err['spice'], 'c', '-', 'Learning-SPICE')
        plot_cdf(all_err['music'], 'k', '-', 'MUSIC')

        plt.xlabel('Absolute Error (Degrees)', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', labelsize=12)
        plt.xlim([0, 60])
        plt.ylim([0, 1.0])
        plt.legend(loc='lower right', fontsize=11, edgecolor='0.8')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(save_dir, f"CDF_Comparison_rho{rho}.png"), dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()
