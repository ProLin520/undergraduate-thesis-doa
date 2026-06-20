import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding, calc_rmse, get_continuous_angle_k7, music_algorithm_k7
from dl_models.CNN_model import CNN_Regression
from dl_models.SPE_CNN import std_CNN
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.MLP import LearningSPICE_SP_MLP, scm_to_vec72

ext_lib = r"D:\Python\Project\doa_estimation\Graduation\external\DOA_est_Master-master"
proj_root = r"D:\Python\Project\doa_estimation"
if ext_lib not in sys.path:
    sys.path.insert(0, ext_lib)
if proj_root not in sys.path:
    sys.path.insert(1, proj_root)

from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, nearest_value, save_csv

PLOT_STYLES = {
    'ViT': {'color': 'r', 'marker': 'o'},
    'IQ-ResNet': {'color': 'm', 'marker': 's'},
    'SPE-CNN': {'color': 'g', 'marker': '^'},
    'REG-CNN': {'color': 'y', 'marker': 'D'},
    'Learning-SPICE': {'color': 'c', 'marker': 'v'},
    'MUSIC': {'color': 'k', 'marker': '*'}
}
PLOT_ORDER = ['ViT', 'REG-CNN', 'SPE-CNN', 'IQ-ResNet', 'Learning-SPICE', 'MUSIC']


def build_random_theta_once(theta_num=2000, min_delta_theta=8):
    theta_set = Create_random_k_input_theta(7, -90, 90, theta_num, min_delta_theta=min_delta_theta)
    theta_set = np.array(theta_set, dtype=np.float32)
    valid_mask = (~np.isnan(theta_set).any(axis=1)) & (np.max(theta_set, axis=1) <= 90) & (np.min(theta_set, axis=1) >= -90)
    theta_set = theta_set[valid_mask]
    print(f"固定 random 七信源角度样本数: {len(theta_set)}")
    return theta_set


def build_dataset_from_theta(rho, snap, snr, theta_set, batch_size=128):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    Create_datasets(dataset, k=7, theta_set=theta_set.copy(), batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset


def load_models(device, rho):
    models = {}

    embeding_dim = 768
    model_vit = VisionTransformer(embed_layer=scm_embeding(8, embeding_dim), embed_dim=embeding_dim, out_dims=7, drop_ratio=0, attn_drop_ratio=0).to(device)
    vit_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base_transfer\weight_transfer_SevenSource_rho{rho}.pth"
    print(f"[ViT] loading: {vit_weight}")
    model_vit.load_state_dict(torch.load(vit_weight, map_location=device))
    model_vit.eval()
    models['ViT'] = model_vit

    model_iq = IQ_ResNet(num_classes=181).to(device)
    iq_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\SevenSource\IQ_ResNet_Gaussian_SevenSource_rho{rho}.pth"
    print(f"[IQ-ResNet] loading: {iq_weight}")
    model_iq.load_state_dict(torch.load(iq_weight, map_location=device))
    model_iq.eval()
    models['IQ-ResNet'] = model_iq

    model_spe = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    spe_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\SPE_CNN_Gaussian_SevenSource_rho{rho}.pth"
    print(f"[SPE-CNN] loading: {spe_weight}")
    model_spe.load_state_dict(torch.load(spe_weight, map_location=device))
    model_spe.eval()
    models['SPE-CNN'] = model_spe

    model_reg = CNN_Regression(out_dim=7).to(device)
    reg_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\CNN_Regression_Gaussian_SevenSource_rho{rho}.pth"
    print(f"[REG-CNN] loading: {reg_weight}")
    model_reg.load_state_dict(torch.load(reg_weight, map_location=device))
    model_reg.eval()
    models['REG-CNN'] = model_reg

    model_spice = LearningSPICE_SP_MLP(M=8).to(device)
    spice_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\SevenSource\LearningSPICE_Gaussian_SevenSource_rho{rho}.pth"
    print(f"[Learning-SPICE] loading: {spice_weight}")
    model_spice.load_state_dict(torch.load(spice_weight, map_location=device))
    model_spice.eval()
    models['Learning-SPICE'] = model_spice

    return models


def evaluate_all_models(models, loader_scm, loader_y, device, snap, M=8, K=7):
    mse_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    total_batches = 0

    with torch.no_grad():
        for (inputs_scm, _), (inputs_complex, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
            B = inputs_complex.shape[0]
            inputs_complex = inputs_complex.to(device)
            true_angles = labels_doa_y.to(device).float().view(-1, K)
            true_sorted, _ = torch.sort(true_angles, dim=1)

            # ===== ViT =====
            inputs_scm = inputs_scm.to(device).float()
            max_v = torch.max(torch.abs(inputs_scm.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_vit = inputs_scm / (max_v + 1e-8)
            pred_vit = models['ViT'](X_vit)
            pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)
            mse_accum['ViT'] += calc_rmse(pred_vit_sorted, true_sorted)

            # ===== 共用协方差 =====
            R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

            # ===== REG-CNN =====
            X_reg = torch.zeros(B, 2, M, M, device=device)
            X_reg[:, 0] = R.real
            X_reg[:, 1] = R.imag
            max_reg = torch.max(torch.abs(X_reg.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_reg = X_reg / (max_reg + 1e-8)
            pred_reg = models['REG-CNN'](X_reg)
            pred_reg_sorted, _ = torch.sort(pred_reg, dim=1)
            mse_accum['REG-CNN'] += calc_rmse(pred_reg_sorted, true_sorted)

            # ===== SPE-CNN =====
            X_spe = torch.zeros(B, 3, M, M, device=device)
            X_spe[:, 0] = R.real
            X_spe[:, 1] = R.imag
            X_spe[:, 2] = R.angle() / torch.pi
            max_spe = torch.max(torch.abs(R.view(B, -1)), dim=1)[0].view(B, 1, 1)
            X_spe[:, 0] = X_spe[:, 0] / (max_spe + 1e-8)
            X_spe[:, 1] = X_spe[:, 1] / (max_spe + 1e-8)
            pred_spe = get_continuous_angle_k7(models['SPE-CNN'](X_spe), K=7, radius=2)
            pred_spe_sorted, _ = torch.sort(pred_spe, dim=1)
            mse_accum['SPE-CNN'] += calc_rmse(pred_spe_sorted, true_sorted)

            # ===== IQ-ResNet =====
            inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
            rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
            inputs_iq = inputs_iq / (rms_val + 1e-8)
            pred_iq = get_continuous_angle_k7(models['IQ-ResNet'](inputs_iq), K=7, radius=2)
            pred_iq_sorted, _ = torch.sort(pred_iq, dim=1)
            mse_accum['IQ-ResNet'] += calc_rmse(pred_iq_sorted, true_sorted)

            # ===== Learning-SPICE =====
            max_spice = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
            R_norm = R / (max_spice + 1e-8)
            X_spice = scm_to_vec72(R_norm)
            pred_spice = get_continuous_angle_k7(models['Learning-SPICE'](X_spice), K=7, radius=2)
            pred_spice_sorted, _ = torch.sort(pred_spice, dim=1)
            mse_accum['Learning-SPICE'] += calc_rmse(pred_spice_sorted, true_sorted)

            # ===== MUSIC =====
            pred_music = torch.zeros(B, K, device=device)
            for i in range(B):
                R_np = R[i].cpu().numpy()
                R_np = 0.5 * (R_np + R_np.conj().T)
                doa = music_algorithm_k7(R_np, M=M)
                pred_music[i] = torch.tensor(doa, device=device)
            pred_music_sorted, _ = torch.sort(pred_music, dim=1)
            mse_accum['MUSIC'] += calc_rmse(pred_music_sorted, true_sorted)

            total_batches += 1

    rmse_dict = {name: float(np.sqrt(mse_accum[name] / total_batches)) for name in mse_accum}
    return rmse_dict


def plot_curve(x_list, results, xlabel, save_path):
    plt.figure(figsize=(9, 6))
    for name in PLOT_ORDER:
        vals = results[name]
        style = PLOT_STYLES[name]
        plt.plot(x_list, vals, color=style['color'], marker=style['marker'], linestyle='-', label=name)

    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (Degree)', fontsize=14, fontweight='bold')
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 50
    snr_list = [-20, -15, -10, -5, 0, 5, 10]
    batch_size = 128
    theta_num = 2000
    min_delta_theta = 8

    save_dir = rf"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_K_7_rho{rho}"
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 Loading models...")
    models = load_models(device, rho)

    print("🚀 Building fixed random theta set once...")
    theta_set = build_random_theta_once(theta_num=theta_num, min_delta_theta=min_delta_theta)

    results = {'ViT': [], 'IQ-ResNet': [], 'SPE-CNN': [], 'REG-CNN': [], 'Learning-SPICE': [], 'MUSIC': []}

    for snr in snr_list:
        print(f"\n📦 rho=0 | random seven-source | SNR={snr} dB | theta_num={len(theta_set)} | min_delta={min_delta_theta}")
        dataset = build_dataset_from_theta(rho=rho, snap=snap, snr=snr, theta_set=theta_set, batch_size=batch_size)
        loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
        loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')

        rmse_dict = evaluate_all_models(models, loader_scm, loader_y, device, snap=snap, M=8, K=7)
        for name in results:
            results[name].append(rmse_dict[name])

        print(" | ".join([f"{name}: {rmse_dict[name]:.3f}°" for name in results]))

    with open(os.path.join(save_dir, 'seven_source_random_rmse.json'), 'w', encoding='utf-8') as f:
        json.dump({'rho': rho, 'snr_list': snr_list, 'theta_num': int(len(theta_set)), 'min_delta_theta': min_delta_theta, 'rmse_results': results}, f, indent=4, ensure_ascii=False)

    rmse_records = [{"rho": rho, "snr": snr, "ViT": results["ViT"][idx], "IQ-ResNet": results["IQ-ResNet"][idx], "SPE-CNN": results["SPE-CNN"][idx], "REG-CNN": results["REG-CNN"][idx], "Learning-SPICE": results["Learning-SPICE"][idx], "MUSIC": results["MUSIC"][idx]} for idx, snr in enumerate(snr_list)]
    key_snrs = {nearest_value(snr_list, target) for target in [0, 5, 10]}
    key_records = [record for record in rmse_records if record["snr"] in key_snrs]
    save_csv(rmse_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_random_snr_rmse_rho{rho}.csv")
    save_csv(key_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_random_snr_key_points_rho{rho}.csv")

    plot_curve(snr_list, results, 'Signal-to-Noise Ratio (dB)',
               os.path.join(save_dir, 'seven_source_random_rmse.png'))
    # plot_curve(snr_list, results, 'Signal-to-Noise Ratio (dB)', f'Seven-Source Random-Input RMSE vs SNR (rho={rho})',
    #            os.path.join(save_dir, 'seven_source_random_rmse.png'))


if __name__ == '__main__':
    main()
