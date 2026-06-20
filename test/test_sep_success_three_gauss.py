import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

ext_lib = r"D:\Python\Project\doa_estimation\Graduation\external\DOA_est_Master-master"
proj_root = r"D:\Python\Project\doa_estimation"
if ext_lib not in sys.path:
    sys.path.insert(0, ext_lib)
if proj_root not in sys.path:
    sys.path.insert(1, proj_root)

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_datasets
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding, get_continuous_angle_k3, music_algorithm_k3
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.SPE_CNN import std_CNN
from dl_models.CNN_model import CNN_Regression
from dl_models.MLP import LearningSPICE_SP_MLP, scm_to_vec72
from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, first_x_reach_threshold, save_csv


PLOT_STYLES = {
    'ViT': {'color': 'r', 'marker': 'o'},
    'IQ-ResNet': {'color': 'm', 'marker': 's'},
    'SPE-CNN': {'color': 'g', 'marker': '^'},
    'REG-CNN': {'color': 'y', 'marker': 'D'},
    'Learning-SPICE': {'color': 'c', 'marker': 'v'},
    'MUSIC': {'color': 'k', 'marker': '*'}
}
PLOT_ORDER = ['ViT', 'REG-CNN', 'SPE-CNN', 'IQ-ResNet', 'Learning-SPICE', 'MUSIC']


def calc_success_rate_batch(pred_sorted, labels_sorted):
    delta = labels_sorted[:, 2] - labels_sorted[:, 1]
    threshold = delta / 2.0
    abs_err = torch.abs(pred_sorted - labels_sorted)
    success = (abs_err[:, 0] < threshold) & (abs_err[:, 1] < threshold) & (abs_err[:, 2] < threshold)
    return success.float().mean().item()


def build_fixed_triplet_dataset(rho, snap, snr, triplet=(-10.0, 0.0, 10.0), batch_size=128, num_samples=2000):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    theta_set = np.tile(np.array([triplet], dtype=np.float32), (num_samples, 1))
    Create_datasets(dataset, k=3, theta_set=theta_set, batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset


def build_fixed_delta_dataset(rho, snap, snr, delta, batch_size=128, num_samples=2000):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    triplet = (-float(delta), 0.0, float(delta))
    theta_set = np.tile(np.array([triplet], dtype=np.float32), (num_samples, 1))
    Create_datasets(dataset, k=3, theta_set=theta_set, batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset, triplet


def load_models(device, rho):
    models = {}

    embeding_dim = 768
    model_vit = VisionTransformer(embed_layer=scm_embeding(8, embeding_dim), embed_dim=embeding_dim, out_dims=3, drop_ratio=0, attn_drop_ratio=0).to(device)
    if rho == 0.0:
        vit_weight = r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base\weight_base_ThreeSource.pth"
        # vit_weight = r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base_transfer\weight_transfer_ThreeSource_rho0.0.pth"
    else:
        vit_weight = r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base_transfer\weight_transfer_ThreeSource_rho1.0.pth"
    model_vit.load_state_dict(torch.load(vit_weight, map_location=device))
    model_vit.eval()
    models['ViT'] = model_vit

    model_iq = IQ_ResNet(num_classes=181).to(device)
    model_iq.load_state_dict(torch.load(rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\ThreeSource\IQ_ResNet_Gaussian_ThreeSource_rho{rho}.pth", map_location=device))
    model_iq.eval()
    models['IQ-ResNet'] = model_iq

    model_spe = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    model_spe.load_state_dict(torch.load(rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\ThreeSource\SPE_CNN_Gaussian_ThreeSource_rho{rho}.pth", map_location=device))
    model_spe.eval()
    models['SPE-CNN'] = model_spe

    model_reg = CNN_Regression(out_dim=3).to(device)
    model_reg.load_state_dict(torch.load(rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\ThreeSource\CNN_Regression_Gaussian_ThreeSource_rho{rho}.pth", map_location=device))
    model_reg.eval()
    models['REG-CNN'] = model_reg

    model_spice = LearningSPICE_SP_MLP(M=8, out_dim=181).to(device)
    model_spice.load_state_dict(torch.load(rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\ThreeSource\LearningSPICE_Gaussian_ThreeSource_rho{rho}.pth", map_location=device))
    model_spice.eval()
    models['Learning-SPICE'] = model_spice

    return models


def evaluate_all_models_success(models, loader_scm, loader_y, device, snap, M=8, K=3):
    success_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    total_batches = 0

    with torch.no_grad():
        for (inputs_scm, _), (inputs_complex, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
            B = inputs_complex.shape[0]
            inputs_complex = inputs_complex.to(device)
            true_angles = labels_doa_y.to(device).float().view(-1, K)
            true_sorted, _ = torch.sort(true_angles, dim=1)

            inputs_scm = inputs_scm.to(device).float()
            max_v = torch.max(torch.abs(inputs_scm.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_vit = inputs_scm / (max_v + 1e-8)
            pred_vit = models['ViT'](X_vit)
            pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)
            success_accum['ViT'] += calc_success_rate_batch(pred_vit_sorted, true_sorted)

            R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

            X_reg = torch.zeros(B, 2, M, M, device=device)
            X_reg[:, 0] = R.real
            X_reg[:, 1] = R.imag
            max_reg = torch.max(torch.abs(X_reg.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_reg = X_reg / (max_reg + 1e-8)
            pred_reg = models['REG-CNN'](X_reg)
            pred_reg_sorted, _ = torch.sort(pred_reg, dim=1)
            success_accum['REG-CNN'] += calc_success_rate_batch(pred_reg_sorted, true_sorted)

            X_spe = torch.zeros(B, 3, M, M, device=device)
            X_spe[:, 0] = R.real
            X_spe[:, 1] = R.imag
            X_spe[:, 2] = R.angle() / torch.pi
            max_spe = torch.max(torch.abs(R.view(B, -1)), dim=1)[0].view(B, 1, 1)
            X_spe[:, 0] = X_spe[:, 0] / (max_spe + 1e-8)
            X_spe[:, 1] = X_spe[:, 1] / (max_spe + 1e-8)
            pred_spe = get_continuous_angle_k3(models['SPE-CNN'](X_spe), K=3, radius=2)
            pred_spe_sorted, _ = torch.sort(pred_spe, dim=1)
            success_accum['SPE-CNN'] += calc_success_rate_batch(pred_spe_sorted, true_sorted)

            inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
            rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
            inputs_iq = inputs_iq / (rms_val + 1e-8)
            pred_iq = get_continuous_angle_k3(models['IQ-ResNet'](inputs_iq), K=3, radius=2)
            pred_iq_sorted, _ = torch.sort(pred_iq, dim=1)
            success_accum['IQ-ResNet'] += calc_success_rate_batch(pred_iq_sorted, true_sorted)

            max_spice = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
            R_norm = R / (max_spice + 1e-8)
            X_spice = scm_to_vec72(R_norm)
            pred_spice = get_continuous_angle_k3(models['Learning-SPICE'](X_spice), K=3, radius=2)
            pred_spice_sorted, _ = torch.sort(pred_spice, dim=1)
            success_accum['Learning-SPICE'] += calc_success_rate_batch(pred_spice_sorted, true_sorted)

            pred_music = torch.zeros(B, K, device=device)
            for i in range(B):
                R_np = R[i].cpu().numpy()
                R_np = 0.5 * (R_np + R_np.conj().T)
                pred_music[i] = torch.tensor(music_algorithm_k3(R_np, num_sources=K, M=M), device=device)
            pred_music_sorted, _ = torch.sort(pred_music, dim=1)
            success_accum['MUSIC'] += calc_success_rate_batch(pred_music_sorted, true_sorted)

            total_batches += 1

    success_dict = {name: float(success_accum[name] / total_batches) for name in success_accum}
    return success_dict


@torch.no_grad()
def evaluate_vit_only_success(model_vit, loader_scm, loader_y, device, K=3):
    success_accum = 0.0
    total_batches = 0

    for (inputs_scm, _), (_, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
        B = inputs_scm.shape[0]
        inputs_scm = inputs_scm.to(device).float()
        true_angles = labels_doa_y.to(device).float().view(-1, K)
        true_sorted, _ = torch.sort(true_angles, dim=1)

        max_v = torch.max(torch.abs(inputs_scm.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        X_vit = inputs_scm / (max_v + 1e-8)
        pred_vit = model_vit(X_vit)
        pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)

        success_accum += calc_success_rate_batch(pred_vit_sorted, true_sorted)
        total_batches += 1

    if total_batches == 0:
        print("Warning: evaluate_vit_only_success got 0 batches.")
        return 0.0
    return float(success_accum / total_batches)


def plot_curve(x_list, results, xlabel, save_path, extra_series=None):
    plt.figure(figsize=(9, 6))
    for name in PLOT_ORDER:
        vals = results[name]
        style = PLOT_STYLES[name]
        plt.plot(x_list, vals, color=style['color'], marker=style['marker'], linestyle='-', label=name)

    if extra_series is not None:
        for name, vals, color, marker, ls in extra_series:
            plt.plot(x_list, vals, color=color, marker=marker, linestyle=ls, label=name)

    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel('Probability of Success', fontsize=14, fontweight='bold')
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(-0.02, 1.02)
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_rho0_fixed_triplet_success(device):
    rho = 1.0
    snap = 50
    triplet = (-10.0, 0.0, 10.0)
    snr_list = [-20, -15, -10, -5, 0, 5, 10]
    batch_size = 128
    num_samples = 2000

    save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_k_3_rho0.0"
    os.makedirs(save_dir, exist_ok=True)

    models = load_models(device, rho)
    results = {name: [] for name in PLOT_STYLES.keys()}

    for snr in snr_list:
        print(f"\n📦 rho=0 | triplet={triplet} | SNR={snr} dB")
        dataset = build_fixed_triplet_dataset(rho=rho, snap=snap, snr=snr, triplet=triplet, batch_size=batch_size, num_samples=num_samples)
        loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
        loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')

        success_dict = evaluate_all_models_success(models, loader_scm, loader_y, device, snap=snap)
        for name in results:
            results[name].append(success_dict[name])

        print(" | ".join([f"{name}: {success_dict[name]:.3f}" for name in results]))

    with open(os.path.join(save_dir, f'rho{rho}_fixed_triplet_success_theta10.json'), 'w', encoding='utf-8') as f:
        json.dump({'rho': rho, 'triplet': triplet, 'snr_list': snr_list, 'success_results': results}, f, indent=4, ensure_ascii=False)

    csv_records = [{"rho": rho, "triplet": str(triplet), "snr": snr, "ViT": results["ViT"][idx], "IQ-ResNet": results["IQ-ResNet"][idx], "SPE-CNN": results["SPE-CNN"][idx], "REG-CNN": results["REG-CNN"][idx], "Learning-SPICE": results["Learning-SPICE"][idx], "MUSIC": results["MUSIC"][idx]} for idx, snr in enumerate(snr_list)]
    save_csv(csv_records, GAUSS_DATA_DIR / "ThreeSource" / "gauss_three_rho0_fixed_triplet_success.csv")

    plot_curve(snr_list, results, 'Signal-to-Noise Ratio (dB)',
               os.path.join(save_dir, f'rho{rho}_fixed_triplet_success_theta10.png'))


def run_rho1_delta_success(device):
    rho = 1.0
    snap = 50
    fixed_snr = 0.0
    delta_list = [4, 5, 6, 8, 10, 12, 14, 16]
    batch_size = 128
    num_samples = 2000

    save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_k_3_rho1.0"
    os.makedirs(save_dir, exist_ok=True)

    models = load_models(device, rho)
    results = {name: [] for name in PLOT_STYLES.keys()}

    for delta in delta_list:
        dataset, triplet = build_fixed_delta_dataset(rho=rho, snap=snap, snr=fixed_snr, delta=delta, batch_size=batch_size, num_samples=num_samples)
        print(f"\n📦 rho=1 | triplet={triplet} | SNR={fixed_snr} dB")

        loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
        loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')

        success_dict = evaluate_all_models_success(models, loader_scm, loader_y, device, snap=snap)
        for name in results:
            results[name].append(success_dict[name])

        print(" | ".join([f"{name}: {success_dict[name]:.3f}" for name in results]))

    with open(os.path.join(save_dir, f'rho{rho}_delta_success.json'), 'w', encoding='utf-8') as f:
        json.dump({'rho': rho, 'fixed_snr': fixed_snr, 'delta_list': delta_list, 'success_results': results}, f, indent=4, ensure_ascii=False)

    csv_records = [{"rho": rho, "fixed_snr": fixed_snr, "delta": delta, "ViT": results["ViT"][idx], "IQ-ResNet": results["IQ-ResNet"][idx], "SPE-CNN": results["SPE-CNN"][idx], "REG-CNN": results["REG-CNN"][idx], "Learning-SPICE": results["Learning-SPICE"][idx], "MUSIC": results["MUSIC"][idx]} for idx, delta in enumerate(delta_list)]
    threshold_records = [{"model": name, "success_threshold": 0.9, "min_delta": first_x_reach_threshold(delta_list, values, 0.9)} for name, values in results.items()]
    save_csv(csv_records, GAUSS_DATA_DIR / "ThreeSource" / "gauss_three_rho1_delta_success.csv")
    save_csv(threshold_records, GAUSS_DATA_DIR / "ThreeSource" / "gauss_three_rho1_delta_success_thresholds.csv")

    plot_curve(delta_list, results, 'Delta (Degree) in [-Δ, 0, +Δ]',
             os.path.join(save_dir, f'rho{rho}_delta_success.png'))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_rho0_fixed_triplet_success(device)

    run_rho1_delta_success(device)


if __name__ == '__main__':
    main()
