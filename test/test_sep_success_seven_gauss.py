import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_datasets
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


def calc_sourcewise_recall_batch(pred_sorted, true_sorted, threshold):
    hit_mask = torch.abs(pred_sorted - true_sorted) < threshold
    return hit_mask.float().mean().item()


def calc_full_success_batch(pred_sorted, true_sorted, threshold):
    hit_mask = torch.abs(pred_sorted - true_sorted) < threshold
    full_success = hit_mask.all(dim=1).float()
    return full_success.mean().item()


def build_fixed_spacing_dataset(rho, snap, snr, d, batch_size=128, num_samples=2000):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    template = np.array([-3*d, -2*d, -1*d, 0.0, 1*d, 2*d, 3*d], dtype=np.float32)
    theta_set = np.tile(template, (num_samples, 1))
    Create_datasets(dataset, k=7, theta_set=theta_set, batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset, template.tolist()


def load_models(device, rho):
    models = {}

    embeding_dim = 768
    model_vit = VisionTransformer(embed_layer=scm_embeding(8, embeding_dim), embed_dim=embeding_dim, out_dims=7, drop_ratio=0, attn_drop_ratio=0).to(device)
    vit_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base_transfer\weight_transfer_SevenSource_rho{rho}.pth"
    model_vit.load_state_dict(torch.load(vit_weight, map_location=device))
    model_vit.eval()
    models['ViT'] = model_vit

    model_iq = IQ_ResNet(num_classes=181).to(device)
    iq_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\IQ_ResNet\SevenSource\IQ_ResNet_Gaussian_SevenSource_rho{rho}.pth"
    model_iq.load_state_dict(torch.load(iq_weight, map_location=device))
    model_iq.eval()
    models['IQ-ResNet'] = model_iq

    model_spe = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    spe_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\SPE_CNN_Gaussian_SevenSource_rho{rho}.pth"
    model_spe.load_state_dict(torch.load(spe_weight, map_location=device))
    model_spe.eval()
    models['SPE-CNN'] = model_spe

    model_reg = CNN_Regression(out_dim=7).to(device)
    reg_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\CNN\SevenSource\CNN_Regression_Gaussian_SevenSource_rho{rho}.pth"
    model_reg.load_state_dict(torch.load(reg_weight, map_location=device))
    model_reg.eval()
    models['REG-CNN'] = model_reg

    model_spice = LearningSPICE_SP_MLP(M=8).to(device)
    spice_weight = rf"D:\Python\Project\doa_estimation\Graduation\result\MLP\SevenSource\LearningSPICE_Gaussian_SevenSource_rho{rho}.pth"
    model_spice.load_state_dict(torch.load(spice_weight, map_location=device))
    model_spice.eval()
    models['Learning-SPICE'] = model_spice

    return models


def evaluate_all_models_spacing(models, loader_scm, loader_y, device, snap, threshold, M=8, K=7):
    recall_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    success_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    mse_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    total_batches = 0

    with torch.no_grad():
        for (inputs_scm, _), (inputs_complex, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
            B = inputs_complex.shape[0]
            inputs_complex = inputs_complex.to(device)
            true_angles = labels_doa_y.to(device).float().view(-1, K)
            true_sorted, _ = torch.sort(true_angles, dim=1)

            # ViT
            inputs_scm = inputs_scm.to(device).float()
            max_v = torch.max(torch.abs(inputs_scm.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_vit = inputs_scm / (max_v + 1e-8)
            pred_vit = models['ViT'](X_vit)
            pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)
            recall_accum['ViT'] += calc_sourcewise_recall_batch(pred_vit_sorted, true_sorted, threshold)
            success_accum['ViT'] += calc_full_success_batch(pred_vit_sorted, true_sorted, threshold)
            mse_accum['ViT'] += calc_rmse(pred_vit_sorted, true_sorted)

            R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

            # REG-CNN
            X_reg = torch.zeros(B, 2, M, M, device=device)
            X_reg[:, 0] = R.real
            X_reg[:, 1] = R.imag
            max_reg = torch.max(torch.abs(X_reg.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
            X_reg = X_reg / (max_reg + 1e-8)
            pred_reg = models['REG-CNN'](X_reg)
            pred_reg_sorted, _ = torch.sort(pred_reg, dim=1)
            recall_accum['REG-CNN'] += calc_sourcewise_recall_batch(pred_reg_sorted, true_sorted, threshold)
            success_accum['REG-CNN'] += calc_full_success_batch(pred_reg_sorted, true_sorted, threshold)
            mse_accum['REG-CNN'] += calc_rmse(pred_reg_sorted, true_sorted)

            # SPE-CNN
            X_spe = torch.zeros(B, 3, M, M, device=device)
            X_spe[:, 0] = R.real
            X_spe[:, 1] = R.imag
            X_spe[:, 2] = R.angle() / torch.pi
            max_spe = torch.max(torch.abs(R.view(B, -1)), dim=1)[0].view(B, 1, 1)
            X_spe[:, 0] = X_spe[:, 0] / (max_spe + 1e-8)
            X_spe[:, 1] = X_spe[:, 1] / (max_spe + 1e-8)
            pred_spe = get_continuous_angle_k7(models['SPE-CNN'](X_spe), K=7, radius=2)
            pred_spe_sorted, _ = torch.sort(pred_spe, dim=1)
            recall_accum['SPE-CNN'] += calc_sourcewise_recall_batch(pred_spe_sorted, true_sorted, threshold)
            success_accum['SPE-CNN'] += calc_full_success_batch(pred_spe_sorted, true_sorted, threshold)
            mse_accum['SPE-CNN'] += calc_rmse(pred_spe_sorted, true_sorted)

            # IQ-ResNet
            inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
            rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
            inputs_iq = inputs_iq / (rms_val + 1e-8)
            pred_iq = get_continuous_angle_k7(models['IQ-ResNet'](inputs_iq), K=7, radius=2)
            pred_iq_sorted, _ = torch.sort(pred_iq, dim=1)
            recall_accum['IQ-ResNet'] += calc_sourcewise_recall_batch(pred_iq_sorted, true_sorted, threshold)
            success_accum['IQ-ResNet'] += calc_full_success_batch(pred_iq_sorted, true_sorted, threshold)
            mse_accum['IQ-ResNet'] += calc_rmse(pred_iq_sorted, true_sorted)

            # Learning-SPICE
            max_spice = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
            R_norm = R / (max_spice + 1e-8)
            X_spice = scm_to_vec72(R_norm)
            pred_spice = get_continuous_angle_k7(models['Learning-SPICE'](X_spice), K=7, radius=2)
            pred_spice_sorted, _ = torch.sort(pred_spice, dim=1)
            recall_accum['Learning-SPICE'] += calc_sourcewise_recall_batch(pred_spice_sorted, true_sorted, threshold)
            success_accum['Learning-SPICE'] += calc_full_success_batch(pred_spice_sorted, true_sorted, threshold)
            mse_accum['Learning-SPICE'] += calc_rmse(pred_spice_sorted, true_sorted)

            # MUSIC
            pred_music = torch.zeros(B, K, device=device)
            for i in range(B):
                R_np = R[i].cpu().numpy()
                R_np = 0.5 * (R_np + R_np.conj().T)
                doa = music_algorithm_k7(R_np, M=M)
                pred_music[i] = torch.tensor(doa, device=device)
            pred_music_sorted, _ = torch.sort(pred_music, dim=1)
            recall_accum['MUSIC'] += calc_sourcewise_recall_batch(pred_music_sorted, true_sorted, threshold)
            success_accum['MUSIC'] += calc_full_success_batch(pred_music_sorted, true_sorted, threshold)
            mse_accum['MUSIC'] += calc_rmse(pred_music_sorted, true_sorted)

            total_batches += 1

    recall_dict = {name: float(recall_accum[name] / total_batches) for name in recall_accum}
    success_dict = {name: float(success_accum[name] / total_batches) for name in success_accum}
    rmse_dict = {name: float(np.sqrt(mse_accum[name] / total_batches)) for name in mse_accum}
    return recall_dict, success_dict, rmse_dict


def plot_curve(x_list, results, xlabel, ylabel, save_path):
    plt.figure(figsize=(9, 6))
    for name in PLOT_ORDER:
        vals = results[name]
        style = PLOT_STYLES[name]
        plt.plot(x_list, vals, color=style['color'], marker=style['marker'], linestyle='-', label=name)

    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    # plt.title(title, fontsize=16, fontweight='bold')
    if 'Recall' in ylabel or 'Success' in ylabel:
        plt.ylim(-0.02, 1.02)
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 50
    fixed_snr = 5.0
    d_list = [6, 7, 8, 10, 12, 14, 16]
    batch_size = 128
    num_samples = 2000

    save_dir = rf"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_K_7_rho{rho}"
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 Loading models...")
    models = load_models(device, rho)

    recall_results = {name: [] for name in PLOT_STYLES.keys()}
    success_results = {name: [] for name in PLOT_STYLES.keys()}
    rmse_results = {name: [] for name in PLOT_STYLES.keys()}
    template_records = []

    for d in d_list:
        threshold = d / 2.0
        dataset, template = build_fixed_spacing_dataset(rho=rho, snap=snap, snr=fixed_snr, d=d, batch_size=batch_size, num_samples=num_samples)

        print(f"\n📦 rho=0 | fixed_snr={fixed_snr} dB | d={d} | template={template} | threshold={threshold:.2f}")

        loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
        loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')

        recall_dict, success_dict, rmse_dict = evaluate_all_models_spacing(models, loader_scm, loader_y, device, snap=snap, threshold=threshold, M=8, K=7)

        for name in recall_results:
            recall_results[name].append(recall_dict[name])
            success_results[name].append(success_dict[name])
            rmse_results[name].append(rmse_dict[name])

        template_records.append({'d': d, 'template': template, 'threshold': threshold})
        print("Recall -> " + " | ".join([f"{name}: {recall_dict[name]:.3f}" for name in recall_results]))
        print("RMSE   -> " + " | ".join([f"{name}: {rmse_dict[name]:.3f}°" for name in rmse_results]))

    with open(os.path.join(save_dir, 'seven_source_spacing_group_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'rho': rho, 'fixed_snr': fixed_snr, 'd_list': d_list, 'templates': template_records, 'recall_results': recall_results,
                   'full_success_results': success_results, 'rmse_results': rmse_results}, f, indent=4, ensure_ascii=False)

    def make_records(metric_results):
        return [{"rho": rho, "fixed_snr": fixed_snr, "d": d, "ViT": metric_results["ViT"][idx], "IQ-ResNet": metric_results["IQ-ResNet"][idx], "SPE-CNN": metric_results["SPE-CNN"][idx], "REG-CNN": metric_results["REG-CNN"][idx], "Learning-SPICE": metric_results["Learning-SPICE"][idx], "MUSIC": metric_results["MUSIC"][idx]} for idx, d in enumerate(d_list)]

    threshold_records = [{"model": name, "recall_threshold": 0.9, "min_d": first_x_reach_threshold(d_list, values, 0.9)} for name, values in recall_results.items()]
    save_csv(make_records(rmse_results), GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_spacing_rmse_rho{rho}.csv")
    save_csv(make_records(recall_results), GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_spacing_recall_rho{rho}.csv")
    save_csv(make_records(success_results), GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_spacing_full_success_rho{rho}.csv")
    save_csv(threshold_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_spacing_recall_thresholds_rho{rho}.csv")

    plot_curve(d_list, recall_results, 'Spacing d (Degree)', 'Source-wise Recall',
               os.path.join(save_dir, 'seven_source_spacing_group_recall.png'))
    plot_curve(d_list, success_results, 'Spacing d (Degree)', 'Full Success Rate',
               os.path.join(save_dir, 'seven_source_spacing_group_success.png'))
    plot_curve(d_list, rmse_results, 'Spacing d (Degree)', 'RMSE (Degree)',
               os.path.join(save_dir, 'seven_source_spacing_group_rmse.png'))


if __name__ == '__main__':
    main()
