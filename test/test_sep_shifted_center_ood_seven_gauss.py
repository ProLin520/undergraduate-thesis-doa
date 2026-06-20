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

from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, save_csv

PLOT_STYLES = {
    'ViT': {'color': 'r', 'marker': 'o'},
    'IQ-ResNet': {'color': 'm', 'marker': 's'},
    'SPE-CNN': {'color': 'g', 'marker': '^'},
    'REG-CNN': {'color': 'y', 'marker': 'D'},
    'Learning-SPICE': {'color': 'c', 'marker': 'v'},
    'MUSIC': {'color': 'k', 'marker': '*'}
}
PLOT_ORDER = ['ViT', 'REG-CNN', 'SPE-CNN', 'IQ-ResNet', 'Learning-SPICE', 'MUSIC']

def to_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def build_shifted_spacing_template(center, d):
    theta = np.array([center - 3*d, center - 2*d, center - d, center, center + d, center + 2*d, center + 3*d], dtype=np.float32)
    if theta.min() < -90 or theta.max() > 90:
        raise ValueError(f"非法模板: center={center}, d={d}, theta={theta.tolist()}")
    return theta


def build_shifted_fixed_dataset(rho, snap, snr, center, d, batch_size=128, num_samples=2000):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    template = build_shifted_spacing_template(center, d)
    theta_set = np.tile(template, (num_samples, 1)).astype(np.float32)
    Create_datasets(dataset, k=7, theta_set=theta_set, batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset, template.tolist()


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


def calc_sourcewise_recall_batch(pred_sorted, true_sorted, threshold):
    hit_mask = torch.abs(pred_sorted - true_sorted) <= threshold
    return hit_mask.float().mean().item()


def calc_full_success_batch(pred_sorted, true_sorted, threshold):
    hit_mask = torch.abs(pred_sorted - true_sorted) <= threshold
    return hit_mask.all(dim=1).float().mean().item()


@torch.no_grad()
def evaluate_all_models_shifted(models, loader_scm, loader_y, device, snap, threshold, M=8, K=7):
    recall_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    success_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    mse_accum = {'ViT': 0.0, 'IQ-ResNet': 0.0, 'SPE-CNN': 0.0, 'REG-CNN': 0.0, 'Learning-SPICE': 0.0, 'MUSIC': 0.0}
    total_batches = 0

    for (inputs_scm, _), (inputs_complex, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
        B = inputs_complex.shape[0]
        inputs_complex = inputs_complex.to(device)
        true_angles = labels_doa_y.to(device).float().view(-1, K)
        true_sorted, _ = torch.sort(true_angles, dim=1)

        # ViT
        inputs_scm = inputs_scm.to(device).float()
        max_v = torch.max(torch.abs(inputs_scm.reshape(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        X_vit = inputs_scm / (max_v + 1e-8)
        pred_vit = models['ViT'](X_vit)
        pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)

        recall_accum['ViT'] += calc_sourcewise_recall_batch(pred_vit_sorted, true_sorted, threshold)
        success_accum['ViT'] += calc_full_success_batch(pred_vit_sorted, true_sorted, threshold)
        mse_accum['ViT'] += to_float(calc_rmse(pred_vit_sorted, true_sorted))

        # Shared covariance
        R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

        # REG-CNN
        X_reg = torch.zeros(B, 2, M, M, device=device)
        X_reg[:, 0] = R.real
        X_reg[:, 1] = R.imag
        max_reg = torch.max(torch.abs(X_reg.reshape(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        X_reg = X_reg / (max_reg + 1e-8)
        pred_reg = models['REG-CNN'](X_reg)
        pred_reg_sorted, _ = torch.sort(pred_reg, dim=1)

        recall_accum['REG-CNN'] += calc_sourcewise_recall_batch(pred_reg_sorted, true_sorted, threshold)
        success_accum['REG-CNN'] += calc_full_success_batch(pred_reg_sorted, true_sorted, threshold)
        mse_accum['REG-CNN'] += to_float(calc_rmse(pred_reg_sorted, true_sorted))

        # SPE-CNN
        X_spe = torch.zeros(B, 3, M, M, device=device)
        X_spe[:, 0] = R.real
        X_spe[:, 1] = R.imag
        X_spe[:, 2] = R.angle() / torch.pi
        max_spe = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].view(B, 1, 1)
        X_spe[:, 0] = X_spe[:, 0] / (max_spe + 1e-8)
        X_spe[:, 1] = X_spe[:, 1] / (max_spe + 1e-8)
        pred_spe = get_continuous_angle_k7(models['SPE-CNN'](X_spe), K=7, radius=2)
        pred_spe_sorted, _ = torch.sort(pred_spe, dim=1)

        recall_accum['SPE-CNN'] += calc_sourcewise_recall_batch(pred_spe_sorted, true_sorted, threshold)
        success_accum['SPE-CNN'] += calc_full_success_batch(pred_spe_sorted, true_sorted, threshold)
        mse_accum['SPE-CNN'] += to_float(calc_rmse(pred_spe_sorted, true_sorted))

        # IQ-ResNet
        inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
        rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
        inputs_iq = inputs_iq / (rms_val + 1e-8)
        pred_iq = get_continuous_angle_k7(models['IQ-ResNet'](inputs_iq), K=7, radius=2)
        pred_iq_sorted, _ = torch.sort(pred_iq, dim=1)

        recall_accum['IQ-ResNet'] += calc_sourcewise_recall_batch(pred_iq_sorted, true_sorted, threshold)
        success_accum['IQ-ResNet'] += calc_full_success_batch(pred_iq_sorted, true_sorted, threshold)
        mse_accum['IQ-ResNet'] += to_float(calc_rmse(pred_iq_sorted, true_sorted))

        # Learning-SPICE
        max_spice = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
        R_norm = R / (max_spice + 1e-8)
        X_spice = scm_to_vec72(R_norm)
        pred_spice = get_continuous_angle_k7(models['Learning-SPICE'](X_spice), K=7, radius=2)
        pred_spice_sorted, _ = torch.sort(pred_spice, dim=1)

        recall_accum['Learning-SPICE'] += calc_sourcewise_recall_batch(pred_spice_sorted, true_sorted, threshold)
        success_accum['Learning-SPICE'] += calc_full_success_batch(pred_spice_sorted, true_sorted, threshold)
        mse_accum['Learning-SPICE'] += to_float(calc_rmse(pred_spice_sorted, true_sorted))

        # MUSIC
        pred_music = torch.zeros(B, K, device=device)
        for i in range(B):
            R_np = R[i].cpu().numpy()
            R_np = 0.5 * (R_np + R_np.conj().T)
            pred_music[i] = torch.tensor(music_algorithm_k7(R_np, M=M), device=device)
        pred_music_sorted, _ = torch.sort(pred_music, dim=1)

        recall_accum['MUSIC'] += calc_sourcewise_recall_batch(pred_music_sorted, true_sorted, threshold)
        success_accum['MUSIC'] += calc_full_success_batch(pred_music_sorted, true_sorted, threshold)
        mse_accum['MUSIC'] += to_float(calc_rmse(pred_music_sorted, true_sorted))

        total_batches += 1

    recall_dict = {name: float(recall_accum[name] / total_batches) for name in recall_accum}
    success_dict = {name: float(success_accum[name] / total_batches) for name in success_accum}
    rmse_dict = {name: float(np.sqrt(mse_accum[name] / total_batches)) for name in mse_accum}
    return recall_dict, success_dict, rmse_dict


def plot_curve(center_list, results, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in PLOT_ORDER:
        vals = results[name]
        style = PLOT_STYLES[name]
        ax.plot(center_list, vals, color=style['color'], marker=style['marker'], linestyle='-', label=name)

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    # plt.title(title, fontsize=16, fontweight='bold')
    if 'Recall' in ylabel or 'Success' in ylabel:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_shifted_center_ood_test(device):
    rho = 0.0
    snap = 50
    fixed_snr = 5.0
    batch_size = 128
    num_samples = 2000

    d_test_list = [8, 12]
    center_list = [-24, -16, -8, 0, 8, 16, 24]

    save_dir = rf"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_K_7_rho{rho}\M_8_K_7_rho{rho}_shifted_center"
    os.makedirs(save_dir, exist_ok=True)

    models = load_models(device, rho)

    results = {
        'rho': rho,
        'snap': snap,
        'fixed_snr': fixed_snr,
        'd_test_list': d_test_list,
        'center_list': center_list,
        'recall_results': {},
        'full_success_results': {},
        'rmse_results': {},
        'templates': {}
    }
    key_point_records = []

    def make_records(metric_results, d):
        return [{"rho": rho, "fixed_snr": fixed_snr, "d": d, "center": center, "ViT": metric_results["ViT"][idx], "IQ-ResNet": metric_results["IQ-ResNet"][idx], "SPE-CNN": metric_results["SPE-CNN"][idx], "REG-CNN": metric_results["REG-CNN"][idx], "Learning-SPICE": metric_results["Learning-SPICE"][idx], "MUSIC": metric_results["MUSIC"][idx]} for idx, center in enumerate(center_list)]

    for d in d_test_list:
        d_key = f"d={d}"
        results['recall_results'][d_key] = {name: [] for name in PLOT_STYLES.keys()}
        results['full_success_results'][d_key] = {name: [] for name in PLOT_STYLES.keys()}
        results['rmse_results'][d_key] = {name: [] for name in PLOT_STYLES.keys()}
        results['templates'][d_key] = []

        for center in center_list:
            threshold = d / 2.0
            dataset, template = build_shifted_fixed_dataset(rho=rho, snap=snap, snr=fixed_snr, center=center, d=d, batch_size=batch_size, num_samples=num_samples)

            print(f"\n📦 shifted-center OOD | d={d} | center={center} | template={template} | threshold={threshold:.2f}")

            loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
            loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')

            recall_dict, success_dict, rmse_dict = evaluate_all_models_shifted(models, loader_scm, loader_y, device, snap=snap, threshold=threshold, M=8, K=7)

            for name in PLOT_STYLES.keys():
                results['recall_results'][d_key][name].append(recall_dict[name])
                results['full_success_results'][d_key][name].append(success_dict[name])
                results['rmse_results'][d_key][name].append(rmse_dict[name])

            results['templates'][d_key].append({'center': center, 'template': template, 'threshold': threshold})

            print("Recall -> " + " | ".join([f"{name}: {recall_dict[name]:.3f}" for name in PLOT_STYLES.keys()]))
            print("Succ   -> " + " | ".join([f"{name}: {success_dict[name]:.3f}" for name in PLOT_STYLES.keys()]))
            print("RMSE   -> " + " | ".join([f"{name}: {rmse_dict[name]:.3f}°" for name in PLOT_STYLES.keys()]))

        with open(os.path.join(save_dir, f'seven_source_shifted_center_ood_{d_key}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'rho': rho,
                'snap': snap,
                'fixed_snr': fixed_snr,
                'd': d,
                'center_list': center_list,
                'templates': results['templates'][d_key],
                'recall_results': results['recall_results'][d_key],
                'full_success_results': results['full_success_results'][d_key],
                'rmse_results': results['rmse_results'][d_key]
            }, f, indent=4, ensure_ascii=False)

        rmse_records = make_records(results['rmse_results'][d_key], d)
        recall_records = make_records(results['recall_results'][d_key], d)
        success_records = make_records(results['full_success_results'][d_key], d)
        save_csv(rmse_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_shifted_center_rmse_d{d}_rho{rho}.csv")
        save_csv(recall_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_shifted_center_recall_d{d}_rho{rho}.csv")
        save_csv(success_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_shifted_center_full_success_d{d}_rho{rho}.csv")
        for metric, records in [("rmse", rmse_records), ("recall", recall_records), ("full_success", success_records)]:
            for record in records:
                if record["center"] in [-24, 0, 24]:
                    for model in PLOT_STYLES.keys():
                        key_point_records.append({"rho": rho, "fixed_snr": fixed_snr, "d": d, "center": record["center"], "metric": metric, "model": model, "value": record[model]})

        plot_curve(center_list, results['recall_results'][d_key], 'Center Shift c (Degree)', 'Source-wise Recall',
                   os.path.join(save_dir, f'seven_source_shifted_center_ood_recall_{d_key}.png'))
        plot_curve(center_list, results['full_success_results'][d_key], 'Center Shift c (Degree)', 'Full Success Rate',
                   os.path.join(save_dir, f'seven_source_shifted_center_ood_success_{d_key}.png'))
        plot_curve(center_list, results['rmse_results'][d_key], 'Center Shift c (Degree)', 'RMSE (Degree)',
                   os.path.join(save_dir, f'seven_source_shifted_center_ood_rmse_{d_key}.png'))

    with open(os.path.join(save_dir, 'seven_source_shifted_center_ood_all.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    save_csv(key_point_records, GAUSS_DATA_DIR / "SevenSource" / f"gauss_seven_shifted_center_key_points_rho{rho}.csv")

    print(f"\n✅ shifted-center OOD 测试完成，结果保存到: {save_dir}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_shifted_center_ood_test(device)


if __name__ == '__main__':
    main()
