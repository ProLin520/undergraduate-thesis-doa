import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

root = Path(__file__).resolve().parents[2]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path = [str(ext_lib), str(root)] + sys.path

from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, nearest_value, save_csv
from data.data_create.Create_k_source_dataset90 import Create_datasets, Create_random_k_input_theta
from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from dl_models.CNN_model import CNN_Regression
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.MLP import LearningSPICE_SP_MLP, scm_to_vec72
from dl_models.SPE_CNN import std_CNN
from dl_models.embeding_layer import calc_rmse, get_continuous_angle_k3, music_algorithm_k3, scm_embeding
from dl_models.vit_model import VisionTransformer


PLOT_STYLES = {
    "ViT": {"color": "r", "marker": "o"},
    "IQ-ResNet": {"color": "m", "marker": "s"},
    "SPE-CNN": {"color": "g", "marker": "^"},
    "REG-CNN": {"color": "y", "marker": "D"},
    "Learning-SPICE": {"color": "c", "marker": "v"},
    "MUSIC": {"color": "k", "marker": "*"},
}
PLOT_ORDER = ["ViT", "REG-CNN", "SPE-CNN", "IQ-ResNet", "Learning-SPICE", "MUSIC"]


def build_random_theta_once(theta_num=2000, min_delta_theta=8):
    theta_set = Create_random_k_input_theta(3, -90, 90, theta_num, min_delta_theta=min_delta_theta)
    theta_set = np.array(theta_set, dtype=np.float32)
    valid_mask = (
        (~np.isnan(theta_set).any(axis=1))
        & (np.max(theta_set, axis=1) <= 90)
        & (np.min(theta_set, axis=1) >= -90)
    )
    return theta_set[valid_mask]


def build_dataset_from_theta(rho, snap, snr, theta_set, batch_size=128):
    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    Create_datasets(dataset, k=3, theta_set=theta_set.copy(), batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
    return dataset


def load_models(device, rho):
    models = {}

    model_vit = VisionTransformer(
        embed_layer=scm_embeding(8, 768),
        embed_dim=768,
        out_dims=3,
        drop_ratio=0,
        attn_drop_ratio=0,
    ).to(device)
    if rho == 0.0:
        vit_weight = root / "Graduation" / "result" / "vit" / "vit_M_8_k_3_base" / "weight_base_ThreeSource.pth"
    else:
        vit_weight = root / "Graduation" / "result" / "vit" / "vit_M_8_k_3_base_transfer" / f"weight_transfer_ThreeSource_rho{rho}.pth"
    model_vit.load_state_dict(torch.load(vit_weight, map_location=device))
    model_vit.eval()
    models["ViT"] = model_vit

    model_iq = IQ_ResNet(num_classes=181).to(device)
    iq_weight = root / "Graduation" / "result" / "IQ_ResNet" / "ThreeSource" / f"IQ_ResNet_Gaussian_ThreeSource_rho{rho}.pth"
    model_iq.load_state_dict(torch.load(iq_weight, map_location=device))
    model_iq.eval()
    models["IQ-ResNet"] = model_iq

    model_spe = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    spe_weight = root / "Graduation" / "result" / "CNN" / "ThreeSource" / f"SPE_CNN_Gaussian_ThreeSource_rho{rho}.pth"
    model_spe.load_state_dict(torch.load(spe_weight, map_location=device))
    model_spe.eval()
    models["SPE-CNN"] = model_spe

    model_reg = CNN_Regression(out_dim=3).to(device)
    reg_weight = root / "Graduation" / "result" / "CNN" / "ThreeSource" / f"CNN_Regression_Gaussian_ThreeSource_rho{rho}.pth"
    model_reg.load_state_dict(torch.load(reg_weight, map_location=device))
    model_reg.eval()
    models["REG-CNN"] = model_reg

    model_spice = LearningSPICE_SP_MLP(M=8, out_dim=181).to(device)
    spice_weight = root / "Graduation" / "result" / "MLP" / "ThreeSource" / f"LearningSPICE_Gaussian_ThreeSource_rho{rho}.pth"
    model_spice.load_state_dict(torch.load(spice_weight, map_location=device))
    model_spice.eval()
    models["Learning-SPICE"] = model_spice

    return models


@torch.no_grad()
def evaluate_all_models(models, loader_scm, loader_y, device, snap, M=8, K=3):
    mse_accum = {name: 0.0 for name in PLOT_STYLES}
    total_batches = 0

    for (inputs_scm, _), (inputs_complex, labels_doa_y) in tqdm(zip(loader_scm, loader_y), leave=False):
        B = inputs_complex.shape[0]
        inputs_complex = inputs_complex.to(device)
        true_angles = labels_doa_y.to(device).float().view(-1, K)
        true_sorted, _ = torch.sort(true_angles, dim=1)

        inputs_scm = inputs_scm.to(device).float()
        max_vit = torch.max(torch.abs(inputs_scm.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        pred_vit = models["ViT"](inputs_scm / (max_vit + 1e-8))
        pred_vit_sorted, _ = torch.sort(pred_vit, dim=1)
        mse_accum["ViT"] += calc_rmse(pred_vit_sorted, true_sorted)

        R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

        X_reg = torch.zeros(B, 2, M, M, device=device)
        X_reg[:, 0] = R.real
        X_reg[:, 1] = R.imag
        max_reg = torch.max(torch.abs(X_reg.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        pred_reg = models["REG-CNN"](X_reg / (max_reg + 1e-8))
        pred_reg_sorted, _ = torch.sort(pred_reg, dim=1)
        mse_accum["REG-CNN"] += calc_rmse(pred_reg_sorted, true_sorted)

        X_spe = torch.zeros(B, 3, M, M, device=device)
        X_spe[:, 0] = R.real
        X_spe[:, 1] = R.imag
        X_spe[:, 2] = R.angle() / torch.pi
        max_spe = torch.max(torch.abs(R.view(B, -1)), dim=1)[0].view(B, 1, 1)
        X_spe[:, 0] = X_spe[:, 0] / (max_spe + 1e-8)
        X_spe[:, 1] = X_spe[:, 1] / (max_spe + 1e-8)
        pred_spe = get_continuous_angle_k3(models["SPE-CNN"](X_spe), K=K, radius=2)
        pred_spe_sorted, _ = torch.sort(pred_spe, dim=1)
        mse_accum["SPE-CNN"] += calc_rmse(pred_spe_sorted, true_sorted)

        inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
        rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
        pred_iq = get_continuous_angle_k3(models["IQ-ResNet"](inputs_iq / (rms_val + 1e-8)), K=K, radius=2)
        pred_iq_sorted, _ = torch.sort(pred_iq, dim=1)
        mse_accum["IQ-ResNet"] += calc_rmse(pred_iq_sorted, true_sorted)

        max_spice = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
        X_spice = scm_to_vec72(R / (max_spice + 1e-8))
        pred_spice = get_continuous_angle_k3(models["Learning-SPICE"](X_spice), K=K, radius=2)
        pred_spice_sorted, _ = torch.sort(pred_spice, dim=1)
        mse_accum["Learning-SPICE"] += calc_rmse(pred_spice_sorted, true_sorted)

        pred_music = torch.zeros(B, K, device=device)
        for i in range(B):
            R_np = R[i].cpu().numpy()
            R_np = 0.5 * (R_np + R_np.conj().T)
            pred_music[i] = torch.tensor(music_algorithm_k3(R_np, num_sources=K, M=M), device=device)
        pred_music_sorted, _ = torch.sort(pred_music, dim=1)
        mse_accum["MUSIC"] += calc_rmse(pred_music_sorted, true_sorted)

        total_batches += 1

    return {name: float(np.sqrt(mse_accum[name] / total_batches)) for name in mse_accum}


def plot_curve(snr_list, results, rho, save_path):
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0})
    plt.figure(figsize=(9, 6))
    for name in PLOT_ORDER:
        vals = results[name]
        style = PLOT_STYLES[name]
        plt.plot(
            snr_list,
            vals,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            label=name,
        )

    plt.xlabel("SNR (dB)", fontsize=14, fontweight="bold")
    plt.ylabel("RMSE (Degree)", fontsize=14, fontweight="bold")
    plt.tick_params(axis="both", labelsize=12)
    plt.legend(loc="upper right", fontsize=10, edgecolor="0.8")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path / f"RMSE_Comparison_rho{rho}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rho = 0.0
    snap = 50
    snr_list = [-20, -15, -10, -5, 0, 5, 10]
    batch_size = 128
    theta_num = 2000
    min_delta_theta = 8

    save_dir = root / "Graduation" / "result" / "plot" / "gauss" / f"M_8_K_3_rho{rho}"
    save_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(device, rho)
    theta_set = build_random_theta_once(theta_num=theta_num, min_delta_theta=min_delta_theta)
    results = {name: [] for name in PLOT_STYLES}

    for snr in snr_list:
        print(f"rho={rho} | random three-source | SNR={snr} dB | theta_num={len(theta_set)}")
        dataset = build_dataset_from_theta(rho=rho, snap=snap, snr=snr, theta_set=theta_set, batch_size=batch_size)
        loader_scm = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style="torch", input_type="scm", output_type="doa")
        loader_y = array_Dataloader(dataset, batch_size=batch_size, shuffle=False, load_style="torch", input_type="y_t", output_type="doa")

        rmse_dict = evaluate_all_models(models, loader_scm, loader_y, device, snap=snap, M=8, K=3)
        for name in results:
            results[name].append(rmse_dict[name])
        print(" | ".join([f"{name}: {rmse_dict[name]:.3f}" for name in PLOT_ORDER]))

    with open(save_dir / "three_source_random_rmse.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rho": rho,
                "snr_list": snr_list,
                "theta_num": int(len(theta_set)),
                "min_delta_theta": min_delta_theta,
                "rmse_results": results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    rmse_records = [
        {
            "rho": rho,
            "snr": snr,
            **{name: results[name][idx] for name in PLOT_ORDER},
        }
        for idx, snr in enumerate(snr_list)
    ]
    key_snrs = {nearest_value(snr_list, target) for target in [-10, 0, 10]}
    key_records = [record for record in rmse_records if record["snr"] in key_snrs]
    save_csv(rmse_records, GAUSS_DATA_DIR / "ThreeSource" / f"gauss_three_random_snr_rmse_rho{rho}.csv")
    save_csv(key_records, GAUSS_DATA_DIR / "ThreeSource" / f"gauss_three_random_snr_key_points_rho{rho}.csv")

    plot_curve(snr_list, results, rho, save_dir)


if __name__ == "__main__":
    main()
