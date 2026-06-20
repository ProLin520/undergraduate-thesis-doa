import sys
import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[2]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"

if str(ext_lib) not in sys.path:
    sys.path = [str(ext_lib), str(root)] + sys.path
# Fix utils conflict if present
if 'utils' in sys.modules:
    del sys.modules['utils']

from data_creater.file_dataloader import file_array_Dataloader
from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding
from Graduation.utils.metrics_utils import GAUSS_DATA_DIR, save_csv


@torch.no_grad()
def evaluate_model_on_loader(model, data_loader, loss_function, device, k):
    model.eval()
    accu_loss = 0.0
    total_samples = 0
    success_count = 0
    success_threshold = 2.0  # Success judgment: absolute error < 2 degrees

    # Collate all absolute errors for ECDF
    all_abs_errors = []

    data_loader = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, data in enumerate(data_loader):
        input, labels = data
        input, labels = input.to(device), labels.to(device)

        # CRITICAL: Dynamic Physical-level Normalization Patch
        batch_size = input.shape[0]
        # Calculate max absolute value per sample for [0, 1] scaling
        max_vals = torch.max(torch.abs(input.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        input = input / (max_vals + 1e-8)

        # DOA ViT inference
        pred = model(input)

        # Force dimension alignment to prevent disastrous PyTorch broadcasting (2D vs 1D label)
        pred = pred.view(-1, k)
        labels = labels.view(-1, k)

        loss = loss_function(pred, labels)
        accu_loss += loss.item() * input.size(0)
        total_samples += input.size(0)

        # Collate absolute errors
        errors = torch.abs(pred - labels)
        success_count += torch.sum(errors < success_threshold).item()
        all_abs_errors.extend(errors.cpu().numpy().flatten())

    rmse = np.sqrt(accu_loss / total_samples)
    success_rate = success_count / total_samples
    return rmse, success_rate, np.array(all_abs_errors)


def plot_and_save_triple_results(snrs, rmse_b_id, rmse_b_mu, rmse_t_mu,
                                 errors_b_id, errors_b_mu, errors_t_mu, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot RMSE Comparison Curve
    plt.figure(figsize=(9, 6))
    # Grey line (Lower Baseline): Perfect Array
    plt.plot(snrs, rmse_b_id, marker='d', linestyle=':', linewidth=2, color='#808080',
             label='Base Model (Perfect Array rho=0)')
    # Blue line: Mutual Coupling Damage
    plt.plot(snrs, rmse_b_mu, marker='o', linestyle='--', linewidth=2, color='#1f77b4',
             label='Base Model (Error Array rho=1)')
    # Red line: Repaired via Transfer
    plt.plot(snrs, rmse_t_mu, marker='^', linestyle='-', linewidth=2, color='#d62728',
             label='Transfer Model (Error Array rho=1)')

    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('RMSE (Degrees)', fontsize=13)
    # plt.title('RMSE vs. SNR Comparison (Three Baselines)', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(save_dir, 'Trans_Comparison_RMSE.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Plot ECDF Comparison Curve (New Required Metric)
    plt.figure(figsize=(9, 6))
    def get_cumulative_error(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    # Calculate CDFs for all three error sets
    x_b_id, y_b_id = get_cumulative_error(errors_b_id)
    x_b_mu, y_b_mu = get_cumulative_error(errors_b_mu)
    x_t_mu, y_t_mu = get_cumulative_error(errors_t_mu)

    # Plot CDFs with matching styles/colors
    plt.plot(x_b_id, y_b_id, linestyle=':', linewidth=2.5, color='#808080', label='Base Model (Perfect Array rho=0)')
    plt.plot(x_b_mu, y_b_mu, linestyle='--', linewidth=2, color='#1f77b4', label='Base Model (Error Array rho=1)')
    plt.plot(x_t_mu, y_t_mu, linestyle='-', linewidth=2, color='#d62728', label='Transfer Model (Error Array rho=1)')

    # Critical Viewport: Journal standard often cuts at 5 or 10 degrees error
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.xlabel('Absolute Error (Degrees)', fontsize=13)
    plt.ylabel('Cumulative Probability (ECDF)', fontsize=13)
    # plt.title('DOA Estimation Error Cumulative Distribution (Three Cases)', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.savefig(os.path.join(save_dir, 'Trans_Comparison_ECDF.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Triple comparison visualization (RMSE, ECDF) saved to:\n{save_dir}")


def main(args):
    print(f"========== Start Triple Test (M={args.M}, K={args.k}, Rho_M={args.rho}) ==========")
    embeding_dim = 768

    # --- Robust Model Loading ---
    def load_safe_model(weight_path_raw):
        # Clean potential '\xa0' from string copy
        weight_path = weight_path_raw.replace('\xa0', '').strip()

        m = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
        if not os.path.exists(weight_path):
            print(f"❌ Cannot find weights: {weight_path}")
            return None
        m.load_state_dict(torch.load(weight_path, map_location=args.device), strict=True)
        m.to(args.device)
        return m

    # Strictly load the two required models from explicitly defined paths
    base_model = load_safe_model(os.path.join(args.base_root, 'weight_base_best.pth'))
    transfer_model = load_safe_model(os.path.join(args.trans_root, 'weight_transfer_best.pth'))

    if base_model is None or transfer_model is None: return

    loss_function = torch.nn.MSELoss()

    # Store results dynamically, filtering out non-existent SNR files
    tested_snrs_aligned = []

    # Triple Data Lists (im=Ideal ModelData, mm=Mutual ModelData, tm=Transfer ModelData)
    results_rmse = {'b_id': [], 'b_mu': [], 't_mu': []}
    global_errors = {'b_id': [], 'b_mu': [], 't_mu': []}

    # Iterate required SNRs, but only plot those present in both mutual AND ideal folders
    for snr in args.snrs:
        # Resolve target paths explicitly defined by user
        file_mutual = os.path.join(args.mutual_data_root, f'test_{args.test_name}_snr_{snr}.npz')
        file_ideal = os.path.join(args.ideal_data_root, f'test_{args.test_name}_snr_{snr}.npz')

        # Critical Check: Plotter breaks if mutual exists but ideal is missing (shaping mismatch)
        if not (os.path.exists(file_mutual) and os.path.exists(file_ideal)):
            print(f" SNR {snr} skipped: Missing mutual OR ideal data file.")
            continue

        print(f"\n Aligned Load: SNR {snr} dB")
        tested_snrs_aligned.append(snr)

        # Load aligned dataloaders
        mutual_dataloader = file_array_Dataloader(file_mutual, batch_size=256, shuffle=False,
                                                  load_style='torch', input_type='scm', output_type='doa')
        ideal_dataloader = file_array_Dataloader(file_ideal, batch_size=256, shuffle=False,
                                                 load_style='torch', input_type='scm', output_type='doa')

        # 1. Test_Rho0.0 BASE Model on IDEAL (rho=0) Data (Lower grey baseline)
        b_id_rmse, _, b_id_err = evaluate_model_on_loader(base_model, ideal_dataloader, loss_function, args.device,
                                                          args.k)

        # 2. Test_Rho0.0 BASE Model on MUTUAL (rho=1) Data (Blue line)
        b_mu_rmse, _, b_mu_err = evaluate_model_on_loader(base_model, mutual_dataloader, loss_function, args.device,
                                                          args.k)

        # 3. Test_Rho0.0 TRANSFER Model on MUTUAL (rho=1) Data (Red line)
        t_mu_rmse, _, t_mu_err = evaluate_model_on_loader(transfer_model, mutual_dataloader, loss_function, args.device,
                                                          args.k)

        # Append aligned results standard metrics (CSV/Plotting)
        results_rmse['b_id'].append(b_id_rmse)
        results_rmse['b_mu'].append(b_mu_rmse)
        results_rmse['t_mu'].append(t_mu_rmse)

        global_errors['b_id'].extend(b_id_err)
        global_errors['b_mu'].extend(b_mu_err)
        global_errors['t_mu'].extend(t_mu_err)

        print(
            f"  NR: {snr:3d} dB | Ideal(G): {b_id_rmse:5.2f}° | Broken(B): {b_mu_rmse:5.2f}° | Fixed(R): {t_mu_rmse:5.2f}°")

    # Final Visualization Check Standard Metric
    if tested_snrs_aligned:
        plot_and_save_triple_results(tested_snrs_aligned,
                                     results_rmse['b_id'], results_rmse['b_mu'], results_rmse['t_mu'],
                                     np.array(global_errors['b_id']), np.array(global_errors['b_mu']),
                                     np.array(global_errors['t_mu']),
                                     save_dir=args.save_root)
        snrs = np.asarray(tested_snrs_aligned)
        rmse_b_id = np.asarray(results_rmse['b_id'], dtype=float)
        rmse_b_mu = np.asarray(results_rmse['b_mu'], dtype=float)
        rmse_t_mu = np.asarray(results_rmse['t_mu'], dtype=float)
        improvement_percent = np.where(rmse_b_mu != 0, (rmse_b_mu - rmse_t_mu) / rmse_b_mu * 100, np.nan)
        rmse_records = [{"snr": snr, "base_on_rho0_rmse": b_id, "base_on_rho1_rmse": b_mu, "transfer_on_rho1_rmse": t_mu, "improvement_percent": imp} for snr, b_id, b_mu, t_mu, imp in zip(snrs, rmse_b_id, rmse_b_mu, rmse_t_mu, improvement_percent)]
        quantile_records = [
            {"group": "base_on_rho0", "mean_abs_error": float(np.mean(np.asarray(global_errors['b_id'], dtype=float))), "p90_abs_error": float(np.percentile(np.asarray(global_errors['b_id'], dtype=float), 90))},
            {"group": "base_on_rho1", "mean_abs_error": float(np.mean(np.asarray(global_errors['b_mu'], dtype=float))), "p90_abs_error": float(np.percentile(np.asarray(global_errors['b_mu'], dtype=float), 90))},
            {"group": "transfer_on_rho1", "mean_abs_error": float(np.mean(np.asarray(global_errors['t_mu'], dtype=float))), "p90_abs_error": float(np.percentile(np.asarray(global_errors['t_mu'], dtype=float), 90))},
        ]
        save_csv(rmse_records, GAUSS_DATA_DIR / "SingleSource" / "gauss_single_transfer_rmse.csv")
        save_csv(quantile_records, GAUSS_DATA_DIR / "SingleSource" / "gauss_single_transfer_error_quantiles.csv")
    else:
        print(" Aligned simulation failed: No common SNR data found between folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Scenario standard metric
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--test_name', type=str,
                        default="random_input")  # Standard file prefix (Create_datasets standard metric)

    # Update target SNRs based on standard boundaries
    parser.add_argument('--snrs', default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')

    # Path Alignment Standard metric
    base_path = r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_1_base"
    trans_path = r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_1_base_transfer"
    save_path = r"D:\Python\Project\doa_estimation\Graduation\result\plot\gauss\M_8_K_1_rho0.0"

    # Hardcoded path explicitly defined by user
    mutual_data_path = r"D:\Python\Project\doa_estimation\Graduation\data\ViT\ViT_M_8_K_1\M_8_k_1_test_rho1.0"
    ideal_data_path = r"D:\Python\Project\doa_estimation\Graduation\data\ViT\ViT_M_8_K_1\M_8_k_1_test_rho0.0"

    parser.add_argument('--base_root', type=str, default=base_path)
    parser.add_argument('--trans_root', type=str, default=trans_path)
    parser.add_argument('--save_root', type=str, default=save_path)
    parser.add_argument('--mutual_data_root', type=str, default=mutual_data_path)
    parser.add_argument('--ideal_data_root', type=str, default=ideal_data_path)

    args = parser.parse_args()
    main(args)
