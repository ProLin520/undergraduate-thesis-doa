import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

root = Path(__file__).resolve().parents[2]
ext_lib = root / "external" / "DOA_est_Master-master"

if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root.parent))

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.CNN_model import CNN_Regression
from dl_models.embeding_layer import calc_rmse


VAL_SNR_LIST = [-20, -15, -10, -5, 0, 5, 10]


def prepare_2ch_input(inputs_complex, T, device):
    B, M, _ = inputs_complex.shape
    R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T

    X = torch.zeros(B, 2, M, M, device=device)
    X[:, 0, :, :] = R.real
    X[:, 1, :, :] = R.imag

    max_val = torch.max(torch.abs(X.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
    X = X / (max_val + 1e-8)
    return X


def build_snr_val_loaders(rho, snap, batch_size=128, theta_num=2000, min_delta_theta=5):
    theta_val = Create_random_k_input_theta(k=3, start_angle=-90, end_angle=90, theta_num=theta_num, min_delta_theta=min_delta_theta)
    theta_val = np.array(theta_val)
    valid_mask_val = (~np.isnan(theta_val).any(axis=1)) & (np.max(theta_val, axis=1) <= 90) & (np.min(theta_val, axis=1) >= -90)
    theta_val = theta_val[valid_mask_val]

    val_loaders = {}
    for snr in VAL_SNR_LIST:
        val_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
        Create_datasets(val_dataset, k=3, theta_set=theta_val.copy(), batch_size=batch_size, snap=snap, snr=snr, shared_snr=True)
        val_loaders[snr] = array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False, load_style='torch', input_type='y_t', output_type='doa')
    return val_loaders


def get_stage_cfg(epoch):
    if epoch < 30:
        return {"snr": (0, 10), "min_delta_theta": 10}
    elif epoch < 70:
        return {"snr": (-10, 10), "min_delta_theta": 7}
    else:
        return {"snr": (-20, 10), "min_delta_theta": 5}


def evaluate_reg_snr_curve(model, val_loaders, device, snap):
    model.eval()
    snr_rmse = {}

    with torch.no_grad():
        for snr, loader in val_loaders.items():
            val_loss = 0.0
            val_steps = 0

            for inputs_complex, labels_doa in loader:
                inputs_complex = inputs_complex.to(device)
                true_angles = labels_doa.float().to(device).view(-1, 3)

                X_val = prepare_2ch_input(inputs_complex, snap, device)
                pred_angles = model(X_val)

                val_loss += calc_rmse(pred_angles, true_angles)
                val_steps += 1

            snr_rmse[snr] = np.sqrt(val_loss / val_steps)

    avg_rmse = float(np.mean(list(snr_rmse.values())))
    return avg_rmse, snr_rmse


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 0.0
    snap = 50

    print(f"🚀 启动 REG-CNN 高斯训练 | Rho={rho}")

    model = CNN_Regression(out_dim=3).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    train_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    val_loaders = build_snr_val_loaders(rho=rho, snap=snap, batch_size=128, theta_num=2000, min_delta_theta=5)

    save_dir = os.path.join(root, 'result', 'CNN', 'ThreeSource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'CNN_Regression_Gaussian_ThreeSource_rho{rho}.pth')

    best_val_rmse = float('inf')
    num_epochs = 100

    for epoch in range(num_epochs):
        train_dataset.clear()
        cfg = get_stage_cfg(epoch)

        theta_train = Create_random_k_input_theta(k=3, start_angle=-90, end_angle=90, theta_num=10000, min_delta_theta=cfg["min_delta_theta"])
        theta_train = np.array(theta_train)
        valid_mask = (~np.isnan(theta_train).any(axis=1)) & (np.max(theta_train, axis=1) <= 90) & (np.min(theta_train, axis=1) >= -90)
        theta_train = theta_train[valid_mask]

        Create_datasets(train_dataset, k=3, theta_set=theta_train, batch_size=128, snap=snap, snr=cfg["snr"], shared_snr=True)

        train_loader = array_Dataloader(train_dataset, batch_size=128, shuffle=True, load_style='torch', input_type='y_t', output_type='doa')

        model.train()
        train_loss = 0.0
        train_steps = 0

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] 正在训练 REG-CNN...")
        for inputs_complex, labels_doa in tqdm(train_loader, leave=False):
            inputs_complex = inputs_complex.to(device)
            true_angles = labels_doa.float().to(device).view(-1, 3)

            X_train = prepare_2ch_input(inputs_complex, snap, device)

            optimizer.zero_grad()
            pred_angles = model(X_train)

            pred_sorted, _ = torch.sort(pred_angles, dim=1)
            true_sorted, _ = torch.sort(true_angles, dim=1)

            loss = criterion(pred_sorted, true_sorted)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_val_rmse, snr_rmse = evaluate_reg_snr_curve(model, val_loaders, device, snap)
        scheduler.step(avg_val_rmse)

        print(f" -> Train Loss: {train_loss / train_steps:.4f} | ValAvg: {avg_val_rmse:.4f}° | V@10: {snr_rmse[10]:.4f}° | V@0: {snr_rmse[0]:.4f}° | V@-10: {snr_rmse[-10]:.4f}° | V@-20: {snr_rmse[-20]:.4f}° | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 已保存最优模型 | Avg: {best_val_rmse:.4f}° | [-20,-10,0,10] = [{snr_rmse[-20]:.3f}, {snr_rmse[-10]:.3f}, {snr_rmse[0]:.3f}, {snr_rmse[10]:.3f}]")

    print(f"\n✅ REG-CNN 训练结束！最优模型已保存至: {save_path}")


if __name__ == "__main__":
    main()