import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import sys
from pathlib import Path

ext_lib = r"D:\Python\Project\doa_estimation\Graduation\external\DOA_est_Master-master"
proj_root = r"D:\Python\Project\doa_estimation"

if ext_lib not in sys.path:
    sys.path.insert(0, ext_lib)
if proj_root not in sys.path:
    sys.path.insert(1, proj_root)

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import get_continuous_angle_k3, calc_rmse


VAL_SNR_LIST = [-20, -15, -10, -5, 0, 5, 10]
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


def evaluate_iq_snr_curve(model, val_loaders, device):
    model.eval()
    snr_rmse = {}

    with torch.no_grad():
        for snr, loader in val_loaders.items():
            val_loss = 0.0
            val_steps = 0

            for inputs_complex, labels_doa in loader:
                inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
                rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
                inputs = (inputs_iq / (rms_val + 1e-8)).to(device)

                true_angles = labels_doa.float().to(device).view(-1, 3)
                outputs = model(inputs)
                pred_angles = get_continuous_angle_k3(outputs, K=3, radius=2)

                val_loss += calc_rmse(pred_angles, true_angles)
                val_steps += 1

            snr_rmse[snr] = np.sqrt(val_loss / val_steps)

    avg_rmse = float(np.mean(list(snr_rmse.values())))
    return avg_rmse, snr_rmse


def train_and_eval_iq_resnet_online(device='cuda', rho=0.0, snap=50):
    print(f"🚀 启动 IQ-ResNet 高斯流型在线训练 (Rho={rho})")

    model = IQ_ResNet(num_classes=181).to(device)

    pos_weight = torch.tensor([59.3], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    num_epochs = 100
    best_val_rmse = float('inf')

    save_dir = r"/result/IQ_ResNet/ThreeSource"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"IQ_ResNet_Gaussian_ThreeSource_rho{rho}.pth")

    # ===== 训练集容器：每轮重采样 =====
    train_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)

    # ===== 固定验证集：只生成一次 =====
    val_loaders = build_snr_val_loaders(rho=rho, snap=snap, batch_size=128, theta_num=2000, min_delta_theta=5)

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        train_dataset.clear()
        cfg = get_stage_cfg(epoch)
        theta_train = Create_random_k_input_theta(k=3, start_angle=-90, end_angle=90, theta_num=10000,
                                                  min_delta_theta=cfg["min_delta_theta"])
        theta_train = np.array(theta_train)
        valid_mask = (~np.isnan(theta_train).any(axis=1)) & (np.max(theta_train, axis=1) <= 90) & (
                    np.min(theta_train, axis=1) >= -90)
        theta_train = theta_train[valid_mask]
        Create_datasets(train_dataset, k=3, theta_set=theta_train, batch_size=128, snap=snap, snr=cfg["snr"],
                        shared_snr=True)

        train_loader = array_Dataloader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            load_style='torch',
            input_type='y_t',
            output_type='spatial_sp'
        )

        model.train()
        train_loss, train_steps = 0.0, 0

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] 正在训练 IQ-ResNet...")
        for inputs_complex, labels_onehot in tqdm(train_loader, leave=False):
            inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
            rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
            inputs = (inputs_iq / (rms_val + 1e-8)).to(device)

            labels = labels_onehot.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        # ---------- 验证 ----------
        avg_val_rmse, snr_rmse = evaluate_iq_snr_curve(model, val_loaders, device)
        scheduler.step(avg_val_rmse)

        print(f" -> Train Loss: {train_loss / train_steps:.4f} | ValAvg: {avg_val_rmse:.4f}° "
              f"| V@-20: {snr_rmse[-20]:.4f}° |  V@-10: {snr_rmse[-10]:.4f}° | V@0: {snr_rmse[0]:.4f}° "
              f"| V@10: {snr_rmse[10]:.4f}° | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), save_path)
            print(
                f"⭐ 已保存最优模型 | Avg: {best_val_rmse:.4f}° | [-20,-10,0,10] = "
                f"[{snr_rmse[-20]:.3f}, {snr_rmse[-10]:.3f}, {snr_rmse[0]:.3f}, {snr_rmse[10]:.3f}]")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_and_eval_iq_resnet_online(device=device, rho=1.0, snap=50)