import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from pathlib import Path

root = Path(__file__).resolve().parents[3]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.SPE_CNN import std_CNN
from dl_models.embeding_layer import get_continuous_angle


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 100
    print(f" 启动 SPE-CNN (文献经典) 在线高斯训练 | Rho={rho} | Snap={snap}")

    # 1. 初始化 SPE-CNN (输入3通道, 8阵元, 181分类)
    model = std_CNN(3, 8, 181, sp_mode=True, start_angle=-90, end_angle=90).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_rmse = float('inf')
    # 保存路径
    save_dir = os.path.join(root, 'Graduation', 'result', 'CNN', 'SingleSource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'SPE_CNN_Gaussian_8ULA_K1_rho{rho}.pth')

    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)

    for epoch in range(30):
        # --- 在线生成训练集 ---
        dataset.clear()
        theta_train = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=4000)
        Create_datasets(dataset, k=1, theta_set=theta_train, batch_size=128, snap=snap, snr=(-20, 10))
        train_loader = array_Dataloader(dataset, batch_size=128, shuffle=True, load_style='torch',
                                        input_type='y_t', output_type='spatial_sp')

        model.train()
        train_loss = 0.0
        for step, (inputs_complex, labels_onehot) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Train")):
            B, M, T = inputs_complex.shape
            inputs_complex = inputs_complex.to(device)
            labels = torch.argmax(labels_onehot, dim=1).long().to(device)

            #  动态构造 SPE-CNN 专属的 3 通道协方差特征
            R_complex = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T
            X_spe = torch.zeros(B, 3, M, M, device=device)
            X_spe[:, 0, :, :] = R_complex.real
            X_spe[:, 1, :, :] = R_complex.imag
            X_spe[:, 2, :, :] = R_complex.angle() / torch.pi

            # 归一化
            max_spe = torch.max(torch.abs(R_complex.view(B, -1)), dim=1)[0].view(B, 1, 1)
            X_spe[:, 0, :, :] /= (max_spe + 1e-8)
            X_spe[:, 1, :, :] /= (max_spe + 1e-8)

            optimizer.zero_grad()
            outputs = model(X_spe)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 在线生成验证集 ---
        dataset.clear()
        theta_val = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=1000)
        Create_datasets(dataset, k=1, theta_set=theta_val, batch_size=128, snap=snap, snr=(-20, 10))
        val_loader = array_Dataloader(dataset, batch_size=128, shuffle=False, load_style='torch',
                                      input_type='y_t', output_type='spatial_sp')

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for inputs_complex, labels_onehot in val_loader:
                B, M, T = inputs_complex.shape
                inputs_complex = inputs_complex.to(device)
                # 取出真实角度（依然通过 argmax，但转为 float 的真实角度）
                labels_idx = torch.argmax(labels_onehot, dim=1)
                true_angles = (labels_idx.float() - 90.0).to(device)

                R_complex = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T
                X_spe = torch.zeros(B, 3, M, M, device=device)
                X_spe[:, 0, :, :] = R_complex.real
                X_spe[:, 1, :, :] = R_complex.imag
                X_spe[:, 2, :, :] = R_complex.angle() / torch.pi
                max_spe = torch.max(torch.abs(R_complex.view(B, -1)), dim=1)[0].view(B, 1, 1)
                X_spe[:, 0, :, :] /= (max_spe + 1e-8)
                X_spe[:, 1, :, :] /= (max_spe + 1e-8)

                outputs = model(X_spe)

                # 🌟 调用连续角度提取函数
                pred_angles = get_continuous_angle(outputs, radius=2)

                # 算 MSE
                mse = torch.mean((pred_angles - true_angles) ** 2)
                val_loss += mse.item()
                val_steps += 1

        # 计算 RMSE
        val_rmse = np.sqrt(val_loss / val_steps)
        scheduler.step()

        print(
            f" -> Train Loss: {train_loss / (step + 1):.4f} | Val RMSE: {val_rmse:.4f}° | LR: {optimizer.param_groups[0]['lr']}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 已保存最优 SPE-CNN 模型 (RMSE: {best_val_rmse:.4f}°)")


if __name__ == "__main__":
    main()