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


def prepare_2ch_input(inputs_complex, T, device):
    """严格还原 2 通道 SCM 特征: Real, Imag"""
    B, M, _ = inputs_complex.shape
    R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T

    X = torch.zeros(B, 2, M, M, device=device)
    X[:, 0, :, :] = R.real
    X[:, 1, :, :] = R.imag

    # Keras 原版归一化
    max_val = torch.max(torch.abs(X.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
    X = X / (max_val + 1e-8)
    return X


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 100
    print(f" 启动 CNN_Regression 高斯训练 | 2通道模式 | Rho={rho}")

    model = CNN_Regression().to(device)
    criterion = nn.MSELoss()  # 回归任务使用均方误差
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    save_dir = os.path.join(root, 'result', 'CNN', 'SingleSource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'CNN_Regression_Gaussian_rho{rho}.pth')

    best_val_rmse = float('inf')  # 越小越好

    for epoch in range(30):
        # --- A. 训练阶段 ---
        dataset.clear()
        theta_train = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=4000)
        Create_datasets(dataset, k=1, theta_set=theta_train, batch_size=128, snap=snap, snr=(-20, 10))
        train_loader = array_Dataloader(dataset, batch_size=128, shuffle=True, load_style='torch', input_type='y_t',
                                        output_type='doa')

        model.train()
        train_loss, train_steps = 0.0, 0
        for inputs_complex, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False):
            inputs_complex = inputs_complex.to(device)
            labels = labels.to(device).float().view(-1, 1)  # 回归需要 (Batch, 1)

            X = prepare_2ch_input(inputs_complex, snap, device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        # --- B. 验证阶段 ---
        dataset.clear()
        theta_val = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=1000)
        Create_datasets(dataset, k=1, theta_set=theta_val, batch_size=128, snap=snap, snr=(-20, 10))
        val_loader = array_Dataloader(dataset, batch_size=128, shuffle=False, load_style='torch', input_type='y_t',
                                      output_type='doa')

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for inputs_complex, labels in val_loader:
                inputs_complex = inputs_complex.to(device)
                labels = labels.to(device).float().view(-1, 1)

                X_val = prepare_2ch_input(inputs_complex, snap, device)
                outputs = model(X_val)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_steps += 1

        # 计算 RMSE
        val_rmse = np.sqrt(val_loss / val_steps)
        scheduler.step()

        print(
            f"[Epoch {epoch + 1}/30] Train Loss: {train_loss / train_steps:.4f} | Val RMSE: {val_rmse:.2f}° | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最优模型 (RMSE 最小)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            print(f" 已保存当前最优模型: {val_rmse:.2f}°")

    print(f"\n 训练完成！最终模型路径: {save_path}")


if __name__ == "__main__":
    main()