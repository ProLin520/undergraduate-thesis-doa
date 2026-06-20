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
from dl_models.MLP import LearningSPICE_SP_MLP, scm_to_vec72
from dl_models.embeding_layer import get_continuous_angle


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 100
    M = 8
    print(f"🚀 启动 Learning-SPICE (矩阵重构去噪 MLP) | Rho={rho}")

    model = LearningSPICE_SP_MLP(M=8, out_dim=181).to(device)  # 新模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    save_dir = os.path.join(root, 'result', 'MLP', 'SingleSource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'LearningSPICE_Gaussian_rho{rho}.pth')

    best_val_rmse = float('inf')

    for epoch in range(30):
        # --- 训练阶段 ---
        dataset.clear()
        theta_train = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=4000)
        Create_datasets(dataset, k=1, theta_set=theta_train, batch_size=128, snap=snap, snr=(-20, 10))
        train_loader = array_Dataloader(dataset, batch_size=128, shuffle=True, load_style='torch', input_type='y_t',
                                        output_type='spatial_sp')

        model.train()
        train_loss, train_steps = 0.0, 0
        for inputs_complex, labels_onehot in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False):
            inputs_complex = inputs_complex.to(device)
            # 🌟 核心：Onehot 转为分类索引
            labels = torch.argmax(labels_onehot, dim=1).long().to(device)
            B = inputs_complex.shape[0]

            R_noisy = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap
            max_val = torch.max(torch.abs(R_noisy.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
            R_noisy = R_noisy / (max_val + 1e-8)
            X_input = scm_to_vec72(R_noisy)

            optimizer.zero_grad()
            outputs = model(X_input)
            loss = criterion(outputs, labels)  # 🌟 直接算交叉熵
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        # --- 验证阶段 ---
        dataset.clear()
        theta_val = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=1000)
        Create_datasets(dataset, k=1, theta_set=theta_val, batch_size=128, snap=snap, snr=(-20, 10))
        val_loader = array_Dataloader(dataset, batch_size=128, shuffle=False, load_style='torch', input_type='y_t',
                                      output_type='spatial_sp')

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            # 🌟 修复 1：这里应该是遍历 val_loader，且不需要 tqdm 描述为 Train
            for inputs_complex, labels_onehot in val_loader:
                inputs_complex = inputs_complex.to(device)
                B = inputs_complex.shape[0]

                R_noisy = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap
                max_val = torch.max(torch.abs(R_noisy.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
                R_noisy = R_noisy / (max_val + 1e-8)
                X_input = scm_to_vec72(R_noisy)

                # 🌟 修复 2：删除了验证集里不需要的 optimizer.zero_grad()
                outputs = model(X_input)

                # 计算连续角度的 RMSE
                labels_idx = torch.argmax(labels_onehot, dim=1)
                true_angles = (labels_idx.float() - 90.0).to(device)

                pred_angles = get_continuous_angle(outputs, radius=2)

                mse = torch.mean((pred_angles - true_angles) ** 2)
                val_loss += mse.item()
                val_steps += 1

        val_rmse = np.sqrt(val_loss / val_steps)
        scheduler.step()
        print(f"-> Val RMSE: {val_rmse:.4f}°")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()