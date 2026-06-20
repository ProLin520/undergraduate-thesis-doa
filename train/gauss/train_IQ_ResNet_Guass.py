import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
from tqdm import tqdm
import sys
from pathlib import Path

# 确保路径对齐
root = Path(__file__).resolve().parents[3]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.embeding_layer import get_continuous_angle


def train_and_eval_iq_resnet_online(device='cuda', rho=0.0, snap=200):
    print(f"🚀 启动 IQ-ResNet 高斯流型在线训练 (Rho={rho}, 引入标签平滑)")

    # 初始化 IQ-ResNet (181 分类任务)
    model = IQ_ResNet(num_classes=181).to(device)

    # 🌟 核心优化：引入标签平滑，防止 -20dB 下模型对纯噪声过度自信拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    best_val_rmse = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        train_loss = 0.0

        # 1. 在线生成本轮的训练集
        dataset.clear()
        theta_train = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=4000)
        Create_datasets(dataset, k=1, theta_set=theta_train, batch_size=128, snap=snap, snr=(-20, 10))
        train_loader = array_Dataloader(dataset, batch_size=128, shuffle=True, load_style='torch',
                                        input_type='y_t', output_type='spatial_sp')

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] 正在训练...")
        for step, (inputs_complex, labels_onehot) in enumerate(tqdm(train_loader, leave=False)):
            inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()

            # 🌟 RMS 均方根功率归一化 (保留此优秀设计)
            rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
            inputs_iq = inputs_iq / (rms_val + 1e-8)
            inputs = inputs_iq.to(device)

            labels = torch.argmax(labels_onehot, dim=1).long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 2. 在线生成验证集
        dataset.clear()
        theta_val = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=1000)
        Create_datasets(dataset, k=1, theta_set=theta_val, batch_size=128, snap=snap, snr=(-20, 10))
        val_loader = array_Dataloader(dataset, batch_size=128, shuffle=False, load_style='torch',
                                      input_type='y_t', output_type='spatial_sp')

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for inputs_complex, labels_onehot in tqdm(val_loader, leave=False):
                inputs_iq = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()

                rms_val = torch.sqrt(torch.mean(inputs_iq ** 2, dim=(2, 3), keepdim=True))
                inputs_iq = inputs_iq / (rms_val + 1e-8)

                inputs = inputs_iq.to(device)
                # 取出真实角度
                labels_idx = torch.argmax(labels_onehot, dim=1)
                true_angles = (labels_idx.float() - 90.0).to(device)

                outputs = model(inputs)

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
            f"-> Train Loss: {train_loss / (step + 1):.4f} | Val RMSE: {val_rmse:.4f}° | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"⭐ 已更新最优 IQ-ResNet 权重 (RMSE: {best_val_rmse:.4f}°)")

        # 3. 保存模型
    model.load_state_dict(best_model_wts)
    save_dir = r"/result/IQ_ResNet/SingleSource"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'IQ_ResNet_Gaussian_rho{rho}.pth')

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 训练结束！最优模型 (RMSE: {best_val_rmse:.4f}°) 已精准保存至: {save_path}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_and_eval_iq_resnet_online(device=device, rho=1.0, snap=100)