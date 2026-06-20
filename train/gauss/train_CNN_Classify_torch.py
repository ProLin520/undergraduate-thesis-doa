import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dl_models.CNN_model import CNN_Classify

root = Path(__file__).resolve().parents[2]
ext_lib = root / "external" / "DOA_est_Master-master"

if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root.parent))  # 添加 doa_estimation 根目录

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets


def prepare_2ch_input(inputs_complex, T, device):
    """将复数快拍转换为 2 通道 SCM 特征: Real, Imag """
    B, M, _ = inputs_complex.shape
    R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T

    X = torch.zeros(B, 2, M, M, device=device)
    X[:, 0, :, :] = R.real
    X[:, 1, :, :] = R.imag

    # 归一化实部和虚部
    max_val = torch.max(torch.abs(R.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
    X = X / (max_val + 1e-8)
    return X

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 0.0
    snap = 200
    print(f" CNN_Classify 高斯训练 | 2通道模式 | Rho={rho}")

    model = CNN_Classify(num_classes=181).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
    save_dir = os.path.join(root, 'result', 'CNN', 'SingleSource')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'CNN_Classify_Gaussian_rho{rho}.pth')

    best_val_acc = 0.0

    for epoch in range(30):
        # --- A. 训练阶段 ---
        dataset.clear()
        theta_train = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=4000)
        Create_datasets(dataset, k=1, theta_set=theta_train, batch_size=128, snap=snap, snr=(-20, 20))
        train_loader = array_Dataloader(dataset, batch_size=128, shuffle=True, load_style='torch', input_type='y_t',
                                        output_type='spatial_sp')

        model.train()
        train_loss, train_steps = 0.0, 0
        for inputs_complex, labels_onehot in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False):
            inputs_complex = inputs_complex.to(device)
            labels = torch.argmax(labels_onehot, dim=1).long().to(device)

            X = prepare_2ch_input(inputs_complex, snap, device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        # --- B. 验证阶段 (完整版逻辑) ---
        dataset.clear()
        theta_val = Create_random_k_input_theta(k=1, start_angle=-90, end_angle=90, theta_num=1000)
        Create_datasets(dataset, k=1, theta_set=theta_val, batch_size=128, snap=snap, snr=(-20, 20))
        val_loader = array_Dataloader(dataset, batch_size=128, shuffle=False, load_style='torch', input_type='y_t',
                                      output_type='spatial_sp')

        model.eval()
        correct, val_total = 0, 0
        with torch.no_grad():
            for inputs_complex, labels_onehot in val_loader:
                inputs_complex = inputs_complex.to(device)
                labels = torch.argmax(labels_onehot, dim=1).long().to(device)

                X_val = prepare_2ch_input(inputs_complex, snap, device)
                outputs = model(X_val)
                predicted = torch.argmax(outputs, dim=1)

                val_total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / val_total if val_total > 0 else 0
        scheduler.step()

        print(
            f"[Epoch {epoch + 1}/30] Loss: {train_loss / train_steps:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" 已保存当前最优模型: {val_acc:.4f}")

    print(f"\n 训练完成！最终模型路径: {save_path}")


if __name__ == "__main__":
    main()