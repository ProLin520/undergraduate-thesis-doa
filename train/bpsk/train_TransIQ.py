import numpy as np
import argparse
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# 路径配置
root = Path(__file__).resolve().parents[4]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))

from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from utils.early_stop import EarlyStopping


# ==========================================
# 🌟 核心修改 1：使用离线 Dataset 读取 SCM 数据
# ==========================================
class SCM_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)  # 已经是 (N, 2, 8, 8) 格式
        onehot_labels = np.load(label_path)
        self.labels = np.argmax(onehot_labels, axis=1).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(data_loader):
        input_data = input_data.to(device)
        labels = labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        # 加上 1e-8 防止除以 0
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        # 🌟 核心修改 2：去除 permute，因为离线数据已经是 (B, 2, 8, 8)
        pred = model(input_data)

        loss = loss_function(pred, labels)

        loss.backward()
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        accu_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device):
    model.eval()
    accu_loss = 0.0

    # 🌟 修复：在套用 tqdm 之前，提前获取真实的样本总数
    total_samples = len(data_loader.dataset)

    # 将进度条变量名改为 loop，避免覆盖原始的 data_loader
    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = input_data.to(device)
        labels = labels.to(device)

        pred = model(input_data)
        # 取最大概率的索引作为预测类别 (0 到 180)
        pred_class = torch.argmax(pred, dim=1)

        # 计算当前 batch 的预测角度和真实角度的平方误差和
        # (类别索引直接对应真实角度: 0 -> -90, 180 -> +90，差值即为角度差)
        batch_sq_err = torch.sum((pred_class.float() - labels.float()) ** 2).item()
        accu_loss += batch_sq_err

    # 计算全局 RMSE
    rmse_degrees = np.sqrt(accu_loss / total_samples)
    return rmse_degrees


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==========================================
    # 🌟 核心修改 3：指向纯净的 SCM_Single_Source_Rho0.0 路径
    # ==========================================
    train_dataset = SCM_Dataset(args.train_data_path, args.train_label_path)
    val_dataset = SCM_Dataset(args.val_data_path, args.val_label_path)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 初始化 ViT Base 模型
    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=181, drop_ratio=0, attn_drop_ratio=0).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    save_path = args.save_root
    os.makedirs(save_path, exist_ok=True)
    min_val_loss = float('inf')

    print("================ 开始训练 Base 模型 (Rho=0.0) ================")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_rmse = evaluate(model, val_loader, loss_function, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] Train MSE: {train_loss:.5f} | Val RMSE: {val_rmse:.3f}° | LR: {optimizer.param_groups[0]['lr']:.2e}")

        lr_schedule.step(val_rmse)
        if val_rmse <= min_val_loss:
            min_val_loss = val_rmse
            torch.save(model.state_dict(), os.path.join(save_path, f'weight_base_bestIQ_rho{args.rho}.pth'))
            print(f'>>> 模型已保存, 最小验证 RMSE: {min_val_loss:.3f}°')

        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rho', type=float, default=0.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    # 指向离线生成的纯净数据
    data_dir = os.path.join(root_parent, 'Graduation', 'data', 'IQ_Data', 'Single_Source')
    dataset_dir = os.path.join(data_dir, f'SCM_Single_Source_Rho{parser.get_default("rho")}')
    parser.add_argument('--train_data_path', type=str, default=os.path.join(dataset_dir, 'Train', 'vit_train_data.npy'))
    parser.add_argument('--train_label_path', type=str, default=os.path.join(dataset_dir, 'Train', 'train_labels.npy'))
    parser.add_argument('--val_data_path', type=str, default=os.path.join(dataset_dir, 'Val', 'vit_val_data.npy'))
    parser.add_argument('--val_label_path', type=str, default=os.path.join(dataset_dir, 'Val', 'val_labels.npy'))

    save_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base')
    parser.add_argument('--save_root', type=str, default=save_root)

    args = parser.parse_args()
    main(args)
