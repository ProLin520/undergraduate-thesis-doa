import numpy as np
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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


class SCM_TwoSource_Dataset(Dataset):
    def __init__(self, base_path, split_name):
        snrs = np.arange(-25, 26, 5)
        data_list, label_list = [], []

        print(f"正在加载 {split_name} 数据 (按 SNR 合并)...")
        for snr in tqdm(snrs):
            data_path = os.path.join(base_path, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy")
            label_path = os.path.join(base_path, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")

            data_list.append(np.load(data_path))
            label_list.append(np.load(label_path))

        self.data = np.concatenate(data_list, axis=0)

        # 🌟 双信源任务核心：绝对不能用 argmax！直接保留 181 维的 Multi-hot 浮点数组
        self.labels = np.concatenate(label_list, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 🌟 配合 BCEWithLogitsLoss，输入数据和标签都必须是 float32 类型
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    accu_loss = 0.0
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(data_loader):
        input_data, labels = input_data.to(device), labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        pred = model(input_data)
        loss = loss_function(pred, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        accu_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data, labels = input_data.to(device), labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        outputs = model(input_data)

        # 🌟 双信源评估：获取预测值最高的两个索引
        _, predicted_indices = torch.topk(outputs, 2, dim=1)
        predicted_multi_hot = torch.zeros_like(labels).scatter_(1, predicted_indices, 1)

        total += labels.size(0)
        correct += (predicted_multi_hot == labels).all(dim=1).sum().item()

    # 返回 Exact Match 准确率
    return correct / total


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_dataset = SCM_TwoSource_Dataset(args.data_dir, "Train")
    val_dataset = SCM_TwoSource_Dataset(args.data_dir, "Val")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=181, drop_ratio=0, attn_drop_ratio=0).to(device)

    # 🌟 双信源必须使用 BCEWithLogitsLoss
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 因为准确率是越高越好，所以把 mode 改为 'max'
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    os.makedirs(args.save_root, exist_ok=True)
    max_val_acc = 0.0

    print("================ 开始训练 ViT 双信源模型 ================")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] Train BCE Loss: {train_loss:.5f} | Val Exact Match Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        lr_schedule.step(val_acc)
        if val_acc >= max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_root, f'weight_base_TwoSource_rho{args.rho}.pth'))
            print(f'>>> 模型已保存, 当前最高验证准确率: {max_val_acc:.4f}')

            # 🌟 核心修复：传入 (1 - val_acc) 错误率，让原本追求"越小越好"的早停机制正常工作！
        early_stopping(1.0 - val_acc)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    # ViT 一般需要比 CNN 小一点的学习率
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rho', type=float, default=0.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    # 数据集路径
    data_dir = os.path.join(root_parent, 'Graduation', 'data', 'IQ_Data', 'Two_Source')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(data_dir, f'SCM_Two_Source_Rho{parser.get_default("rho")}'))
    # 保存路径
    parser.add_argument('--save_root', type=str,
                        default=os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_2_base'))

    args = parser.parse_args()
    main(args)
