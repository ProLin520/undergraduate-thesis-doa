import numpy as np
import argparse
import os
import sys
import copy
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

root = Path(__file__).resolve().parents[3]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))
if 'utils' in sys.modules:
    del sys.modules['utils']

from data.data_create.signal_datasets90 import ULA_dataset
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from utils.early_stop import EarlyStopping


class SCM_SevenSource_Dataset(Dataset):
    def __init__(self, base_path, split_name):
        snrs = [0, 5, 10, 15, 20]
        data_list, label_list = [], []

        print(f"正在加载 {split_name} 数据...")
        for snr in tqdm(snrs):
            data_path = os.path.join(base_path, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy")
            label_path = os.path.join(base_path, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")
            data_list.append(np.load(data_path))
            label_list.append(np.load(label_path))

        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


def transfer_learning_seven_source(transfer_model, base_model, data_loader, feature_bank, optimizer, loss_f, device,
                                   epoch):
    transfer_model.train()
    base_model.eval()
    accu_loss = 0.0
    optimizer.zero_grad()

    w_task = 1.0
    w_cos = 0.1
    w_gram = 0.001

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = input_data.to(device)
        labels = labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        pred_features = transfer_model(input_data, logits=True)
        pred = transfer_model.head(pred_features)

        # -------------------------------------------------------------
        # 🌟 核心物理对齐：提取 7 个单信源锚点并平均！
        # labels 中包含 7 个 1。矩阵乘法后，除以 7.0 完成均值叠加。
        # -------------------------------------------------------------
        fit_features = torch.matmul(labels, feature_bank) / 7.0

        task_loss = loss_f(pred, labels)

        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=device) - torch.sum(pred_vec * fit_vec, dim=-1))

        Gram_target = (pred_features.transpose(-1, -2) @ pred_features) / batch_size
        Gram_target = 0.5 * (Gram_target + Gram_target.transpose(-1, -2))

        Gram_source = (fit_features.transpose(-1, -2) @ fit_features) / batch_size
        Gram_source = 0.5 * (Gram_source + Gram_source.transpose(-1, -2))

        loss_gram = torch.mean((Gram_target - Gram_source) ** 2)

        loss = (w_task * task_loss) + (w_cos * loss_cos) + (w_gram * loss_gram)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=1.0)
        accu_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss / (step + 1)


@torch.no_grad()
def evaluate_seven_source(model, data_loader, device):
    model.eval()
    total_hits = 0
    total_targets = 0

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data, labels = input_data.to(device), labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        outputs = model(input_data)

        # 取前 7 个最大值
        _, predicted_indices = torch.topk(outputs, 7, dim=1)
        predicted_multi_hot = torch.zeros_like(labels).scatter_(1, predicted_indices, 1)

        # 计算 Hit Rate
        total_hits += (predicted_multi_hot * labels).sum().item()
        total_targets += labels.sum().item()

    return total_hits / total_targets


def main(args):
    device = torch.device(args.device)
    embeding_dim = 768

    # 1. 加载七信源的 Base 模型 (作为起点)
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                                   out_dims=181, drop_ratio=0, attn_drop_ratio=0).to(device)
    base_weight_path = os.path.join(args.root, 'weight_base_SevenSource_Article_rho0.0.pth')
    if not os.path.exists(base_weight_path):
        print(f"❌ 找不到 Base 模型权重: {base_weight_path}，请先运行 train_TransIQ_SevenSource.py！")
        return
    base_model.load_state_dict(torch.load(base_weight_path, map_location=device))
    base_model.eval()

    # 2. 提取 181 维理想单信源特征库 (k=1 保持不变，这是字典！)
    print(">>> 正在生成理想流型特征库 (Rho=0.0)...")
    base_dataset = ULA_dataset(args.M, -90, 90, 1, rho=0.0)
    base_dataset.Create_DOA_data(args.k, np.arange(-90, 91)[:, None], np.full((181, 1), 20),
                                 s_t_type='gauss_input', snap=1024, snr_set=1)

    ideal_vit_data = np.zeros((181, 2, args.M, args.M), dtype=np.float32)
    for i in range(181):
        scm = base_dataset.ori_scm[i]
        ideal_vit_data[i, 0, :, :] = np.real(scm)
        ideal_vit_data[i, 1, :, :] = np.imag(scm)
        if np.max(np.abs(ideal_vit_data[i])) > 1e-8:
            ideal_vit_data[i] /= np.max(np.abs(ideal_vit_data[i]))

    ideal_tensor = torch.tensor(ideal_vit_data).to(device)
    with torch.no_grad():
        # 注意：这里我们用 base_model 提取基础字典
        feature_bank = base_model(ideal_tensor, logits=True)
    print(">>> 纯净特征库 (Feature Bank) 提取完成！")

    # 3. 加载七信源目标域数据
    train_dataset = SCM_SevenSource_Dataset(args.data_dir, "Train")
    val_dataset = SCM_SevenSource_Dataset(args.data_dir, "Val")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    transfer_model = copy.deepcopy(base_model)
    optimizer = optim.AdamW(transfer_model.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_function = torch.nn.BCEWithLogitsLoss()
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    save_path = args.save_root
    os.makedirs(save_path, exist_ok=True)
    max_val_hit_rate = 0.0

    print("\n================ 开始七信源物理特征解耦训练 ================")
    for epoch in range(args.epochs):
        train_loss = transfer_learning_seven_source(transfer_model, base_model, train_loader, feature_bank,
                                                    optimizer, loss_function, device, epoch + 1)
        val_hit_rate = evaluate_seven_source(transfer_model, val_loader, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] BCE+Cos+Gram Loss: {train_loss:.5f} | Val Hit Rate: {val_hit_rate:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        lr_schedule.step(val_hit_rate)
        if val_hit_rate >= max_val_hit_rate:
            max_val_hit_rate = val_hit_rate
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight_transfer_SevenSource_Article_rho{args.rho}.pth'))
            print(f'>>> 模型已保存, 当前最高命中率: {max_val_hit_rate:.4f}')

        early_stopping(1.0 - val_hit_rate)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)  # 基础字典必须保持 k=1
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rho', type=float, default=0.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    # Base 模型来源
    base_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_7_base')
    parser.add_argument('--root', type=str, default=base_root)
    # 迁移后模型去向
    parser.add_argument('--save_root', type=str,
                        default=os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_7_base_transfer'))

    # 数据集源头 (可以是 Rho=0，利用迁移学习解耦；如果是 Rho=1 就把路径改成带 Rho1.0)
    data_dir = os.path.join(root_parent, 'Graduation', 'data', 'IQ_Data', 'Seven_Source')
    dataset_dir = os.path.join(data_dir, f'SCM_Seven_Source_Article_Rho{parser.get_default("rho")}')
    parser.add_argument('--data_dir', type=str, default=dataset_dir)

    args = parser.parse_args()
    main(args)
