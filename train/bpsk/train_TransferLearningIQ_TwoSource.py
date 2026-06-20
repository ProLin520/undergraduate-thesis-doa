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


# =================================================================
# 1. 适配双信源的 Dataset (按 SNR 遍历加载 Multi-hot 标签)
# =================================================================
class SCM_TwoSource_Dataset(Dataset):
    def __init__(self, base_path, split_name):
        snrs = np.arange(-25, 26, 5)
        data_list, label_list = [], []

        print(f"正在加载 {split_name} 数据...")
        for snr in tqdm(snrs):
            data_path = os.path.join(base_path, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy")
            label_path = os.path.join(base_path, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")
            data_list.append(np.load(data_path))
            label_list.append(np.load(label_path))

        self.data = np.concatenate(data_list, axis=0)
        # 保持 Multi-hot 格式，类型必须为 float32 以便后续计算 BCE Loss 和 矩阵乘法
        self.labels = np.concatenate(label_list, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


def transfer_learning_two_source(transfer_model, base_model, data_loader, feature_bank, optimizer, loss_f, device,
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
        labels = labels.to(device)  # Shape: [Batch, 181], Multi-hot

        # Max-Abs 归一化
        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        # 1. 目标域预测与特征提取
        pred_features = transfer_model(input_data, logits=True)
        pred = transfer_model.head(pred_features)

        # -------------------------------------------------------------
        # 🌟 核心魔法：使用矩阵乘法从 feature_bank 构造双信源理想特征！
        # labels: [B, 181] (包含两个1), feature_bank: [181, 768]
        # 结果 fit_features: [B, 768]，恰好是两个独立理想特征的均值！
        # -------------------------------------------------------------
        fit_features = torch.matmul(labels, feature_bank) / 2.0

        # 3. 计算 Task 损失
        task_loss = loss_f(pred, labels)  # 此时 loss_f 是 BCEWithLogitsLoss

        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=device) - torch.sum(pred_vec * fit_vec, dim=-1))

        Gram_target = (pred_features.transpose(-1, -2) @ pred_features) / batch_size
        Gram_target = 0.5 * (Gram_target + Gram_target.transpose(-1, -2))

        Gram_source = (fit_features.transpose(-1, -2) @ fit_features) / batch_size
        Gram_source = 0.5 * (Gram_source + Gram_source.transpose(-1, -2))

        loss_gram = torch.mean((Gram_target - Gram_source) ** 2)
        # -------------------------------------------------------------
        # 👆 上面的 Cosine 损失和 Gram 矩阵损失代码【一字未改】
        # -------------------------------------------------------------

        # 5. 总损失计算
        loss = (w_task * task_loss) + (w_cos * loss_cos) + (w_gram * loss_gram)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=1.0)
        accu_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss / (step + 1)


@torch.no_grad()
def evaluate_two_source(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = input_data.to(device)
        labels = labels.to(device)

        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        outputs = model(input_data)

        # 获取 Top-2 索引，比对 Exact Match
        _, predicted_indices = torch.topk(outputs, 2, dim=1)
        predicted_multi_hot = torch.zeros_like(labels).scatter_(1, predicted_indices, 1)

        total += labels.size(0)
        correct += (predicted_multi_hot == labels).all(dim=1).sum().item()

    return correct / total


def main(args):
    device = torch.device(args.device)
    embeding_dim = 768

    # ================= 1: 加载 Base 模型 =================
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                                   out_dims=181, drop_ratio=0, attn_drop_ratio=0).to(device)
    base_weight_path = os.path.join(args.root, 'weight_base_TwoSource_rho0.0.pth')
    if not os.path.exists(base_weight_path):
        print(f"❌ 找不到 Base 模型权重: {base_weight_path}，请先训练双信源的基础 ViT！")
        return
    base_model.load_state_dict(torch.load(base_weight_path, map_location=device))
    base_model.eval()

    # ================= 2: 提取 181 维理想特征库 =================
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
        feature_bank = base_model(ideal_tensor, logits=True)
    print(">>> Feature Bank 提取完成！")

    # ================= 3: 加载目标域 (Rho=1.0) 双信源数据 =================
    train_dataset = SCM_TwoSource_Dataset(args.data_dir, "Train")
    val_dataset = SCM_TwoSource_Dataset(args.data_dir, "Val")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    transfer_model = copy.deepcopy(base_model)
    optimizer = optim.AdamW(transfer_model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 🌟 必须改为二元交叉熵
    loss_function = torch.nn.BCEWithLogitsLoss()

    # 因为指标改为了准确率 (Exact Match)，调度器和早停都要改为 'max' 模式
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    save_path = args.save_root
    os.makedirs(save_path, exist_ok=True)
    max_val_acc = 0.0

    print("\n================ 开始特征域自适应训练 (双信源) ================")
    for epoch in range(args.epochs):
        train_loss = transfer_learning_two_source(transfer_model, base_model, train_loader, feature_bank,
                                                  optimizer, loss_function, device, epoch + 1)
        val_acc = evaluate_two_source(transfer_model, val_loader, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] BCE+Cos+Gram Loss: {train_loss:.5f} | Val Exact Match: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        lr_schedule.step(val_acc)
        if val_acc >= max_val_acc:
            max_val_acc = val_acc
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight_transfer_TwoSource_rho{args.rho}.pth'))
            print(f'>>> 模型已保存, 当前最高准确率: {max_val_acc:.4f}')

        early_stopping(1.0 - val_acc)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)  # 虽然是双信源，但这里控制的是基础信号生成，保持1即可
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rho', type=float, default=1.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    base_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_2_base')
    parser.add_argument('--root', type=str, default=base_root)
    parser.add_argument('--save_root', type=str,
                        default=os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_2_transfer'))

    data_dir = os.path.join(root_parent, 'Graduation', 'data', 'IQ_Data', 'Two_Source')
    dataset_dir = os.path.join(data_dir, f'SCM_Two_Source_Rho{parser.get_default("rho")}')
    parser.add_argument('--data_dir', type=str, default=dataset_dir)

    args = parser.parse_args()
    main(args)
