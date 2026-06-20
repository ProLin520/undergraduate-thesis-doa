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

root = Path(__file__).resolve().parents[4]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))
if 'utils' in sys.modules:
    del sys.modules['utils']

from data.data_create.signal_datasets90 import ULA_dataset
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from data.data_create.theta_creater import same_data_Creater
from utils.early_stop import EarlyStopping


class SCM_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        onehot_labels = np.load(label_path)

        # 🌟 核心修改 1：不再减去 90，也不再除以 90.0
        # 直接拿 argmax 的结果作为 0~180 的类别索引
        self.labels = np.argmax(onehot_labels, axis=1).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)

        # 🌟 核心修改 2：分类标签必须是整数 (torch.long)，且不要加中括号 []
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


def transfer_learning(transfer_model, base_model, data_loader, feature_bank, optimizer, loss_f, device, epoch):
    transfer_model.train()
    base_model.eval()
    accu_loss = 0.0  # 改为浮点数统计
    optimizer.zero_grad()

    w_task = 1.0
    w_cos = 0.1
    w_gram = 0.001

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = input_data.to(device)
        # 🌟 核心修改 1：确保 labels 是 [0, 180] 的整型索引
        labels = labels.to(device).long()

        # 每次输入前进行 Max-Abs 归一化
        batch_size = input_data.shape[0]
        max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
        input_data = input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)

        # 1. 目标域预测与特征提取
        pred_features = transfer_model(input_data, logits=True)
        # 🌟 核心修改 2：去除 squeeze(-1)，因为现在输出是 [Batch, 181] 的分类概率
        pred = transfer_model.head(pred_features)

        # 2. 从 feature_bank 抓取导师特征
        # 🌟 核心修改 3：因为 labels 已经是 0~180 的整数索引，直接用！不用再乘 90 + 90 了
        angles_idx = torch.clamp(labels, min=0, max=180)
        fit_features = feature_bank[angles_idx].to(device)

        # 3. 计算 Task 损失与 Cosine 损失
        labels = labels.view(-1)
        task_loss = loss_f(pred, labels)  # 这里 loss_f 是 CrossEntropyLoss

        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=device) - torch.sum(pred_vec * fit_vec, dim=-1))

        # 4. 计算严格对称的 Gram 矩阵损失
        Gram_target = (pred_features.transpose(-1, -2) @ pred_features) / batch_size
        Gram_target = 0.5 * (Gram_target + Gram_target.transpose(-1, -2))

        Gram_source = (fit_features.transpose(-1, -2) @ fit_features) / batch_size
        Gram_source = 0.5 * (Gram_source + Gram_source.transpose(-1, -2))

        loss_gram = torch.mean((Gram_target - Gram_source) ** 2)

        # 5. 总损失计算
        loss = (w_task * task_loss) + (w_cos * loss_cos) + (w_gram * loss_gram)
        loss.backward()

        # 梯度裁剪防炸
        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=1.0)

        accu_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device):
    model.eval()
    accu_loss = 0.0
    total_samples = 0  # 🌟 用于精确计算 RMSE

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = input_data.to(device)
        labels = labels.to(device)

        # 🌟 核心修改 4：输出 181 维概率，用 argmax 选出预测角度
        pred = model(input_data)
        pred_class = torch.argmax(pred, dim=1)

        # 计算误差平方和 (索引即角度，直接相减)
        batch_sq_err = torch.sum((pred_class.float() - labels.float()) ** 2).item()
        accu_loss += batch_sq_err
        total_samples += labels.size(0)

    # 🌟 核心修改 5：直接算出真实的 RMSE，不用再乘 90.0 了
    rmse_degrees = np.sqrt(accu_loss / total_samples)
    return rmse_degrees


def main(args):
    device = torch.device(args.device)
    embeding_dim = 768

    # ================= 3: 加载 Base 模型 (巨人的肩膀) =================
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=181, drop_ratio=0, attn_drop_ratio=0).to(device)
    base_weight_path = os.path.join(args.root, 'weight_base_bestIQ_rho0.0.pth')
    if not os.path.exists(base_weight_path):
        print(f"❌ 找不到 Base 模型权重: {base_weight_path}，请先运行 train_TransIQ.py！")
        return
    base_model.load_state_dict(torch.load(base_weight_path, map_location=device))
    base_model.eval()

    # ================= 4: 生成理想流型特征库 (Feature Bank) =================
    print(">>> 正在生成理想流型特征库 (Rho=0.0) 作为对齐锚点...")
    base_dataset = ULA_dataset(args.M, -90, 90, 1, rho=0.0)
    # 为 181 个角度生成理想信号
    base_dataset.Create_DOA_data(args.k, np.arange(-90, 91)[:, None], np.full((181, 1), 20),
                                 s_t_type='gauss_input', snap=1024, snr_set=1)

    # 提取理想特征
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

    # ================= 5: 加载带误差的目标域数据 (Rho=1.0) =================
    print(f">>> 加载目标域训练数据: {args.train_data_path}")
    train_dataset = SCM_Dataset(args.train_data_path, args.train_label_path)
    val_dataset = SCM_Dataset(args.val_data_path, args.val_label_path)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 初始化 Transfer 模型 (直接克隆 Base 权重)
    transfer_model = copy.deepcopy(base_model)
    optimizer = optim.AdamW(transfer_model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_function = torch.nn.CrossEntropyLoss()
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    save_path = args.save_root
    os.makedirs(save_path, exist_ok=True)
    min_val_loss = float('inf')

    print("\n================ 开始特征域自适应训练 (BPSK 赛道) ================")
    for epoch in range(args.epochs):
        train_loss = transfer_learning(transfer_model, base_model, train_loader, feature_bank,
                                       optimizer, loss_function, device, epoch + 1)
        val_rmse = evaluate(transfer_model, val_loader, loss_function, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] Loss: {train_loss:.5f} | Val RMSE: {val_rmse:.3f}° | LR: {optimizer.param_groups[0]['lr']:.2e}")

        lr_schedule.step(val_rmse)
        if val_rmse <= min_val_loss:
            min_val_loss = val_rmse
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight_transfer_bestIQ_rho{args.rho}.pth'))
            print(f'>>> 模型已保存, RMSE: {min_val_loss:.3f}°')

        early_stopping(val_rmse)
        if early_stopping.early_stop: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    base_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base')
    parser.add_argument('--root', type=str, default=base_root)

    transfer_save_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base_transfer')
    parser.add_argument('--save_root', type=str, default=transfer_save_root)

    # ================= 目标域数据指向 Rho1.0 =================
    rho_default = 1.0
    data_dir = os.path.join(root_parent, 'Graduation', 'data', 'IQ_Data', 'Single_Source')
    dataset_dir = os.path.join(data_dir, f'SCM_Single_Source_Rho{rho_default}')
    parser.add_argument('--train_data_path', type=str, default=os.path.join(dataset_dir, 'Train', 'vit_train_data.npy'))
    parser.add_argument('--train_label_path', type=str, default=os.path.join(dataset_dir, 'Train', 'train_labels.npy'))
    parser.add_argument('--val_data_path', type=str, default=os.path.join(dataset_dir, 'Val', 'vit_val_data.npy'))
    parser.add_argument('--val_label_path', type=str, default=os.path.join(dataset_dir, 'Val', 'val_labels.npy'))

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--rho', type=float, default=rho_default)

    args = parser.parse_args()
    main(args)
