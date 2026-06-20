import numpy as np
import argparse
import os
import sys
import copy
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

root = Path(__file__).resolve().parents[3]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"
if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))
if 'utils' in sys.modules:
    del sys.modules['utils']
from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from data.data_create.theta_creater import same_data_Creater
from utils.early_stop import EarlyStopping


def transfer_learning(transfer_model, base_model, data_loader, data_creater, optimizer, loss_f, device, epoch, snap,
                      snr):
    transfer_model.train()
    base_model.eval()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # --- 权重配置 (可根据训练情况微调) ---
    w_task = 1.0  # 预测角度的主力损失权重
    w_cos = 0.5  # 余弦方向对齐权重
    w_gram = 0.001  # Gram矩阵分布对齐权重 (Gram值通常很大，需要极小权重)
    # -----------------------------------

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data
        input, labels = input.to(device), labels.to(device)

        batch_size = input.shape[0]
        max_vals = torch.max(torch.abs(input.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        input = input / (max_vals + 1e-8)

        pred_doa = transfer_model(input, logits=False)

        pred_flat = pred_doa.view(-1)
        labels_flat = labels.float().view(-1)

        task_loss = loss_f(pred_flat, labels_flat)

        pred_features = transfer_model(input, logits=True)

        with torch.no_grad():
            source_domain_data = data_creater(labels, snap=200, snr=20).to(device)
            fit_features = base_model(source_domain_data, logits=True)

        # 3. 计算余弦相似度损失 (cal_vec_similar)
        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=device) - torch.sum(pred_vec * fit_vec, dim=-1))

        # 4. 计算 Gram 矩阵损失 (cal_Gram_matrix)
        Gram_target = pred_features.transpose(-1, -2) @ pred_features
        Gram_target = 0.5 * (Gram_target + Gram_target.transpose(-1, -2))

        Gram_source = fit_features.transpose(-1, -2) @ fit_features
        Gram_source = 0.5 * (Gram_source + Gram_source.transpose(-1, -2))

        loss_gram = torch.mean((Gram_target - Gram_source) ** 2)
        loss = (w_task * task_loss) + (w_cos * loss_cos) + (w_gram * loss_gram)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=0.1)

        optimizer.step()
        optimizer.zero_grad()

        accu_loss += task_loss.detach()

        data_loader.desc = "[train epoch {}] Task: {:.2f} | Cos: {:.3f} | Gram: {:.3f}".format(
            epoch, task_loss.item(), loss_cos.item(), loss_gram.item() * w_gram
        )

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, epoch, k=1):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data
        input, labels = input.to(device), labels.to(device)

        batch_size = input.shape[0]
        max_vals = torch.max(torch.abs(input.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        input = input / (max_vals + 1e-8)

        pred = model(input, logits=False)

        pred_flat = pred.view(-1)
        labels_flat = labels.float().view(-1)

        loss = loss_function(pred_flat, labels_flat)
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] RMSE_loss: {:.3f}°".format(
            epoch, np.sqrt(accu_loss.item() / (step + 1)))
    return accu_loss.item() / (step + 1)



def main(args):
    train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], 2000)
    val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], 10000)

    save_path = args.save_root
    if not os.path.exists(save_path): os.makedirs(save_path)

    loss_function = torch.nn.MSELoss()
    embeding_dim = 768
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                                   out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)

    # 路径对齐：假设 weight_base_best.pth 在 args.root 目录下
    weight_path = os.path.join(args.root, 'weight_base_best_snr020.pth')
    state_dict = torch.load(weight_path, map_location=args.device)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.to(args.device)

    transfer_model = copy.deepcopy(base_model)

    for param in transfer_model.parameters():
        param.requires_grad = True

    dataset, val_dataset = ULA_dataset(args.M, -90, 90, 1, args.rho), ULA_dataset(args.M, -90, 90, 1, args.rho)
    base_dataset = ULA_dataset(args.M, -90, 90, 1, args.ori_rho)

    Create_datasets(dataset, args.k, train_theta_set, batch_size=100, snap=args.snap, snr=args.snr_range)
    Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=args.snap, snr=args.snr_range)

    data_creater = same_data_Creater(base_dataset, 'scm')

    train_dataloader = array_Dataloader(dataset, 64, load_style='torch', input_type='scm', output_type='doa')
    val_dataloader = array_Dataloader(val_dataset, 64, shuffle=False, load_style='torch', input_type='scm',
                                      output_type='doa')

    parm = [p for p in transfer_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parm, lr=5e-5)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(30, 0)

    min_val_loss = 100
    for epoch in range(args.epochs):
        # 1. 验证 (Epoch 0 时可看到基础性能)
        val_loss = evaluate(transfer_model, val_dataloader, loss_function, args.device, epoch, args.k)
        val_loss = np.sqrt(val_loss)

        # 2. 训练 (传入修复后的 snap 和 snr 参数)
        train_loss = transfer_learning(transfer_model, base_model, train_dataloader, data_creater, optimizer,
                                       loss_function, args.device, epoch + 1, args.snap, args.snr_range)

        lr_schedule.step(val_loss)
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight_transfer_best_snr20_rho1.0.pth'))
            print(f'>>> Epoch {epoch + 1} 模型已保存, RMSE: {min_val_loss:.4f}°')

        early_stopping(val_loss)
        if early_stopping.early_stop: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--snr_range', type=tuple, default=(0, 20))
    parser.add_argument('--snap', type=int, default=200)
    parser.add_argument('--signal_range', type=tuple, default=(-90, 90))
    parser.add_argument('--ori_rho', type=float, default=0)
    parser.add_argument('--rho', type=float, default=1.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
    base_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base')
    parser.add_argument('--root', type=str, default=base_root)
    transfer_save_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base_transfer')

    parser.add_argument('--save_root', type=str, default=transfer_save_root)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    main(args)