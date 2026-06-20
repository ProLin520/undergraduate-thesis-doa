import numpy as np
import argparse
import os
import sys
import copy
import json
import random
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
from data.data_create.theta_creater import same_data_Creater
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding, calc_rmse


VAL_SNR_LIST = [-20, -15, -10, -5, 0, 5, 10]

def save_args(argparser, file):
    with open(file, "w") as f:
        json.dump(vars(argparser), f, indent=4)


def get_stage_cfg(epoch):
    if epoch < 30:
        return {"snr": (0, 10), "min_delta_theta": 10}
    elif epoch < 70:
        return {"snr": (-10, 10), "min_delta_theta": 7}
    else:
        return {"snr": (-20, 10), "min_delta_theta": 5}


def build_snr_val_loaders(rho, snap, signal_range, step_used, k, batch_size=64, theta_num=2000, min_delta_theta=5):
    theta_val = Create_random_k_input_theta(k, signal_range[0], signal_range[1], theta_num, min_delta_theta=min_delta_theta)
    theta_val = np.array(theta_val)
    valid_mask_val = (~np.isnan(theta_val).any(axis=1)) & (np.max(theta_val, axis=1) <= signal_range[1]) & (np.min(theta_val, axis=1) >= signal_range[0])
    theta_val = theta_val[valid_mask_val]

    val_loaders = {}
    for snr in VAL_SNR_LIST:
        val_dataset = ULA_dataset(8, signal_range[0], signal_range[1], step_used, rho)
        Create_datasets(val_dataset, k, theta_val.copy(), batch_size=128, snap=snap, snr=snr, shared_snr=True)
        val_loaders[snr] = array_Dataloader(val_dataset, batch_size, shuffle=False, load_style='torch', input_type='scm', output_type='doa')
    return val_loaders


def transfer_learning_one_epoch(transfer_model, base_model, data_loader, data_creater, optimizer, loss_f, device, epoch, snap, source_snr=20):
    transfer_model.train()
    base_model.eval()
    accu_loss = torch.zeros(1, device=device)
    optimizer.zero_grad()

    w_task = 1.0
    w_cos = 0.5
    w_gram = 0.001

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()

        batch_size = inputs.shape[0]
        max_vals = torch.max(torch.abs(inputs.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        inputs = inputs / (max_vals + 1e-8)

        pred_doa = transfer_model(inputs, logits=False)
        pred_sorted, _ = torch.sort(pred_doa, dim=1)
        labels_sorted, _ = torch.sort(labels, dim=1)
        task_loss = loss_f(pred_sorted, labels_sorted)

        pred_features = transfer_model(inputs, logits=True)

        with torch.no_grad():
            source_domain_data = data_creater(labels, snap=snap, snr=source_snr).to(device)
            fit_features = base_model(source_domain_data, logits=True)

        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=device) - torch.sum(pred_vec * fit_vec, dim=-1))

        gram_target = pred_features.transpose(-1, -2) @ pred_features
        gram_target = 0.5 * (gram_target + gram_target.transpose(-1, -2))
        gram_source = fit_features.transpose(-1, -2) @ fit_features
        gram_source = 0.5 * (gram_source + gram_source.transpose(-1, -2))

        loss_gram = torch.mean((gram_target - gram_source) ** 2)
        loss = w_task * task_loss + w_cos * loss_cos + w_gram * loss_gram

        loss.backward()
        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=0.1)

        optimizer.step()
        optimizer.zero_grad()

        accu_loss += task_loss.detach()
        data_loader.desc = "[train epoch {}] Task: {:.3f} | Cos: {:.3f} | Gram: {:.3f}".format(epoch, task_loss.item(), loss_cos.item(), loss_gram.item() * w_gram)

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_transfer_snr_curve(model, val_loaders, device):
    model.eval()
    snr_rmse = {}

    for snr, loader in val_loaders.items():
        val_loss = 0.0
        val_steps = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            batch_size = inputs.shape[0]
            max_vals = torch.max(torch.abs(inputs.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
            inputs = inputs / (max_vals + 1e-8)

            pred = model(inputs, logits=False)
            val_loss += calc_rmse(pred, labels)
            val_steps += 1

        snr_rmse[snr] = np.sqrt(val_loss / val_steps)

    avg_rmse = float(np.mean(list(snr_rmse.values())))
    return avg_rmse, snr_rmse


def main(args):
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_args(args, os.path.join(save_path, "transfer_laboratory_set.json"))

    loss_function = torch.nn.MSELoss()
    embeding_dim = 768

    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim, out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    weight_path = os.path.join(args.root, 'weight_base_ThreeSource.pth')
    state_dict = torch.load(weight_path, map_location=args.device)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.to(args.device)
    base_model.eval()

    transfer_model = copy.deepcopy(base_model)
    for param in transfer_model.parameters():
        param.requires_grad = True
    transfer_model.to(args.device)

    train_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
    base_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.ori_rho)
    data_creater = same_data_Creater(base_dataset, 'scm')

    val_loaders = build_snr_val_loaders(args.rho, args.snap, args.signal_range, args.step_used, args.k, batch_size=64, theta_num=2000, min_delta_theta=5)

    optimizer = optim.Adam([p for p in transfer_model.parameters() if p.requires_grad], lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_val_rmse = float('inf')

    for epoch in range(args.epochs):
        train_dataset.clear()
        cfg = get_stage_cfg(epoch)

        train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], 10000, min_delta_theta=cfg["min_delta_theta"])
        train_theta_set = np.array(train_theta_set)
        valid_mask = (~np.isnan(train_theta_set).any(axis=1)) & (np.max(train_theta_set, axis=1) <= args.signal_range[1]) & (np.min(train_theta_set, axis=1) >= args.signal_range[0])
        train_theta_set = train_theta_set[valid_mask]

        Create_datasets(train_dataset, args.k, train_theta_set, batch_size=128, snap=args.snap, snr=cfg["snr"], shared_snr=True)

        train_dataloader = array_Dataloader(train_dataset, 64, shuffle=True, load_style='torch', input_type='scm', output_type='doa')

        train_loss = transfer_learning_one_epoch(transfer_model, base_model, train_dataloader, data_creater, optimizer, loss_function, args.device, epoch + 1, args.snap, source_snr=20)
        avg_val_rmse, snr_rmse = evaluate_transfer_snr_curve(transfer_model, val_loaders, args.device)

        scheduler.step(avg_val_rmse)

        print(f"[Epoch {epoch + 1}/{args.epochs}] TrainLoss: {train_loss:.4f} | ValAvg: {avg_val_rmse:.4f}° | V@10: {snr_rmse[10]:.4f}° | V@0: {snr_rmse[0]:.4f}° | V@-10: {snr_rmse[-10]:.4f}° | V@-20: {snr_rmse[-20]:.4f}° | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(transfer_model.state_dict(), os.path.join(save_path, 'weight_transfer_ThreeSource_rho0.0.pth'))
            print(f"⭐ 已保存最优 Transfer 模型 | Avg: {best_val_rmse:.4f}° | [-20,-10,0,10] = [{snr_rmse[-20]:.3f}, {snr_rmse[-10]:.3f}, {snr_rmse[0]:.3f}, {snr_rmse[10]:.3f}]")

    print(f"\n✅ ViT Transfer 训练结束，最优模型保存在: {os.path.join(save_path, 'weight_transfer_ThreeSource_rho0.0.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snap', type=int, default=50)
    parser.add_argument('--signal_range', type=tuple, default=(-90, 90))
    parser.add_argument('--step_used', type=float, default=1)
    parser.add_argument('--ori_rho', type=float, default=0.0)
    parser.add_argument('--rho', type=float, default=0.0)
    parser.add_argument('--root', type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base")
    parser.add_argument('--save_root', type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base_transfer")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    main(args)