import numpy as np
import argparse
import os
import json

import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot

from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from utils.early_stop import EarlyStopping


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch, grid_to_theta=True, k=3):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        input = input.to(device)
        labels = labels.to(device)

        batch_size = input.shape[0]
        max_vals = torch.max(torch.abs(input.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        input = input / (max_vals + 1e-8)

        pred = model(input)

        if grid_to_theta:
            pred = F.sigmoid(pred)
            _, pred = model.sp_to_doa(pred, k)

        pred_flat = pred.view(-1)
        labels_flat = labels.to(device).float().view(-1)

        loss = loss_function(pred_flat, labels_flat)
        loss.backward()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1)))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, epoch, grid_to_theta=True, k=3):
    model.eval()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        input = input.to(device)
        labels = labels.to(device)

        batch_size = input.shape[0]
        max_vals = torch.max(torch.abs(input.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
        input = input / (max_vals + 1e-8)

        pred = model(input)

        if grid_to_theta:
            pred = F.sigmoid(pred)
            _, pred = model.sp_to_doa(pred, k)

        pred_flat = pred.view(-1)
        labels_flat = labels.to(device).float().view(-1)

        loss = loss_function(pred_flat, labels_flat)
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1)))

    return accu_loss.item() / (step + 1)


def main(args):
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 动态生成训练角度
    train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                  args.signal_range[1], 10000, min_delta_theta=2)
    val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                args.signal_range[1], 2000, min_delta_theta=2)

    save_args(args, os.path.join(save_path, 'laboratory_set.json'))
    tb_writer = SummaryWriter(logdir=os.path.join(save_path, 'run'))
    loss_function = torch.nn.MSELoss()

    print(f"开始训练 Base 模型，混合信噪比范围: {args.snr_range} dB")
    id1 = '_mixed_snr_base'
    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    model.to(args.device)

    dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
    val_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)

    Create_datasets(dataset, args.k, train_theta_set, batch_size=100, snap=args.snap, snr=args.snr_range)
    Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=args.snap, snr=args.snr_range)

    train_dataloader = array_Dataloader(dataset, 256, load_style='torch', input_type='scm', output_type='doa')
    val_dataloader = array_Dataloader(val_dataset, 256, shuffle=False, load_style='torch', input_type='scm',
                                      output_type='doa')

    parm = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
    early_stopping = EarlyStopping(30, 0)

    min_val_loss = 100
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer, args.device, epoch + 1, False,
                                     args.k)
        val_loss = evaluate(model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)
        val_loss = np.sqrt(val_loss)

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'weight_base_best_snr020.pth'))
            print(f'epoch {epoch + 1} 模型已保存, 最小验证损失(RMSE): {min_val_loss:.4f}°')


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--snr_range', type=tuple, default=(0, 20))
    parser.add_argument('--snap', type=int, default=200)
    parser.add_argument('--signal_range', type=tuple, default=(-90, 90))
    parser.add_argument('--step_used', type=float, default=1)

    # 这里保持默认值，你在运行时可以通过命令行或者改这里测试
    parser.add_argument('--rho', type=float, default=0.0)

    current_script_path = os.path.abspath(__file__)
    root_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
    save_root = os.path.join(root_parent, 'Graduation', 'result', 'vit', 'vit_M_8_k_1_base')

    parser.add_argument('--save_root', type=str, default=save_root)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--grid_to_theta', type=bool, default=False)
    args = parser.parse_args()
    main(args)