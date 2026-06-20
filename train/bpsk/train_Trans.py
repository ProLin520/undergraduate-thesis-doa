import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


graduation_root = Path(__file__).resolve().parents[2]
external_root = graduation_root / "external" / "DOA_est_Master-master"
for path in (graduation_root, external_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dl_models.embeding_layer import scm_embeding
from dl_models.vit_model import VisionTransformer
from utils.early_stop import EarlyStopping


class BPSKSCMRegressionDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        labels = np.load(label_path)
        if labels.ndim > 1 and labels.shape[-1] > 1:
            labels = np.argmax(labels, axis=1)
        labels = np.asarray(labels).reshape(-1).astype(np.float32)
        if labels.min() >= 0 and labels.max() <= 180:
            labels = labels - 90.0
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def normalize_scm(input_data):
    batch_size = input_data.shape[0]
    max_vals = torch.max(torch.abs(input_data.view(batch_size, -1)), dim=1)[0]
    return input_data / (max_vals.view(batch_size, 1, 1, 1) + 1e-8)


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch):
    model.train()
    accu_loss = torch.zeros(1, device=device)
    optimizer.zero_grad()

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = normalize_scm(input_data.to(device))
        labels = labels.to(device).float()

        pred = model(input_data).view(-1)
        loss = loss_function(pred, labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        accu_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()

        loop.desc = "[train epoch {}] RMSE_loss: {:.3f}".format(
            epoch, np.sqrt(accu_loss.item() / (step + 1))
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training", loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, epoch):
    model.eval()
    accu_loss = torch.zeros(1, device=device)

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = normalize_scm(input_data.to(device))
        labels = labels.to(device).float()

        pred = model(input_data).view(-1)
        loss = loss_function(pred, labels.view(-1))
        accu_loss += loss.detach()

        loop.desc = "[valid epoch {}] RMSE_loss: {:.3f}".format(
            epoch, np.sqrt(accu_loss.item() / (step + 1))
        )

    return accu_loss.item() / (step + 1)


def save_args(args, file):
    with open(file, "w") as f:
        json.dump(vars(args), f, indent=4)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = BPSKSCMRegressionDataset(args.train_data_path, args.train_label_path)
    val_dataset = BPSKSCMRegressionDataset(args.val_data_path, args.val_label_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    embeding_dim = 768
    model = VisionTransformer(
        embed_layer=scm_embeding(args.M, embeding_dim),
        embed_dim=embeding_dim,
        out_dims=args.k,
        drop_ratio=0,
        attn_drop_ratio=0,
    ).to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    os.makedirs(args.save_root, exist_ok=True)
    save_args(args, os.path.join(args.save_root, "laboratory_set.json"))

    min_val_rmse = float("inf")
    print("================ Start BPSK Trans regression base training ================")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device, epoch + 1)
        val_loss = evaluate(model, val_loader, loss_function, device, epoch + 1)
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train RMSE: {train_rmse:.3f} | Val RMSE: {val_rmse:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        lr_schedule.step(val_rmse)
        if val_rmse <= min_val_rmse:
            min_val_rmse = val_rmse
            weight_path = os.path.join(args.save_root, f"weight_base_bestIQ_reg_rho{args.rho}.pth")
            torch.save(model.state_dict(), weight_path)
            print(f">>> Model saved. Best Val RMSE: {min_val_rmse:.3f}")

        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rho", type=float, default=0.0)

    data_dir = graduation_root / "data" / "IQ_Data" / "Single_Source"
    dataset_dir = data_dir / f"SCM_Single_Source_Rho{parser.get_default('rho')}"
    parser.add_argument("--train_data_path", type=str, default=str(dataset_dir / "Train" / "vit_train_data.npy"))
    parser.add_argument("--train_label_path", type=str, default=str(dataset_dir / "Train" / "train_labels.npy"))
    parser.add_argument("--val_data_path", type=str, default=str(dataset_dir / "Val" / "vit_val_data.npy"))
    parser.add_argument("--val_label_path", type=str, default=str(dataset_dir / "Val" / "val_labels.npy"))

    save_root = graduation_root / "result" / "vit" / "vit_M_8_k_1_base"
    parser.add_argument("--save_root", type=str, default=str(save_root))

    args = parser.parse_args()
    main(args)
