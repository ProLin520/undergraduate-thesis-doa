import argparse
import copy
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

from data.data_create.Generate_IQ_Data import generate_iq_sample
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


def bpsk_scm_vit_sample(angle, snr, rho, snap=1024, M=8):
    iq_mat = generate_iq_sample([angle], snap, snr, M=M, rho=rho)
    x_complex = iq_mat[:M, :] + 1j * iq_mat[M:, :]
    scm = (x_complex @ x_complex.conj().T) / snap

    vit_data = np.zeros((2, M, M), dtype=np.float32)
    vit_data[0, :, :] = np.real(scm)
    vit_data[1, :, :] = np.imag(scm)
    max_val = np.max(np.abs(vit_data))
    if max_val > 1e-8:
        vit_data /= max_val
    return vit_data


@torch.no_grad()
def build_feature_bank(base_model, device, args):
    base_model.eval()
    angles = np.arange(-90, 91)
    feature_rows = []

    print(">>> Building BPSK source-domain feature bank...")
    for angle in tqdm(angles, file=sys.stdout, leave=False):
        samples = [
            bpsk_scm_vit_sample(angle, args.feature_snr, args.ori_rho, args.feature_snap, args.M)
            for _ in range(args.feature_samples)
        ]
        sample_tensor = torch.tensor(np.asarray(samples), dtype=torch.float32, device=device)
        sample_tensor = normalize_scm(sample_tensor)
        features = base_model(sample_tensor, logits=True)
        feature_rows.append(features.mean(dim=0))

    return torch.stack(feature_rows, dim=0)


def transfer_learning(transfer_model, data_loader, feature_bank, optimizer, loss_f, device, epoch, args):
    transfer_model.train()
    accu_task_loss = torch.zeros(1, device=device)
    optimizer.zero_grad()

    loop = tqdm(data_loader, file=sys.stdout, leave=False)
    for step, (input_data, labels) in enumerate(loop):
        input_data = normalize_scm(input_data.to(device))
        labels = labels.to(device).float()

        pred_features = transfer_model(input_data, logits=True)
        pred = transfer_model.head(pred_features).view(-1)

        task_loss = loss_f(pred, labels.view(-1))

        angle_idx = torch.clamp(torch.round(labels + 90).long(), min=0, max=180)
        fit_features = feature_bank[angle_idx].to(device)

        pred_vec = torch.nn.functional.normalize(pred_features, dim=-1)
        fit_vec = torch.nn.functional.normalize(fit_features, dim=-1)
        loss_cos = torch.mean(1.0 - torch.sum(pred_vec * fit_vec, dim=-1))

        batch_size = input_data.shape[0]
        gram_target = (pred_features.transpose(-1, -2) @ pred_features) / batch_size
        gram_target = 0.5 * (gram_target + gram_target.transpose(-1, -2))
        gram_source = (fit_features.transpose(-1, -2) @ fit_features) / batch_size
        gram_source = 0.5 * (gram_source + gram_source.transpose(-1, -2))
        loss_gram = torch.mean((gram_target - gram_source) ** 2)

        loss = (args.w_task * task_loss) + (args.w_cos * loss_cos) + (args.w_gram * loss_gram)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
        accu_task_loss += task_loss.detach()

        loop.desc = "[train epoch {}] Task RMSE: {:.3f} | Cos: {:.3f} | Gram: {:.3f}".format(
            epoch,
            np.sqrt(accu_task_loss.item() / (step + 1)),
            loss_cos.item(),
            (args.w_gram * loss_gram).item(),
        )

    return accu_task_loss.item() / (step + 1)


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


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    embeding_dim = 768

    base_model = VisionTransformer(
        embed_layer=scm_embeding(args.M, embeding_dim),
        embed_dim=embeding_dim,
        out_dims=args.k,
        drop_ratio=0,
        attn_drop_ratio=0,
    ).to(device)

    base_weight_path = os.path.join(args.root, f"weight_base_bestIQ_reg_rho{args.ori_rho}.pth")
    if not os.path.exists(base_weight_path):
        print(f"Base regression weight not found: {base_weight_path}")
        print("Please run bpsk/train_Trans.py first, or pass --root to the trained base directory.")
        return

    base_model.load_state_dict(torch.load(base_weight_path, map_location=device), strict=True)
    base_model.eval()

    feature_bank = build_feature_bank(base_model, device, args)

    train_dataset = BPSKSCMRegressionDataset(args.train_data_path, args.train_label_path)
    val_dataset = BPSKSCMRegressionDataset(args.val_data_path, args.val_label_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    transfer_model = copy.deepcopy(base_model).to(device)
    for param in transfer_model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(transfer_model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_function = torch.nn.MSELoss()
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    os.makedirs(args.save_root, exist_ok=True)
    min_val_rmse = float("inf")

    print("================ Start BPSK Trans regression transfer learning ================")
    for epoch in range(args.epochs):
        train_loss = transfer_learning(
            transfer_model, train_loader, feature_bank, optimizer, loss_function, device, epoch + 1, args
        )
        train_rmse = np.sqrt(train_loss)
        val_loss = evaluate(transfer_model, val_loader, loss_function, device, epoch + 1)
        val_rmse = np.sqrt(val_loss)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train RMSE: {train_rmse:.3f} | Val RMSE: {val_rmse:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        lr_schedule.step(val_rmse)
        if val_rmse <= min_val_rmse:
            min_val_rmse = val_rmse
            weight_path = os.path.join(args.save_root, f"weight_transfer_bestIQ_reg_rho{args.rho}.pth")
            torch.save(transfer_model.state_dict(), weight_path)
            print(f">>> Model saved. Best Val RMSE: {min_val_rmse:.3f}")

        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ori_rho", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--feature_snr", type=float, default=20.0)
    parser.add_argument("--feature_snap", type=int, default=1024)
    parser.add_argument("--feature_samples", type=int, default=4)
    parser.add_argument("--w_task", type=float, default=1.0)
    parser.add_argument("--w_cos", type=float, default=0.1)
    parser.add_argument("--w_gram", type=float, default=0.001)

    base_root = graduation_root / "result" / "vit" / "vit_M_8_k_1_base"
    parser.add_argument("--root", type=str, default=str(base_root))

    transfer_save_root = graduation_root / "result" / "vit" / "vit_M_8_k_1_base_transfer"
    parser.add_argument("--save_root", type=str, default=str(transfer_save_root))

    data_dir = graduation_root / "data" / "IQ_Data" / "Single_Source"
    dataset_dir = data_dir / f"SCM_Single_Source_Rho{parser.get_default('rho')}"
    parser.add_argument("--train_data_path", type=str, default=str(dataset_dir / "Train" / "vit_train_data.npy"))
    parser.add_argument("--train_label_path", type=str, default=str(dataset_dir / "Train" / "train_labels.npy"))
    parser.add_argument("--val_data_path", type=str, default=str(dataset_dir / "Val" / "vit_val_data.npy"))
    parser.add_argument("--val_label_path", type=str, default=str(dataset_dir / "Val" / "val_labels.npy"))

    args = parser.parse_args()
    main(args)
