import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ext_lib = r"D:\Python\Project\doa_estimation\Graduation\external\DOA_est_Master-master"
proj_root = r"D:\Python\Project\doa_estimation"
if ext_lib not in sys.path:
    sys.path.insert(0, ext_lib)
if proj_root not in sys.path:
    sys.path.insert(1, proj_root)

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_datasets
from dl_models.CNN_model import CNN_Regression
from dl_models.seven_source_function_gauss import (build_mixed_theta_set, build_random_val_items, build_fixed_family_val_items,
                                                   build_shifted_family_val_items, evaluate_val_items)


def prepare_2ch_input(inputs_complex, T, device):
    B, M, _ = inputs_complex.shape
    R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / T
    X = torch.zeros(B, 2, M, M, device=device)
    X[:, 0] = R.real
    X[:, 1] = R.imag
    max_val = torch.max(torch.abs(X.view(B, -1)), dim=1)[0].view(B, 1, 1, 1)
    X = X / (max_val + 1e-8)
    return X


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rho = 1.0
    snap = 50
    epochs = 100
    batch_size = 128
    val_batch_size = 128

    print(f"🚀 启动 REG-CNN SevenSource | Rho={rho}")

    model = CNN_Regression(out_dim=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    train_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)

    random_val_items = build_random_val_items(rho=rho, snap=snap, batch_size=val_batch_size, theta_num=2000, min_delta_theta=8)
    fixed_val_items = build_fixed_family_val_items(rho=rho, snap=snap, batch_size=val_batch_size, num_samples=512, fixed_snr=5.0)
    shifted_val_items = build_shifted_family_val_items(rho=rho, snap=snap, batch_size=val_batch_size, num_samples=512, fixed_snr=5.0)

    save_dir = r"/result/CNN/SevenSource"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"CNN_Regression_Gaussian_SevenSource_rho{rho}.pth")
    record_path = os.path.join(save_dir, f"best_REG_CNN_SevenSource_rho{rho}_record.json")

    best_score = float('inf')
    best_rand = float('inf')
    best_fixed = float('inf')
    best_shifted = float('inf')

    for epoch in range(epochs):
        train_dataset.clear()

        theta_train, cfg = build_mixed_theta_set(epoch, random_num=6000, centered_num=2500, shifted_num=1500)
        Create_datasets(train_dataset, k=7, theta_set=theta_train, batch_size=128, snap=snap, snr=cfg["snr"], shared_snr=True)
        train_loader = array_Dataloader(train_dataset, batch_size=batch_size, shuffle=True, load_style='torch', input_type='y_t', output_type='doa')

        model.train()
        train_loss, train_steps = 0.0, 0

        print(f"\n[Epoch {epoch + 1}/{epochs}] 正在训练 REG-CNN SevenSource...")
        print(f" -> Stage cfg: snr={cfg['snr']} | random_min_delta={cfg['random_min_delta']} | family_d_range={cfg['family_d_range']} | theta_num={len(theta_train)}")

        for inputs_complex, labels_doa in tqdm(train_loader, leave=False):
            inputs_complex = inputs_complex.to(device)
            true_angles = labels_doa.float().to(device).view(-1, 7)
            X_train = prepare_2ch_input(inputs_complex, snap, device)

            optimizer.zero_grad()
            pred_angles = model(X_train)

            pred_sorted, _ = torch.sort(pred_angles, dim=1)
            true_sorted, _ = torch.sort(true_angles, dim=1)
            loss = criterion(pred_sorted, true_sorted)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        rand_avg, rand_detail = evaluate_val_items(model, "REG-CNN", random_val_items, device, snap)
        fixed_avg, fixed_detail = evaluate_val_items(model, "REG-CNN", fixed_val_items, device, snap)
        shifted_avg, shifted_detail = evaluate_val_items(model, "REG-CNN", shifted_val_items, device, snap)

        score = rand_avg + 0.6 * fixed_avg + 0.6 * shifted_avg
        scheduler.step(score)

        print(f" -> Train Loss: {train_loss / train_steps:.4f} | RandAvg: {rand_avg:.4f}° | FixedAvg: {fixed_avg:.4f}° | ShiftedAvg: {shifted_avg:.4f}° | Score: {score:.4f} | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if score < best_score:
            best_score = score
            best_rand = rand_avg
            best_fixed = fixed_avg
            best_shifted = shifted_avg

            torch.save(model.state_dict(), save_path)

            with open(record_path, "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, "best_score": best_score, "best_rand_avg": best_rand, "best_fixed_avg": best_fixed, "best_shifted_avg": best_shifted, "rand_detail": rand_detail, "fixed_detail": fixed_detail, "shifted_detail": shifted_detail, "save_path": save_path, "stage_cfg": cfg}, f, indent=4, ensure_ascii=False)

            print(f"⭐ 已保存 REG-CNN SevenSource 最优模型 | Score: {best_score:.4f} | RandAvg: {best_rand:.4f}° | FixedAvg: {best_fixed:.4f}° | ShiftedAvg: {best_shifted:.4f}°")

    print(f"\n✅ REG-CNN SevenSource 训练结束，最优模型已保存至: {save_path}")
    print(f"Best Score={best_score:.4f} | Best RandAvg={best_rand:.4f}° | Best FixedAvg={best_fixed:.4f}° | Best ShiftedAvg={best_shifted:.4f}°")


if __name__ == "__main__":
    main()