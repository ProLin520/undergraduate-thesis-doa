import os
import sys
import json
import numpy as np
import torch
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
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from dl_models.seven_source_function_gauss import build_mixed_theta_set, build_random_val_items, build_fixed_family_val_items, build_shifted_family_val_items, evaluate_val_items, normalize_scm


def save_args(argparser, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(vars(argparser), f, indent=4, ensure_ascii=False)


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch):
    model.train()
    accu_loss = torch.zeros(1, device=device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        inputs, labels = data
        inputs = normalize_scm(inputs.to(device).float())
        labels = labels.to(device).float().view(-1, 7)

        pred = model(inputs)
        pred_sorted, _ = torch.sort(pred, dim=1)
        labels_sorted, _ = torch.sort(labels, dim=1)

        loss = loss_function(pred_sorted, labels_sorted)

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss", loss)
            sys.exit(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] RMSE_loss: {:.3f}".format(epoch, np.sqrt(accu_loss.item() / (step + 1)))

    return accu_loss.item() / (step + 1)


def main(args):
    save_path = args.save_root
    os.makedirs(save_path, exist_ok=True)
    save_args(args, os.path.join(save_path, "laboratory_set_v3.json"))

    loss_function = torch.nn.MSELoss()
    embeding_dim = 768
    model_type = "ViT"

    print(f"开始训练 ViT SevenSource v3，rho={args.rho}")

    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim, out_dims=args.k, drop_ratio=0, attn_drop_ratio=0).to(args.device)

    train_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)

    random_val_items = build_random_val_items(rho=args.rho, snap=args.snap, batch_size=128, theta_num=2000, min_delta_theta=8)
    fixed_val_items = build_fixed_family_val_items(rho=args.rho, snap=args.snap, batch_size=128, num_samples=512, fixed_snr=5.0)
    shifted_val_items = build_shifted_family_val_items(rho=args.rho, snap=args.snap, batch_size=128, num_samples=512, fixed_snr=5.0)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_score = float("inf")
    best_rand = float("inf")
    best_fixed = float("inf")
    best_shifted = float("inf")

    weight_name = f"weight_base_SevenSource_rho{args.rho}_v3.pth"
    record_path = os.path.join(save_path, "best_ViT_SevenSource_v3_record.json")

    for epoch in range(args.epochs):
        train_dataset.clear()

        theta_train, cfg = build_mixed_theta_set(epoch, random_num=6000, centered_num=2500, shifted_num=1500)
        Create_datasets(train_dataset, args.k, theta_train, batch_size=128, snap=args.snap, snr=cfg["snr"], shared_snr=True)

        train_loader = array_Dataloader(train_dataset, 128, shuffle=True, load_style='torch', input_type='scm', output_type='doa')

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] 正在训练 ViT SevenSource v3...")
        print(f" -> Stage cfg: snr={cfg['snr']} | random_min_delta={cfg['random_min_delta']} | family_d_range={cfg['family_d_range']} | theta_num={len(theta_train)}")

        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, args.device, epoch + 1)

        rand_avg, rand_detail = evaluate_val_items(model, model_type, random_val_items, args.device, args.snap)
        fixed_avg, fixed_detail = evaluate_val_items(model, model_type, fixed_val_items, args.device, args.snap)
        shifted_avg, shifted_detail = evaluate_val_items(model, model_type, shifted_val_items, args.device, args.snap)

        score = rand_avg + 0.6 * fixed_avg + 0.6 * shifted_avg
        scheduler.step(score)

        print(f"[Epoch {epoch + 1}/{args.epochs}] TrainLoss: {train_loss:.4f} | RandAvg: {rand_avg:.4f}° | FixedAvg: {fixed_avg:.4f}° | ShiftedAvg: {shifted_avg:.4f}° | Score: {score:.4f} | LR: {optimizer.param_groups[0]['lr']:.4e}")

        if score <= best_score:
            best_score = score
            best_rand = rand_avg
            best_fixed = fixed_avg
            best_shifted = shifted_avg

            torch.save(model.state_dict(), os.path.join(save_path, weight_name))

            with open(record_path, "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, "best_score": best_score, "best_rand_avg": best_rand, "best_fixed_avg": best_fixed, "best_shifted_avg": best_shifted, "rand_detail": rand_detail, "fixed_detail": fixed_detail, "shifted_detail": shifted_detail, "stage_cfg": cfg, "save_path": os.path.join(save_path, weight_name)}, f, indent=4, ensure_ascii=False)

            print(f"⭐ 已保存 ViT SevenSource v3 最优模型 | Score={best_score:.4f} | RandAvg={best_rand:.4f}° | FixedAvg={best_fixed:.4f}° | ShiftedAvg={best_shifted:.4f}°")

    print(f"\n✅ ViT SevenSource v3 训练结束，最优模型保存在: {os.path.join(save_path, weight_name)}")
    print(f"Best Score={best_score:.4f} | Best RandAvg={best_rand:.4f}° | Best FixedAvg={best_fixed:.4f}° | Best ShiftedAvg={best_shifted:.4f}°")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--snap', type=int, default=50)
    parser.add_argument('--signal_range', type=tuple, default=(-90, 90))
    parser.add_argument('--step_used', type=float, default=1)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--save_root', type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    main(args)