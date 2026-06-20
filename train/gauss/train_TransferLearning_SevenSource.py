import os
import sys
import json
import copy
import argparse
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
if 'utils' in sys.modules:
    del sys.modules['utils']

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_datasets
from data.data_create.theta_creater import same_data_Creater
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding
from dl_models.seven_source_function_gauss import build_mixed_theta_set, build_random_val_items, build_fixed_family_val_items, build_shifted_family_val_items, evaluate_val_items, normalize_scm


def save_args(args, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def geometry_loss(pred, labels):
    pred_sorted, _ = torch.sort(pred, dim=1)
    labels_sorted, _ = torch.sort(labels, dim=1)

    loss_angle = torch.mean((pred_sorted - labels_sorted) ** 2)

    pred_diff = pred_sorted[:, 1:] - pred_sorted[:, :-1]
    label_diff = labels_sorted[:, 1:] - labels_sorted[:, :-1]
    loss_diff = torch.mean((pred_diff - label_diff) ** 2)

    pred_span = pred_sorted[:, -1] - pred_sorted[:, 0]
    label_span = labels_sorted[:, -1] - labels_sorted[:, 0]
    loss_span = torch.mean((pred_span - label_span) ** 2)

    pred_center = torch.mean(pred_sorted, dim=1)
    label_center = torch.mean(labels_sorted, dim=1)
    loss_center = torch.mean((pred_center - label_center) ** 2)

    return loss_angle, loss_diff, loss_span, loss_center


def feature_align_loss(transfer_features, base_features):
    pred_vec = torch.nn.functional.normalize(transfer_features, dim=-1)
    base_vec = torch.nn.functional.normalize(base_features, dim=-1)
    loss_cos = torch.mean(torch.ones(pred_vec.shape[0], device=pred_vec.device) - torch.sum(pred_vec * base_vec, dim=-1))

    gram_pred = transfer_features.transpose(-1, -2) @ transfer_features
    gram_base = base_features.transpose(-1, -2) @ base_features

    gram_pred = 0.5 * (gram_pred + gram_pred.transpose(-1, -2))
    gram_base = 0.5 * (gram_base + gram_base.transpose(-1, -2))

    loss_gram = torch.mean((gram_pred - gram_base) ** 2)

    return loss_cos, loss_gram


def transfer_learning_one_epoch(transfer_model, base_model, data_loader, data_creater, optimizer, device, epoch, snap, source_snr=20):
    transfer_model.train()
    base_model.eval()

    accu_angle = 0.0
    accu_diff = 0.0
    accu_span = 0.0
    accu_center = 0.0
    accu_cos = 0.0
    accu_gram = 0.0

    w_angle = 1.0
    w_diff = 0.30
    w_span = 0.20
    w_center = 0.05
    w_cos = 0.30
    w_gram = 0.0005

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        inputs, labels = data
        inputs = normalize_scm(inputs.to(device).float())
        labels = labels.to(device).float().view(-1, 7)

        pred_doa = transfer_model(inputs, logits=False)
        loss_angle, loss_diff, loss_span, loss_center = geometry_loss(pred_doa, labels)

        transfer_features = transfer_model(inputs, logits=True)

        with torch.no_grad():
            source_domain_data = data_creater(labels, snap=snap, snr=source_snr).to(device).float()
            source_domain_data = normalize_scm(source_domain_data)
            base_features = base_model(source_domain_data, logits=True)

        loss_cos, loss_gram = feature_align_loss(transfer_features, base_features)

        loss = (
            w_angle * loss_angle
            + w_diff * loss_diff
            + w_span * loss_span
            + w_center * loss_center
            + w_cos * loss_cos
            + w_gram * loss_gram
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_norm=0.1)
        optimizer.step()

        accu_angle += loss_angle.detach().item()
        accu_diff += loss_diff.detach().item()
        accu_span += loss_span.detach().item()
        accu_center += loss_center.detach().item()
        accu_cos += loss_cos.detach().item()
        accu_gram += loss_gram.detach().item()

        data_loader.desc = "[train epoch {}] Angle: {:.3f} | Diff: {:.3f} | Span: {:.3f} | C: {:.3f} | Cos: {:.3f} | Gram: {:.5f}".format(
            epoch,
            loss_angle.item(),
            loss_diff.item(),
            loss_span.item(),
            loss_center.item(),
            loss_cos.item(),
            loss_gram.item() * w_gram
        )

    n = step + 1
    return {
        "angle": accu_angle / n,
        "diff": accu_diff / n,
        "span": accu_span / n,
        "center": accu_center / n,
        "cos": accu_cos / n,
        "gram": accu_gram / n
    }


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    save_args(args, os.path.join(args.save_root, "transfer_seven_source_v3_set.json"))

    embeding_dim = 768
    model_type = "ViT"

    print("🚀 Building ViT SevenSource v3 base model...")
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim, out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    base_weight_path = os.path.join(args.base_root, args.base_weight_name)
    print(f"[Base ViT] loading: {base_weight_path}")
    base_model.load_state_dict(torch.load(base_weight_path, map_location=args.device), strict=True)
    base_model.to(args.device)
    base_model.eval()

    transfer_model = copy.deepcopy(base_model)
    for param in transfer_model.parameters():
        param.requires_grad = True
    transfer_model.to(args.device)

    train_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
    base_dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.ori_rho)
    data_creater = same_data_Creater(base_dataset, 'scm')

    print("🚀 Building v3 validation items...")
    random_val_items = build_random_val_items(rho=args.rho, snap=args.snap, batch_size=args.val_batch_size, theta_num=2000, min_delta_theta=8)
    fixed_val_items = build_fixed_family_val_items(rho=args.rho, snap=args.snap, batch_size=args.val_batch_size, num_samples=512, fixed_snr=5.0)
    shifted_val_items = build_shifted_family_val_items(rho=args.rho, snap=args.snap, batch_size=args.val_batch_size, num_samples=512, fixed_snr=5.0)

    optimizer = optim.Adam([p for p in transfer_model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    best_score = float('inf')
    best_rand = float('inf')
    best_fixed = float('inf')
    best_shifted = float('inf')

    save_weight_path = os.path.join(args.save_root, args.save_weight_name)
    record_path = os.path.join(args.save_root, "best_transfer_seven_source_v3_record.json")

    for epoch in range(args.epochs):
        train_dataset.clear()

        theta_train, cfg = build_mixed_theta_set(epoch, random_num=args.random_num, centered_num=args.centered_num, shifted_num=args.shifted_num)
        Create_datasets(train_dataset, args.k, theta_train, batch_size=128, snap=args.snap, snr=cfg["snr"], shared_snr=True)

        train_loader = array_Dataloader(train_dataset, args.batch_size, shuffle=True, load_style='torch', input_type='scm', output_type='doa')

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] 正在训练 ViT SevenSource Transfer v3...")
        print(f" -> Stage cfg: snr={cfg['snr']} | random_min_delta={cfg['random_min_delta']} | family_d_range={cfg['family_d_range']} | theta_num={len(theta_train)}")

        train_log = transfer_learning_one_epoch(transfer_model, base_model, train_loader, data_creater, optimizer, args.device, epoch + 1, args.snap, source_snr=args.source_snr)

        rand_avg, rand_detail = evaluate_val_items(transfer_model, model_type, random_val_items, args.device, args.snap)
        fixed_avg, fixed_detail = evaluate_val_items(transfer_model, model_type, fixed_val_items, args.device, args.snap)
        shifted_avg, shifted_detail = evaluate_val_items(transfer_model, model_type, shifted_val_items, args.device, args.snap)

        score = rand_avg + args.fixed_score_weight * fixed_avg + args.shifted_score_weight * shifted_avg
        scheduler.step(score)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Angle: {train_log['angle']:.4f} | Diff: {train_log['diff']:.4f} | Span: {train_log['span']:.4f} | "
            f"RandAvg: {rand_avg:.4f}° | FixedAvg: {fixed_avg:.4f}° | ShiftedAvg: {shifted_avg:.4f}° | "
            f"Score: {score:.4f} | LR: {optimizer.param_groups[0]['lr']:.4e}"
        )

        if score < best_score:
            best_score = score
            best_rand = rand_avg
            best_fixed = fixed_avg
            best_shifted = shifted_avg

            torch.save(transfer_model.state_dict(), save_weight_path)

            with open(record_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "best_score": best_score,
                    "best_rand_avg": best_rand,
                    "best_fixed_avg": best_fixed,
                    "best_shifted_avg": best_shifted,
                    "rand_detail": rand_detail,
                    "fixed_detail": fixed_detail,
                    "shifted_detail": shifted_detail,
                    "stage_cfg": cfg,
                    "weight_path": save_weight_path
                }, f, indent=4, ensure_ascii=False)

            print(
                f"⭐ 保存最优 ViT Transfer v3 | "
                f"Score={best_score:.4f} | RandAvg={best_rand:.4f}° | "
                f"FixedAvg={best_fixed:.4f}° | ShiftedAvg={best_shifted:.4f}°"
            )

    print(f"\n✅ ViT SevenSource Transfer v3 训练结束")
    print(f"最优模型保存至: {save_weight_path}")
    print(f"Best Score={best_score:.4f} | Best RandAvg={best_rand:.4f}° | Best FixedAvg={best_fixed:.4f}° | Best ShiftedAvg={best_shifted:.4f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--snap', type=int, default=50)
    parser.add_argument('--signal_range', type=tuple, default=(-90, 90))
    parser.add_argument('--step_used', type=float, default=1)
    parser.add_argument('--ori_rho', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=1.0)

    parser.add_argument('--base_root', type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base")
    parser.add_argument('--base_weight_name', type=str, default="weight_base_SevenSource_rho1.0_v3.pth")
    parser.add_argument('--save_root', type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_7_base_transfer")
    parser.add_argument('--save_weight_name', type=str, default="1weight_transfer_SevenSource_rho1.0_v3.pth")

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)

    # Transfer v3 使用更偏 fixed / shifted 的混合比例
    parser.add_argument('--random_num', type=int, default=5000)
    parser.add_argument('--centered_num', type=int, default=3000)
    parser.add_argument('--shifted_num', type=int, default=2000)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--source_snr', type=float, default=20.0)

    # Transfer 阶段更关注 fixed / shifted，但保留 random
    parser.add_argument('--fixed_score_weight', type=float, default=0.8)
    parser.add_argument('--shifted_score_weight', type=float, default=0.8)

    args = parser.parse_args()
    main(args)