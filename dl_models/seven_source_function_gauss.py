import torch
import numpy as np
import scipy.signal
from data.data_create.Create_k_source_dataset90 import Create_datasets
from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta
from dl_models.MLP import scm_to_vec72
from dl_models.embeding_layer import calc_rmse, get_continuous_angle_k7


VAL_SNR_LIST = [-20, -15, -10, -5, 0, 5, 10]

FIXED_VAL_D_LIST = [7.5, 9.5, 13.5, 17.0]
SHIFTED_VAL_D_LIST = [8.5, 11.5, 15.5]
SHIFTED_VAL_CENTER_LIST = [-30.0, -12.0, 12.0, 30.0]

def get_stage_cfg(epoch):
    if epoch < 30:
        return {"snr": (0, 10), "random_min_delta": 14, "family_d_range": (10.0, 20.0)}
    elif epoch < 70:
        return {"snr": (-10, 10), "random_min_delta": 10, "family_d_range": (8.0, 20.0)}
    else:
        return {"snr": (-20, 10), "random_min_delta": 8, "family_d_range": (6.0, 20.0)}


def to_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def normalize_scm(x):
    B = x.shape[0]
    max_vals = torch.max(torch.abs(x.reshape(B, -1)), dim=1)[0].view(B, 1, 1, 1)
    return x / (max_vals + 1e-8)


def build_random_val_items(rho, snap, batch_size=64, theta_num=2000, min_delta_theta=8):
    theta_val = Create_random_k_input_theta(7, -90, 90, theta_num, min_delta_theta=min_delta_theta)
    theta_val = np.array(theta_val, dtype=np.float32)
    valid_mask = (~np.isnan(theta_val).any(axis=1)) & (np.max(theta_val, axis=1) <= 90) & (np.min(theta_val, axis=1) >= -90)
    theta_val = theta_val[valid_mask]

    val_items = []

    for snr in VAL_SNR_LIST:
        val_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
        Create_datasets(val_dataset, k=7, theta_set=theta_val.copy(), batch_size=128, snap=snap, snr=snr, shared_snr=True)

        loader_scm = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                           load_style='torch', input_type='scm', output_type='doa'))
        loader_y = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False, load_style='torch',
                                    input_type='y_t', output_type='doa'))

        val_items.append({"snr": snr, "loader_scm": loader_scm, "loader_y": loader_y})

    return val_items


def build_fixed_family_val_items(rho, snap, batch_size=64, num_samples=512, fixed_snr=5.0):
    val_items = []

    for d in FIXED_VAL_D_LIST:
        center = 0.0
        template = np.array([center - 3*d, center - 2*d, center - d, center, center + d, center + 2*d, center + 3*d], dtype=np.float32)
        theta_set = np.tile(template, (num_samples, 1)).astype(np.float32)

        val_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
        Create_datasets(val_dataset, k=7, theta_set=theta_set, batch_size=128, snap=snap, snr=fixed_snr, shared_snr=True)

        loader_scm = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                      load_style='torch', input_type='scm', output_type='doa'))
        loader_y = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                    load_style='torch', input_type='y_t', output_type='doa'))

        val_items.append({"d": d, "center": center, "template": template.tolist(), "loader_scm": loader_scm, "loader_y": loader_y})

    return val_items


def build_shifted_family_val_items(rho, snap, batch_size=64, num_samples=512, fixed_snr=5.0):
    val_items = []

    for d in SHIFTED_VAL_D_LIST:
        for center in SHIFTED_VAL_CENTER_LIST:
            template = np.array([center - 3*d, center - 2*d, center - d, center, center + d, center + 2*d, center + 3*d], dtype=np.float32)

            if template.min() < -90 or template.max() > 90:
                continue

            theta_set = np.tile(template, (num_samples, 1)).astype(np.float32)

            val_dataset = ULA_dataset(M=8, st_angle=-90, ed_angle=90, step=1, rho=rho)
            Create_datasets(val_dataset, k=7, theta_set=theta_set, batch_size=128, snap=snap, snr=fixed_snr, shared_snr=True)

            loader_scm = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                               load_style='torch', input_type='scm', output_type='doa'))
            loader_y = list(array_Dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                             load_style='torch', input_type='y_t', output_type='doa'))

            val_items.append({"d": d, "center": center, "template": template.tolist(), "loader_scm": loader_scm, "loader_y": loader_y})

    return val_items


@torch.no_grad()
def predict_batch(model, model_type, inputs_scm, inputs_complex, device, snap, M=8, K=7):
    if model_type == "ViT":
        x = normalize_scm(inputs_scm.to(device).float())
        try:
            pred = model(x, logits=False)
        except TypeError:
            pred = model(x)
        pred_sorted, _ = torch.sort(pred, dim=1)
        return pred_sorted

    inputs_complex = inputs_complex.to(device)
    B = inputs_complex.shape[0]
    R = torch.bmm(inputs_complex, inputs_complex.conj().transpose(1, 2)) / snap

    if model_type == "REG-CNN":
        x = torch.zeros(B, 2, M, M, device=device)
        x[:, 0] = R.real
        x[:, 1] = R.imag
        max_v = torch.max(torch.abs(x.reshape(B, -1)), dim=1)[0].view(B, 1, 1, 1)
        x = x / (max_v + 1e-8)
        pred = model(x)
        pred_sorted, _ = torch.sort(pred, dim=1)
        return pred_sorted

    if model_type == "SPE-CNN":
        x = torch.zeros(B, 3, M, M, device=device)
        x[:, 0] = R.real
        x[:, 1] = R.imag
        x[:, 2] = R.angle() / torch.pi
        max_v = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].view(B, 1, 1)
        x[:, 0] = x[:, 0] / (max_v + 1e-8)
        x[:, 1] = x[:, 1] / (max_v + 1e-8)
        pred = get_continuous_angle_k7(model(x), K=K, radius=2)
        pred_sorted, _ = torch.sort(pred, dim=1)
        return pred_sorted

    if model_type == "IQ-ResNet":
        x = torch.cat([inputs_complex.real, inputs_complex.imag], dim=1).unsqueeze(1).float()
        rms_val = torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True))
        x = x / (rms_val + 1e-8)
        pred = get_continuous_angle_k7(model(x), K=K, radius=2)
        pred_sorted, _ = torch.sort(pred, dim=1)
        return pred_sorted

    if model_type == "Learning-SPICE":
        max_v = torch.max(torch.abs(R.reshape(B, -1)), dim=1)[0].reshape(B, 1, 1)
        R_norm = R / (max_v + 1e-8)
        x = scm_to_vec72(R_norm)
        pred = get_continuous_angle_k7(model(x), K=K, radius=2)
        pred_sorted, _ = torch.sort(pred, dim=1)
        return pred_sorted

    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def evaluate_val_items(model, model_type, val_items, device, snap):
    model.eval()

    rmse_list = []
    detail = []

    for item in val_items:
        total_mse = 0.0
        total_steps = 0

        for (inputs_scm, _), (inputs_complex, labels) in zip(item["loader_scm"], item["loader_y"]):
            labels = labels.to(device).float().view(-1, 7)
            labels_sorted, _ = torch.sort(labels, dim=1)

            pred_sorted = predict_batch(model, model_type, inputs_scm, inputs_complex, device, snap)

            total_mse += to_float(calc_rmse(pred_sorted, labels_sorted))
            total_steps += 1

        if total_steps == 0:
            raise RuntimeError(
                "验证 loader 为空：请检查 build_*_val_items 中是否把 array_Dataloader 转成 list，或是否数据集生成失败。")

        rmse = float(np.sqrt(total_mse / total_steps))
        rmse_list.append(rmse)

        info = {k: item[k] for k in item.keys() if k not in ["loader_scm", "loader_y"]}
        info["rmse"] = rmse
        detail.append(info)

    avg_rmse = float(np.mean(rmse_list))
    return avg_rmse, detail


def sample_random_theta_set(theta_num, min_delta_theta):
    theta_set = Create_random_k_input_theta(7, -90, 90, theta_num, min_delta_theta=min_delta_theta)
    theta_set = np.array(theta_set, dtype=np.float32)
    valid_mask = (~np.isnan(theta_set).any(axis=1)) & (np.max(theta_set, axis=1) <= 90) & (
                np.min(theta_set, axis=1) >= -90)
    return theta_set[valid_mask]


def sample_d_piecewise(num_samples, d_low, d_high):
    n1 = int(num_samples * 0.35)
    n2 = int(num_samples * 0.35)
    n3 = num_samples - n1 - n2

    d1 = np.random.uniform(max(d_low, 6.0), min(d_high, 10.0), size=n1)
    d2 = np.random.uniform(max(d_low, 10.0), min(d_high, 15.0), size=n2)
    d3 = np.random.uniform(max(d_low, 15.0), d_high, size=n3)

    d = np.concatenate([d1, d2, d3]).astype(np.float32)
    np.random.shuffle(d)
    return d


def sample_centered_family_theta_set(num_samples, d_low, d_high):
    d = sample_d_piecewise(num_samples, d_low, d_high)
    center = np.random.uniform(-10.0, 10.0, size=num_samples).astype(np.float32)

    center_low = -90 + 3.0 * d
    center_high = 90 - 3.0 * d
    center = np.clip(center, center_low, center_high)

    theta_set = np.stack([center - 3*d, center - 2*d, center - d, center, center + d, center + 2*d, center + 3*d], axis=1).astype(np.float32)
    return theta_set


def sample_shifted_family_theta_set(num_samples, d_low, d_high):
    d = sample_d_piecewise(num_samples, d_low, d_high)

    center_low = -90 + 3.0 * d
    center_high = 90 - 3.0 * d

    n_inner = int(num_samples * 0.6)
    n_edge = num_samples - n_inner

    center_inner = np.random.uniform(np.maximum(center_low[:n_inner], -30.0), np.minimum(center_high[:n_inner], 30.0)).astype(np.float32)

    d_edge = d[n_inner:]
    edge_low = center_low[n_inner:]
    edge_high = center_high[n_inner:]

    sign = np.random.choice([-1.0, 1.0], size=n_edge).astype(np.float32)
    edge_abs_low = 0.45 * np.minimum(np.abs(edge_low), np.abs(edge_high))
    edge_abs_high = 0.85 * np.minimum(np.abs(edge_low), np.abs(edge_high))
    center_edge = sign * np.random.uniform(edge_abs_low, edge_abs_high).astype(np.float32)
    center_edge = np.clip(center_edge, edge_low, edge_high)

    center = np.concatenate([center_inner, center_edge]).astype(np.float32)
    d = np.concatenate([d[:n_inner], d_edge]).astype(np.float32)

    theta_set = np.stack([center - 3*d, center - 2*d, center - d, center, center + d, center + 2*d, center + 3*d], axis=1).astype(np.float32)
    return theta_set


def build_mixed_theta_set(epoch, random_num=6000, centered_num=2500, shifted_num=1500):
    cfg = get_stage_cfg(epoch)

    theta_rand = sample_random_theta_set(theta_num=random_num, min_delta_theta=cfg["random_min_delta"])
    theta_centered = sample_centered_family_theta_set(num_samples=centered_num, d_low=cfg["family_d_range"][0], d_high=cfg["family_d_range"][1])
    theta_shifted = sample_shifted_family_theta_set(num_samples=shifted_num, d_low=cfg["family_d_range"][0], d_high=cfg["family_d_range"][1])

    theta_all = np.concatenate([theta_rand, theta_centered, theta_shifted], axis=0)
    np.random.shuffle(theta_all)

    return theta_all, cfg

