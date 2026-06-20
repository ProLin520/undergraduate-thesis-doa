import torch
import torch.nn as nn
import numpy as np
import scipy.signal
from scipy.signal import find_peaks

class scm_embeding(nn.Module):
    def __init__(self, M, ebedding_dim):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(2*M, ebedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -3)
        x = torch.flatten(x, -2)
        x = self.linear(x)
        return x


def music_algorithm(R, num_sources=1, M=8, d_lambda=0.5):
    # 移除无效的对角加载，仅保留 1e-8 防止纯数学上的除 0 报错
    R_robust = R + 1e-8 * np.eye(M)

    eigenvalues, eigenvectors = np.linalg.eigh(R_robust)
    idx = np.argsort(eigenvalues)
    En = eigenvectors[:, idx[:-num_sources]]

    # 向量化空间谱搜索
    search_angles = np.arange(-90, 91, 0.5)
    angle_rad = np.deg2rad(search_angles)

    # 严重警告修复：必须使用 -1j，与底层数据的物理流型严丝合缝
    a = np.exp(-1j * 2 * np.pi * d_lambda * np.arange(M)[:, None] * np.sin(angle_rad))

    proj = En.conj().T @ a
    denom = np.sum(np.abs(proj) ** 2, axis=0)

    P_music = 1.0 / (denom + 1e-12)
    return search_angles[np.argmax(P_music)]


def music_algorithm_k3(R, num_sources=3, M=8, d_lambda=0.5):
    """原生网格谱搜索 MUSIC (完美复刻物理分辨极限)"""
    # 移除无效的对角加载，仅保留 1e-8 防止纯数学上的除 0 报错
    R_robust = R + 1e-8 * np.eye(M)

    eigenvalues, eigenvectors = np.linalg.eigh(R_robust)
    idx = np.argsort(eigenvalues)
    En = eigenvectors[:, idx[:-num_sources]]

    # 向量化空间谱搜索 (受限于 0.5度 网格分辨率)
    search_angles = np.arange(-90, 91, 0.5)
    angle_rad = np.deg2rad(search_angles)

    # 使用 -1j，与底层数据的物理流型严丝合缝
    a = np.exp(-1j * 2 * np.pi * d_lambda * np.arange(M)[:, None] * np.sin(angle_rad))

    proj = En.conj().T @ a
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    P_music = 1.0 / (denom + 1e-12)

    # 🌟 核心修改：寻找 K 个局部波峰，而不是全局唯一最大值
    peaks_idx, _ = scipy.signal.find_peaks(P_music)
    peak_vals = P_music[peaks_idx]

    # 按波峰高度从大到小排序，取前 K 个
    sorted_idx = np.argsort(peak_vals)[::-1]
    top_k_indices = peaks_idx[sorted_idx[:num_sources]]
    doas = search_angles[top_k_indices]

    # 🌟 公平机制：如果因为波峰合并找不够 K 个峰，直接复制最后一个找到的峰！
    if len(doas) < num_sources:
        if len(doas) > 0:
            # 采用边缘复制，比如找到了 [11.5]，缺两个，就变成 [11.5, 11.5, 11.5]
            doas = np.pad(doas, (0, num_sources - len(doas)), 'edge')
        else:
            # 只有在极端恶劣（一个峰都没找到）时才补 0
            doas = np.zeros(num_sources)

    return doas


def music_algorithm_k7(R, M=8, d_lambda=0.5, search_step=0.5, min_peak_sep_deg=1.0):
    """七信源专用 MUSIC 角度搜索
    返回长度为 7 的 DOA 数组（单位：度）
    """
    num_sources = 7

    # 轻微数值稳定
    R_robust = R + 1e-8 * np.eye(M)

    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(R_robust)
    idx = np.argsort(eigenvalues)
    En = eigenvectors[:, idx[:-num_sources]]   # M-K = 1 维噪声子空间

    # 空间谱搜索
    search_angles = np.arange(-90, 90 + search_step, search_step)
    angle_rad = np.deg2rad(search_angles)

    a = np.exp(-1j * 2 * np.pi * d_lambda * np.arange(M)[:, None] * np.sin(angle_rad))
    proj = En.conj().T @ a
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    P_music = 1.0 / (denom + 1e-12)

    # ---------- 第一步：找局部峰 ----------
    min_peak_distance = max(1, int(round(min_peak_sep_deg / search_step)))
    peaks_idx, _ = scipy.signal.find_peaks(P_music, distance=min_peak_distance)

    # 如果一个局部峰都找不到，直接从全谱最高点里选
    if len(peaks_idx) == 0:
        candidate_idx = np.argsort(P_music)[::-1]
    else:
        peak_vals = P_music[peaks_idx]
        sorted_peak_order = np.argsort(peak_vals)[::-1]
        candidate_idx = peaks_idx[sorted_peak_order]

    # ---------- 第二步：按角度间隔做 NMS 选 7 个 ----------
    selected = []
    for idx_peak in candidate_idx:
        angle = search_angles[idx_peak]
        if all(abs(angle - search_angles[j]) >= min_peak_sep_deg for j in selected):
            selected.append(idx_peak)
        if len(selected) == num_sources:
            break

    # ---------- 第三步：如果还不够，再从全谱高点补 ----------
    if len(selected) < num_sources:
        global_sorted_idx = np.argsort(P_music)[::-1]
        for idx_peak in global_sorted_idx:
            angle = search_angles[idx_peak]
            if all(abs(angle - search_angles[j]) >= min_peak_sep_deg for j in selected):
                selected.append(idx_peak)
            if len(selected) == num_sources:
                break

    # ---------- 第四步：还不够就复制最后一个 ----------
    doas = search_angles[selected]
    if len(doas) < num_sources:
        if len(doas) > 0:
            doas = np.pad(doas, (0, num_sources - len(doas)), mode='edge')
        else:
            doas = np.zeros(num_sources)

    return np.sort(doas)


def music_batch_fast(X_complex_batch, num_sources=2, M=8, d_lambda=0.5):
    T = X_complex_batch.shape[2]
    R_batch = X_complex_batch @ X_complex_batch.conj().transpose(0, 2, 1) / T
    evals, evecs = np.linalg.eigh(R_batch)
    En = evecs[:, :, :-num_sources]

    search_angles = np.arange(-90, 91, 0.1)
    angle_rad = np.deg2rad(search_angles)
    A_search = np.exp(-1j * 2 * np.pi * d_lambda * np.arange(M)[:, None] * np.sin(angle_rad))
    noise_proj = np.matmul(En.conj().transpose(0, 2, 1), A_search)
    P_music_batch = 1.0 / np.sum(np.abs(noise_proj) ** 2, axis=1)

    preds = []
    min_peak_distance = int(3.0 / 0.1)  # 至少相隔约 3°
    for P_music in P_music_batch:
        peaks, _ = find_peaks(P_music, distance=min_peak_distance)
        if len(peaks) >= 2:
            top2_peaks = peaks[np.argsort(P_music[peaks])[-2:]]
            preds.append(np.sort(search_angles[top2_peaks]))
        else:
            top2_idx = np.argsort(P_music)[-2:]
            preds.append(np.sort(search_angles[top2_idx]))
    return np.array(preds)


def convert_to_complex(iq_matrix):
    M = iq_matrix.shape[0] // 2
    return iq_matrix[:M, :] + 1j * iq_matrix[M:, :]


def get_continuous_angle(outputs, radius=2):
    """用 Windowed Soft-Argmax 提取连续角度，突破网格精度极限"""
    probs = torch.softmax(outputs, dim=1)
    _, max_idx = torch.max(probs, dim=1)
    B = probs.shape[0]
    angles = torch.zeros(B, device=outputs.device)
    grid = torch.linspace(-90, 90, 181, device=outputs.device)

    for i in range(B):
        idx = max_idx[i]
        # 在最高峰左右各取 radius 个点（比如前后2度，共5个点）
        start = max(0, idx - radius)
        end = min(181, idx + radius + 1)

        window_probs = probs[i, start:end]
        window_grid = grid[start:end]

        # 局部概率归一化
        window_probs = window_probs / (window_probs.sum() + 1e-8)
        # 求期望：概率 × 角度
        angles[i] = torch.sum(window_probs * window_grid)

    return angles


def get_continuous_angle_k3(outputs, K=3, radius=2):
    """K=3 专用的 1D-NMS + Windowed Soft-Argmax 亚网格提取 (带安全填充)"""
    device = outputs.device
    B = outputs.shape[0]
    probs = torch.sigmoid(outputs)

    probs_unsqueeze = probs.unsqueeze(1)
    pooled = torch.nn.functional.max_pool1d(probs_unsqueeze, kernel_size=5, stride=1, padding=2)
    peak_mask = (probs_unsqueeze == pooled).float()
    peak_probs = (probs_unsqueeze * peak_mask).squeeze(1)

    continuous_angles = torch.zeros(B, K, device=device)
    grid = torch.linspace(-90, 90, 181, device=device)

    for b in range(B):
        # 1. 过滤出真正的波峰
        valid_peaks = torch.nonzero(peak_probs[b] > 1e-5).squeeze(-1)

        if len(valid_peaks) > 0:
            peak_vals = peak_probs[b, valid_peaks]
            _, sorted_idx = torch.sort(peak_vals, descending=True)
            top_peaks = valid_peaks[sorted_idx]
        else:
            top_peaks = torch.tensor([], dtype=torch.long, device=device)

        last_valid_angle = 0.0  # 用于兜底

        for k in range(K):
            if k < len(top_peaks):
                idx = top_peaks[k].item()
                start = max(0, idx - radius)
                end = min(181, idx + radius + 1)

                window_probs = probs[b, start:end]
                window_grid = grid[start:end]
                window_probs = window_probs / (window_probs.sum() + 1e-8)

                angle = torch.sum(window_probs * window_grid)
                continuous_angles[b, k] = angle
                last_valid_angle = angle
            else:
                if len(top_peaks) > 0:
                    continuous_angles[b, k] = last_valid_angle
                else:
                    continuous_angles[b, k] = 0.0  # 极端情况全部失效才给 0

    return continuous_angles


def calc_rmse(pred_angles, true_angles):
    """计算 K=3 的排列不变性 RMSE"""
    pred_sorted, _ = torch.sort(pred_angles, dim=1)
    true_sorted, _ = torch.sort(true_angles, dim=1)
    mse = torch.mean((pred_sorted - true_sorted) ** 2)
    return mse.item()


def get_continuous_angle_k7(outputs, K=7, radius=2, peak_thresh=1e-5, min_peak_sep_bins=3):
    device = outputs.device
    B = outputs.shape[0]
    probs = torch.sigmoid(outputs)

    probs_unsq = probs.unsqueeze(1)
    pooled = torch.nn.functional.max_pool1d(probs_unsq, kernel_size=5, stride=1, padding=2)
    peak_mask = (probs_unsq == pooled).squeeze(1)
    peak_probs = probs * peak_mask.float()

    grid = torch.linspace(-90, 90, 181, device=device)
    continuous_angles = torch.zeros(B, K, device=device)

    for b in range(B):
        candidate_idx = torch.nonzero(peak_probs[b] > peak_thresh).squeeze(-1)

        if len(candidate_idx) > 0:
            candidate_vals = peak_probs[b, candidate_idx]
            _, order = torch.sort(candidate_vals, descending=True)
            candidate_idx = candidate_idx[order]
        else:
            candidate_idx = torch.tensor([], dtype=torch.long, device=device)

        selected = []

        for idx in candidate_idx:
            idx_int = idx.item()
            if all(abs(idx_int - s) >= min_peak_sep_bins for s in selected):
                selected.append(idx_int)
            if len(selected) == K:
                break

        if len(selected) < K:
            global_order = torch.argsort(probs[b], descending=True)
            for idx in global_order:
                idx_int = idx.item()
                if all(abs(idx_int - s) >= min_peak_sep_bins for s in selected):
                    selected.append(idx_int)
                if len(selected) == K:
                    break

        if len(selected) == 0:
            selected = [90] * K
        elif len(selected) < K:
            selected = selected + [selected[-1]] * (K - len(selected))

        for k in range(K):
            idx = selected[k]
            start = max(0, idx - radius)
            end = min(181, idx + radius + 1)

            window_probs = probs[b, start:end]
            window_grid = grid[start:end]
            window_probs = window_probs / (window_probs.sum() + 1e-8)

            angle = torch.sum(window_probs * window_grid)
            continuous_angles[b, k] = angle

    continuous_angles, _ = torch.sort(continuous_angles, dim=1)
    return continuous_angles
