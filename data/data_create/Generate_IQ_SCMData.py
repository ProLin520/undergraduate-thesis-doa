import os
import numpy as np
from tqdm import tqdm
import itertools
from data.data_create.Generate_IQ_Data import generate_iq_sample


def generate_scm_ideal_gaussian_for_vit(base_dir):
    """
    专门为 ViT 生成最纯净的源域数据 (理想高斯 + 理想阵列)
    mode='stage1': 高信噪比 (0 to 25dB)，用于启蒙
    mode='stage2': 全信噪比 (-25 to 25dB)，用于抗干扰训练
    """
    save_dir = os.path.join(base_dir, "SCM_Ideal_Single_Source_Stage2")
    snrs = np.arange(-25, 26, 5)

    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T, M = 1024, 8
    angles = np.arange(-90, 91)

    print(f"SNR 范围: {snrs} dB | 保存路径: {save_dir}")

    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        vit_data_list, label_list = [], []

        for snr in snrs:
            for angle in tqdm(angles, desc=f"[{split_name}] SNR={snr}dB"):
                for _ in range(50):
                    # 1. 纯净复高斯信号
                    S = (np.random.randn(1, T) + 1j * np.random.randn(1, T)) / np.sqrt(2)
                    A = np.exp(-1j * np.pi * np.arange(M)[:, None] * np.sin(np.deg2rad(angle)))
                    X = A @ S

                    # 加噪并计算协方差矩阵 R
                    sig_p = np.mean(np.abs(X) ** 2)
                    noise = np.sqrt((sig_p / (10 ** (snr / 10))) / 2) * (
                            np.random.randn(M, T) + 1j * np.random.randn(M, T))
                    R = ((X + noise) @ (X + noise).conj().T) / T

                    # 2. 构造 ViT 格式 (2, 8, 8)
                    X_vit = np.zeros((2, M, M), dtype=np.float32)
                    X_vit[0, :, :] = np.real(R)
                    X_vit[1, :, :] = np.imag(R)

                    # 归一化处理
                    max_vit = np.max(np.abs(X_vit))
                    if max_vit > 0: X_vit /= max_vit

                    # 3. 181 维 One-hot 标签
                    label = np.zeros(181, dtype=np.float32)
                    label[angle + 90] = 1.0

                    vit_data_list.append(X_vit)
                    label_list.append(label)

        np.save(os.path.join(save_dir, split_name, f"vit_{split_name.lower()}_data.npy"), np.array(vit_data_list))
        np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels.npy"), np.array(label_list))


def generate_scm_dataset_single_source(base_dir, rho=0.0):
    """
    专为 CNN 和 ViT 生成协方差矩阵(SCM)格式的数据
    底层调用与 IQ-ResNet 完全相同的 generate_iq_sample，保证“控制变量”绝对公平
    """
    folder_name = f"SCM_Single_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    snrs = np.arange(-25, 26, 5)
    T, M = 1024, 8
    angles = np.arange(-90, 91)

    print(f"\n>>> 正在生成 BPSK SCM 离线数据集 | Rho = {rho} | 保存至 {folder_name} ...")

    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        samples_count = 10 if is_val else 20
        cnn_data_list, vit_data_list, label_list = [], [], []

        for snr in snrs:
            for angle in tqdm(angles, desc=f"[{split_name}] SNR={snr}dB"):
                for _ in range(samples_count):
                    iq_mat = generate_iq_sample([angle], T, snr, rho=rho)

                    X_complex = iq_mat[:M, :] + 1j * iq_mat[M:, :]
                    R = (X_complex @ X_complex.conj().T) / T

                    # ====== 构造 CNN 格式 (8, 8, 3) ======
                    X_cnn = np.zeros((M, M, 3), dtype=np.float32)
                    X_cnn[:, :, 0] = np.real(R)
                    X_cnn[:, :, 1] = np.imag(R)
                    X_cnn[:, :, 2] = np.angle(R) / np.pi
                    max_val = np.max(np.abs(R))
                    if max_val > 1e-8:
                        X_cnn[:, :, 0] /= max_val
                        X_cnn[:, :, 1] /= max_val

                    # ====== 构造 ViT 格式 (2, 8, 8) ======
                    X_vit = np.zeros((2, M, M), dtype=np.float32)
                    X_vit[0, :, :] = np.real(R)
                    X_vit[1, :, :] = np.imag(R)
                    max_vit = np.max(np.abs(X_vit))
                    if max_vit > 1e-8:
                        X_vit /= max_vit

                    label = np.zeros(181, dtype=np.float32)
                    label[angle + 90] = 1.0

                    cnn_data_list.append(X_cnn)
                    vit_data_list.append(X_vit)
                    label_list.append(label)

        # 统一保存
        print(f"正在写入 {split_name} 数据...")
        np.save(os.path.join(save_dir, split_name, f"cnn_{split_name.lower()}_data.npy"),
                np.array(cnn_data_list, dtype=np.float32))
        np.save(os.path.join(save_dir, split_name, f"vit_{split_name.lower()}_data.npy"),
                np.array(vit_data_list, dtype=np.float32))
        np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels.npy"),
                np.array(label_list, dtype=np.float32))


def generate_scm_dataset_two_source(base_dir, rho=0.0):
    """
    专为 ViT 和 CNN 生成协方差矩阵(SCM)格式的双信源数据
    """
    folder_name = f"SCM_Two_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    snrs = np.arange(-25, 26, 5)
    T, M = 512, 8
    all_combinations = list(itertools.combinations(np.arange(-90, 91), 2))

    print(f"\n>>> 正在生成 双信源 SCM 离线数据集 | Rho={rho} | 保存至 {save_dir} ...")

    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"

        for snr in snrs:
            cnn_data_list, vit_data_list, label_list = [], [], []

            for doas in tqdm(all_combinations, desc=f"[{split_name}] SNR={snr}dB"):
                iq_mat = generate_iq_sample(list(doas), T, snr, rho=rho)

                # ... (中间转换 SCM、CNN、ViT 的代码与原来完全相同) ...
                X_complex = iq_mat[:M, :] + 1j * iq_mat[M:, :]
                R = (X_complex @ X_complex.conj().T) / T

                X_cnn = np.zeros((M, M, 3), dtype=np.float32)
                X_cnn[:, :, 0] = np.real(R)
                X_cnn[:, :, 1] = np.imag(R)
                X_cnn[:, :, 2] = np.angle(R) / np.pi
                max_val = np.max(np.abs(R))
                if max_val > 1e-8:
                    X_cnn[:, :, 0] /= max_val
                    X_cnn[:, :, 1] /= max_val

                X_vit = np.zeros((2, M, M), dtype=np.float32)
                X_vit[0, :, :] = np.real(R)
                X_vit[1, :, :] = np.imag(R)
                max_vit = np.max(np.abs(X_vit))
                if max_vit > 1e-8:
                    X_vit /= max_vit

                label = np.zeros(181, dtype=np.float32)
                label[doas[0] + 90] = 1.0
                label[doas[1] + 90] = 1.0

                cnn_data_list.append(X_cnn)
                vit_data_list.append(X_vit)
                label_list.append(label)

            np.save(os.path.join(save_dir, split_name, f"cnn_{split_name.lower()}_data_snr{snr}.npy"), np.array(cnn_data_list, dtype=np.float32))
            np.save(os.path.join(save_dir, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy"), np.array(vit_data_list, dtype=np.float32))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"), np.array(label_list, dtype=np.float32))


def generate_scm_dataset_seven_source(base_dir, rho=0.0):
    """
    七源场景 SCM 数据生成: 专为 CNN 和 ViT 提取空间相关矩阵
    底层信号依然调用 generate_iq_sample，保证对比绝对公平。
    """
    folder_name = f"SCM_Seven_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T, M = 1024, 8
    snrs = [0, 5, 10, 15, 20]

    num_train_samples_per_snr = 8000
    num_val_samples_per_snr = 1000

    print(f"\n>>> 正在生成七信号源 SCM 数据集 | Rho={rho} | (采用最小间隔全局随机策略)...")

    def get_valid_7_doas(min_sep=12):
        """核心黑科技进阶版：强制边缘采样 + 最小间隔约束"""
        # 定义危险的边缘区和舒适的中间区
        edge_pool = list(range(-90, -65)) + list(range(65, 91))
        mid_pool = list(range(-65, 65))

        while True:
            # 策略：强行塞入 2 个边缘角度，5 个中间角度
            # (如果觉得 2 个不够狠，可以改成 3 个边缘，4 个中间)
            edges = np.random.choice(edge_pool, 2, replace=False)
            mids = np.random.choice(mid_pool, 5, replace=False)

            angles = np.concatenate([edges, mids])
            angles.sort()

            # 依然保留你的防拥挤约束
            if np.all(np.diff(angles) >= min_sep):
                return angles.tolist()

    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        samples_count = num_val_samples_per_snr if is_val else num_train_samples_per_snr

        for snr in snrs:
            cnn_data_list, vit_data_list, label_list = [], [], []

            for _ in tqdm(range(samples_count), desc=f"[{split_name}] SNR={snr}dB"):
                all_doas = get_valid_7_doas(min_sep=12)
                iq_mat = generate_iq_sample(all_doas, T, snr, rho=rho)

                X_complex = iq_mat[:M, :] + 1j * iq_mat[M:, :]
                R = (X_complex @ X_complex.conj().T) / T

                # ====== 构造 CNN 格式 (8, 8, 3) ======
                X_cnn = np.zeros((M, M, 3), dtype=np.float32)
                X_cnn[:, :, 0], X_cnn[:, :, 1], X_cnn[:, :, 2] = np.real(R), np.imag(R), np.angle(R) / np.pi
                max_val = np.max(np.abs(R))
                if max_val > 1e-8:
                    X_cnn[:, :, 0] /= max_val
                    X_cnn[:, :, 1] /= max_val

                # ====== 构造 ViT 格式 (2, 8, 8) ======
                X_vit = np.zeros((2, M, M), dtype=np.float32)
                X_vit[0, :, :], X_vit[1, :, :] = np.real(R), np.imag(R)
                max_vit = np.max(np.abs(X_vit))
                if max_vit > 1e-8:
                    X_vit /= max_vit

                # Multi-hot 标签
                label_cls = np.zeros(181, dtype=np.uint8)
                for doa in all_doas:
                    label_cls[doa + 90] = 1

                cnn_data_list.append(X_cnn)
                vit_data_list.append(X_vit)
                label_list.append(label_cls)

            # SCM 数据通常较小，但依然使用 float16 极致压缩
            np.save(os.path.join(save_dir, split_name, f"cnn_{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(cnn_data_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(vit_data_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"),
                    np.array(label_list, dtype=np.uint8))


def generate_scm_dataset_seven_source_article(base_dir, rho=0.0):
    """
    终极版：严格按照论文 Fig.11 场景的分区分布生成数据。
    专为分类网络修正：生成 181 维 Multi-hot 标签。
    一键同时生成：IQ (给 ResNet)、SCM (给 CNN/ViT) 的全部数据格式。
    """
    folder_name = f"SCM_Seven_Source_Article_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T, M = 1024, 8
    snrs = [0, 5, 10, 15, 20]

    # 后 2 个信号：在 10 到 90 度中任意挑 2 个组合 (C_81^2 = 3240 种)
    comb_2_sources = list(itertools.combinations(np.arange(10, 91), 2))

    # 前 5 个信号：在 5 个特定的负数及小角度扇区中随机采样
    pools = [
        np.arange(-90, -70), np.arange(-70, -50),
        np.arange(-50, -30), np.arange(-30, -10),
        np.arange(-10, 10)
    ]

    print(f"\n>>> 正在生成论文原版特化七信源数据集 (包含 IQ、SCM 与 181 维标签) | Rho={rho} ...")

    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        samples_per_comb = 3 if is_val else 10  # 验证集每种组合3个，训练集10个

        for snr in snrs:
            iq_list, cnn_list, vit_list, label_list = [], [], [], []

            for base_doas in tqdm(comb_2_sources, desc=f"[{split_name}] SNR={snr}dB"):
                for _ in range(samples_per_comb):
                    other_doas = [np.random.choice(pool) for pool in pools]
                    all_doas = list(base_doas) + other_doas

                    # ====== 1. 生成底层 IQ 数据 ======
                    iq_mat = generate_iq_sample(all_doas, T, snr, rho=rho)

                    # ====== 2. 生成 SCM 协方差数据 ======
                    X_complex = iq_mat[:M, :] + 1j * iq_mat[M:, :]
                    R = (X_complex @ X_complex.conj().T) / T

                    # CNN 格式 (8, 8, 3)
                    X_cnn = np.zeros((M, M, 3), dtype=np.float32)
                    X_cnn[:, :, 0], X_cnn[:, :, 1], X_cnn[:, :, 2] = np.real(R), np.imag(R), np.angle(R) / np.pi
                    max_val = np.max(np.abs(R))
                    if max_val > 1e-8:
                        X_cnn[:, :, 0] /= max_val
                        X_cnn[:, :, 1] /= max_val

                    # ViT 格式 (2, 8, 8)
                    X_vit = np.zeros((2, M, M), dtype=np.float32)
                    X_vit[0, :, :], X_vit[1, :, :] = np.real(R), np.imag(R)
                    max_vit = np.max(np.abs(X_vit))
                    if max_vit > 1e-8:
                        X_vit /= max_vit

                    # ====== 3. 修复核心问题：生成 181 维 Multi-hot 标签 ======
                    label_cls = np.zeros(181, dtype=np.uint8)
                    for doa in all_doas:
                        label_cls[doa + 90] = 1

                    iq_list.append(iq_mat.astype(np.float16))
                    cnn_list.append(X_cnn)
                    vit_list.append(X_vit)
                    label_list.append(label_cls)

            # 统一保存 (加上前缀区分格式，避免文件混乱)
            np.save(os.path.join(save_dir, split_name, f"iq_{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(iq_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"cnn_{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(cnn_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"vit_{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(vit_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"),
                    np.array(label_list, dtype=np.uint8))


if __name__ == "__main__":
    BASE_SAVE_PATH = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data"
    BASE_SAVE_PATH1 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Single_Source"
    BASE_SAVE_PATH2 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Two_Source"
    BASE_SAVE_PATH3 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Seven_Source"
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    # generate_scm_ideal_gaussian_for_vit(BASE_SAVE_PATH1)
    # generate_scm_dataset_single_source(BASE_SAVE_PATH1, rho=0.0)
    # generate_scm_dataset_single_source(BASE_SAVE_PATH1, rho=1.0)
    # generate_scm_dataset_two_source(BASE_SAVE_PATH2, rho=0.0)
    # generate_scm_dataset_two_source(BASE_SAVE_PATH2, rho=1.0)
    # generate_scm_dataset_seven_source(BASE_SAVE_PATH3, rho=0.0)
    generate_scm_dataset_seven_source_article(BASE_SAVE_PATH3)

    print("\n 所有数据集生成完毕！")

