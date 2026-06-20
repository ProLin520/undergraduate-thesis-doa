import os
import numpy as np
from scipy.signal import upfirdn
import itertools
import scipy.linalg
from tqdm import tqdm


def rcosfilter(N, alpha, Ts, Fs):
    """生成升余弦滤波器抽头系数"""
    T_delta = 1 / Fs
    t = np.arange(-N, N + 1) * T_delta
    # 避免分母为 0 引发警告
    t[t == 0] = 1e-8
    t[np.abs(t) == Ts / (2 * alpha)] += 1e-8

    h = np.sinc(t / Ts) * np.cos(np.pi * alpha * t / Ts) / (1 - (2 * alpha * t / Ts) ** 2)
    return h / np.sqrt(np.sum(h ** 2))


def generate_iq_sample(doas, T, snr_db, M=8, rho=0.0):
    """
    根据论文生成单个样本的 16xT IQ矩阵
    """
    num_sources = len(doas)
    oversampling = 8
    roll_off = 0.5
    span = 6

    # BPSK 调制与升余弦滤波
    num_symbols = (T // oversampling) + span * 2 + 5
    h_rc = rcosfilter(span * oversampling, roll_off, 1, oversampling)

    signals = []
    for _ in range(num_sources):
        bits = np.random.randint(0, 2, num_symbols)
        bpsk_syms = 2 * bits - 1  # 映射到 {-1, 1}
        shaped_sig = upfirdn(h_rc, bpsk_syms, up=oversampling)

        # 截取 T 个采样点 (去除瞬态响应)
        start_idx = span * oversampling
        sig_T = shaped_sig[start_idx: start_idx + T]
        signals.append(sig_T)

    signals = np.array(signals)  # Shape: (L, T)

    array_pos = np.arange(M) * 0.5  # 假设 d = 0.5 lambda
    pos_para = np.zeros(M)

    if rho > 0:
        amp_bias = np.array([0.0, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2][:M])
        phase_bias = np.array([0.0, -30, -30, -30, -30, 30, 30, 30][:M])
        pos_bias = np.array([0.0, -1, -1, -1, -1, 1, 1, 1][:M]) * 0.2

        amp_coef = rho * amp_bias
        phase_coef = rho * phase_bias
        pos_para = rho * pos_bias * 0.5  # d = 0.5

        mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
        MC_coef = mc_para ** np.arange(M)
        MC_mtx = scipy.linalg.toeplitz(MC_coef, MC_coef)

        AP_coef = (1 + amp_coef) * np.exp(1j * phase_coef / 180 * np.pi)
        AP_mtx = np.diag(AP_coef)
    else:
        MC_mtx = np.eye(M, dtype=complex)
        AP_mtx = np.eye(M, dtype=complex)

    # 构建带位置误差的流型矩阵
    doas_rad = np.deg2rad(doas)
    steer_vector = -1j * 2 * np.pi * (array_pos + pos_para)
    horizon_vec = np.sin(doas_rad)
    A = np.exp(steer_vector[:, np.newaxis] * horizon_vec[np.newaxis, :])

    # 乘以互耦合和幅相误差
    A = MC_mtx @ AP_mtx @ A

    # 接收信号 X = A * S
    X = A @ signals  # Shape: (M, T)

    # 添加噪声
    signal_power = np.mean(np.abs(X) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(M, T) + 1j * np.random.randn(M, T))

    X_noisy = X + noise

    # 功率归一化 (论文: After power normalization of the received signal...)
    X_normalized = X_noisy / np.sqrt(np.mean(np.abs(X_noisy) ** 2))

    # 提取 I 和 Q 分量并拼接 (最终维度 16xT)
    I_comp = np.real(X_normalized)
    Q_comp = np.imag(X_normalized)
    IQ_matrix = np.vstack((I_comp, Q_comp))

    return IQ_matrix


def generate_single_source(base_dir, rho=0.0):
    """单源场景: T=1024, 分类任务"""
    folder_name = f"Single_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T = 1024
    snrs = np.arange(-25, 26, 5)
    angles = np.arange(-90, 91)

    print(f">>> 正在生成单信号源数据集 (Single Source rho={rho})...")
    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        samples_count = 10 if is_val else 20
        data_list, label_list = [], []

        for snr in snrs:
            for angle in tqdm(angles, desc=f"[{split_name}] SNR={snr}dB"):
                for _ in range(samples_count):
                    iq_mat = generate_iq_sample([angle], T, snr, rho=rho)
                    label = np.zeros(181, dtype=np.float32)
                    label[angle + 90] = 1.0  # One-hot 编码

                    data_list.append(iq_mat.astype(np.float16))
                    label_list.append(label.astype(np.float16))

        np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_data.npy"),
                np.array(data_list))
        np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels.npy"),
                np.array(label_list))


def generate_two_source(base_dir, rho=0.0):
    folder_name = f"Two_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T = 512
    snrs = np.arange(-25, 26, 5)
    all_combinations = list(itertools.combinations(np.arange(-90, 91), 2))

    print(f"\n>>> 正在生成双信号源数据集 (Two Source rho={rho})...")
    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"

        for snr in snrs:
            data_list, label_list = [], []
            for doas in tqdm(all_combinations, desc=f"[{split_name}] SNR={snr}dB"):
                iq_mat = generate_iq_sample(list(doas), T, snr, rho=rho)

                label = np.zeros(181, dtype=np.uint8)
                label[doas[0] + 90] = 1
                label[doas[1] + 90] = 1

                data_list.append(iq_mat.astype(np.float16))
                label_list.append(label)

            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_data_snr{snr}.npy"),
                np.array(data_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"),
                np.array(label_list, dtype=np.uint8))



def generate_seven_source(base_dir, rho=0.0):
    """
    七源场景 IQ 数据生成: 专为 IQ-ResNet 提供底层时域数据
    采用全局随机 + 最小间隔约束，打破死板扇区。
    """
    folder_name = f"Seven_Source_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T = 1024
    snrs = [0, 5, 10, 15, 20] # 按照原论文设置

    num_train_samples_per_snr = 8000
    num_val_samples_per_snr = 1000

    print(f"\n>>> 正在生成七信号源 IQ 数据集 | Rho={rho} | (采用最小间隔全局随机策略)...")

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
            data_list, label_list = [], []
            for _ in tqdm(range(samples_count), desc=f"[{split_name}] SNR={snr}dB"):
                # 获取合法的 7 个随机角度（最小间隔 12 度）
                all_doas = get_valid_7_doas(min_sep=12)

                # 生成底层 IQ 矩阵
                iq_mat = generate_iq_sample(all_doas, T, snr, rho=rho)

                # 生成 Multi-hot 标签 (181 维，有信号的位置设为 1)
                label_cls = np.zeros(181, dtype=np.uint8)
                for doa in all_doas:
                    label_cls[doa + 90] = 1

                data_list.append(iq_mat.astype(np.float16))
                label_list.append(label_cls)

            # 极致压缩存储
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(data_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"),
                    np.array(label_list, dtype=np.uint8))


def generate_seven_source_article(base_dir, rho=0.0):
    """七源场景: T=1024, 回归任务"""
    save_dir = os.path.join(base_dir, f"Seven_Source_Article_Rho{rho}")
    os.makedirs(os.path.join(save_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Val"), exist_ok=True)

    T = 1024
    snrs = [0, 5, 10, 15, 20]
    comb_2_sources = list(itertools.combinations(np.arange(10, 91), 2))
    pools = [
        np.arange(-90, -70), np.arange(-70, -50),
        np.arange(-50, -30), np.arange(-30, -10),
        np.arange(-10, 10)
    ]

    print("\n>>> 正在生成七信号源数据集 (Seven Source)...")
    for is_val in [False, True]:
        split_name = "Val" if is_val else "Train"
        samples_per_comb = 3 if is_val else 10  # 验证集每种组合生成3个，训练集10个

        for snr in snrs:
            data_list, label_list = [], []
            for base_doas in tqdm(comb_2_sources, desc=f"[{split_name}] SNR={snr}dB"):
                for _ in range(samples_per_comb):
                    other_doas = [np.random.choice(pool) for pool in pools]
                    all_doas = list(base_doas) + other_doas

                    iq_mat = generate_iq_sample(all_doas, T, snr, rho=rho)
                    label = np.array(sorted(all_doas), dtype=np.uint8)

                    data_list.append(iq_mat.astype(np.float16))
                    label_list.append(label)

            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_data_snr{snr}.npy"),
                    np.array(data_list, dtype=np.float16))
            np.save(os.path.join(save_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy"),
                    np.array(label_list, dtype=np.uint8))


if __name__ == "__main__":
    BASE_SAVE_PATH = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data"
    BASE_SAVE_PATH1 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Single_Source"
    BASE_SAVE_PATH2 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Two_Source"
    BASE_SAVE_PATH3 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Seven_Source"
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    # generate_single_source(BASE_SAVE_PATH1, rho=0.0)
    # generate_single_source(BASE_SAVE_PATH1, rho=0.3)
    # generate_single_source(BASE_SAVE_PATH1, rho=1.0)

    # generate_two_source(BASE_SAVE_PATH2, rho=0.0)
    # generate_two_source(BASE_SAVE_PATH2, rho=1.0)

    # generate_seven_source(BASE_SAVE_PATH3, rho=0.0)
    generate_seven_source_article(BASE_SAVE_PATH3, rho=0.0)

    print("数据集生成完毕！")
