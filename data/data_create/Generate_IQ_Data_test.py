import os
import numpy as np
from tqdm import tqdm
import itertools
from data.data_create.Generate_IQ_Data import generate_iq_sample

def generate_test_single_fig5(base_dir, rho=0.0):
    """
    生成文献 Fig.5 单信源方法对比测试集。
    设置：SNR=0dB，T=1024，DOA=-90°~90°，每个DOA生成100个样本。
    """
    save_dir = os.path.join(base_dir, f"Test_Rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    T = 1024
    snr = 0
    angles = np.arange(-90, 91)
    samples_per_angle = 100
    num_samples = len(angles) * samples_per_angle

    data = np.empty((num_samples, 16, T), dtype=np.float16)
    labels = np.empty(num_samples, dtype=np.int16)
    onehot = np.zeros((num_samples, 181), dtype=np.uint8)

    print(f"\n>>> 正在生成文献 Fig.5 单信源测试集 | SNR={snr}dB | T={T} | 每角度{samples_per_angle}样本 | Rho={rho}...")
    idx = 0
    for angle in tqdm(angles, desc="Generating Fig.5 Single-Source Test"):
        for _ in range(samples_per_angle):
            data[idx] = generate_iq_sample([int(angle)], T, snr, rho=rho).astype(np.float16)
            labels[idx] = angle
            onehot[idx, angle + 90] = 1
            idx += 1

    np.save(os.path.join(save_dir, "fig5_iq_data_snr0.npy"), data)
    np.save(os.path.join(save_dir, "fig5_true_angles_snr0.npy"), labels)
    np.save(os.path.join(save_dir, "fig5_onehot_labels_snr0.npy"), onehot)
    print(f">>> Fig.5 单信源测试集生成完毕，共 {num_samples} 个样本，已保存至: {save_dir}")


def generate_test_single(base_dir, target_snr=0, samples_per_angle=50, rho=0.0):
    """专门生成用于测试评估的数据集"""
    folder_name = f"Test_4000_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    T = 4000
    angles = np.arange(-90, 91)
    data_list = []
    label_list = []

    print(f"正在生成专门的测试集: SNR = {target_snr} dB, Rho = {rho}...")
    for angle in tqdm(angles):
        for _ in range(samples_per_angle):
            iq_mat = generate_iq_sample([angle], T, target_snr, rho=rho)
            data_list.append(iq_mat.astype(np.float16))
            label_list.append(angle)

    np.save(os.path.join(save_dir, f"test_data_snr{target_snr}.npy"), np.array(data_list))
    np.save(os.path.join(save_dir, f"test_labels_snr{target_snr}.npy"), np.array(label_list))


def generate_test_two_sep5(base_dir, rho=0.0):
    """生成用于复现 Fig.8 的测试集: 间距 5度, SNR=0dB, 每个角度100个样本"""
    save_dir = os.path.join(base_dir, f"Test_5_deg_Rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    T = 512  # 双信源的快拍数为 512
    snr = 5
    samples_per_angle = 100

    # 角度设置: 源1 (-90 to 85), 源2 (-85 to 90)
    angles_source1 = np.arange(-90, 86)
    data_list = []
    label_list = []

    print(f">>> 正在生成 Fig 8 测试集 (SNR={snr}dB, angle_diff=5)...")
    for a1 in tqdm(angles_source1):
        a2 = a1 + 5
        for _ in range(samples_per_angle):
            iq_mat = generate_iq_sample([a1, a2], T, snr, rho=rho)
            data_list.append(iq_mat.astype(np.float16))
            # 存下真实的两个角度
            label_list.append([a1, a2])

    np.save(os.path.join(save_dir, f"test_data_5_deg.npy"), np.array(data_list))
    np.save(os.path.join(save_dir, f"test_labels_5_deg.npy"), np.array(label_list))


def generate_test_two_sep4_snr4(base_dir, rho=0.0):
    """
    生成 Fig 9a 的专属离线测试集: 间距 4度, SNR 从 -25 到 25 (步长 4dB)
    """
    save_dir = os.path.join(base_dir, f"Test_sep4_snr4_Rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    T = 512
    snrs = np.arange(-25, 25, 4)  # -25, -23, ..., 25
    samples_per_angle = 100
    angles_source1 = np.arange(-90, 87)  # 源 1: -90 到 86

    print(f"\n>>> 正在生成 Fig 9a 离线测试集 (间距 4°)...")
    for snr in snrs:
        iq_list, label_list = [], []
        print(f"正在生成 SNR = {snr} dB ...")

        for a1 in tqdm(angles_source1, leave=False):
            a2 = a1 + 4
            for _ in range(samples_per_angle):
                # 生成底层 IQ 数据
                iq_mat = generate_iq_sample([a1, a2], T, snr, rho=rho)
                iq_list.append(iq_mat.astype(np.float32))
                # 存下真实的两个连续角度
                label_list.append([a1, a2])

        # 按 SNR 分开保存，防止内存溢出
        np.save(os.path.join(save_dir, f"test_iq_data_snr{snr}.npy"), np.array(iq_list, dtype=np.float32))
        np.save(os.path.join(save_dir, f"test_labels_snr{snr}.npy"), np.array(label_list, dtype=np.float32))

    print(f"离线测试集 (Fig 9a) 生成完毕！保存在 {save_dir}")


def generate_test_two_sep(base_dir, rho=1.0):
    """
    生成 Fig 9b 离线测试集: 固定 SNR=5dB, 阵列误差 rho=1.0
    源1固定在 0度, 源2在 0 + delta_theta
    delta_theta = [1, 2, 5, 8, 10, 15, 20]
    """
    save_dir = os.path.join(base_dir, f"Test_sep_Rho{rho}")
    os.makedirs(save_dir, exist_ok=True)

    T = 512
    snr = 5
    delta_thetas = [1, 2, 5, 8, 10, 15, 20]
    samples_per_scenario = 1000

    print(f"\n>>> 正在生成 Fig 9b 离线测试集 (Rho={rho}, SNR={snr}dB)...")
    for dt in delta_thetas:
        iq_list, label_list = [], []
        a1, a2 = 0, dt  # 第一个源 0度，第二个源 dt度

        for _ in tqdm(range(samples_per_scenario), desc=f"Delta = {dt}°"):
            # 🌟 注意这里传入了 rho=1.0 模拟阵列误差
            iq_mat = generate_iq_sample([a1, a2], T, snr, rho=rho)
            iq_list.append(iq_mat.astype(np.float32))
            label_list.append([a1, a2])

        np.save(os.path.join(save_dir, f"test_iq_data_delta{dt}.npy"), np.array(iq_list, dtype=np.float32))
        np.save(os.path.join(save_dir, f"test_labels_delta{dt}.npy"), np.array(label_list, dtype=np.float32))

    print(f"离线测试集 (Fig 9b) 生成完毕！保存在 {save_dir}")


def generate_seven_source_test(base_dir, rho=0.0):
    folder_name = f"Seven_Source_Article_test_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    T = 1024
    snr = 10
    scan_angles = np.arange(10, 81)

    print(f"\n>>> 正在生成 Fig. 11 专属扫角测试集 | SNR={snr}dB | Rho={rho} ...")

    data_list = []
    label_list = []

    for ang6 in tqdm(scan_angles, desc="Generating Fig 11 Test_Rho0.0 Data"):
        ang7 = ang6 + 10
        # 严格按照文献设定的 7 个角度
        true_doas = [-80, -60, -40, -20, 0, ang6, ang7]

        # 生成底层 IQ 矩阵
        iq_mat = generate_iq_sample(true_doas, T, snr, rho=rho)

        # 极致压缩存储
        data_list.append(iq_mat.astype(np.float16))
        # 直接存储真实角度值，方便画图时作为 Baseline
        label_list.append(np.array(true_doas, dtype=np.int16))

    # 保存为 numpy 文件
    np.save(os.path.join(save_dir, "sector_iq_data.npy"), np.array(data_list, dtype=np.float16))
    np.save(os.path.join(save_dir, "sector_true_angles.npy"), np.array(label_list, dtype=np.int16))


def generate_seven_source_random_test(base_dir, rho=0.0):
    """
    终极泛化测试集：全局随机 + 最小间隔约束。
    用于测试“特化模型”在遇到未见过的随机空间分布时，是否会崩溃。
    """
    folder_name = f"Seven_Source_Random_test_Rho{rho}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    T = 1024
    snr = 10  # 依然采用 10dB 进行极限压测
    num_samples = 1000  # 1000 个样本足够画出密集的散点图

    print(f"\n>>> 正在生成 全局随机泛化 测试集 | SNR={snr}dB | Rho={rho} ...")

    def get_valid_7_doas(min_sep=12):
        while True:
            angles = np.random.choice(np.arange(-90, 91), 7, replace=False)
            angles.sort()
            if np.all(np.diff(angles) >= min_sep):
                return angles.tolist()

    data_list = []
    label_list = []

    for _ in tqdm(range(num_samples), desc="Generating Random Test_Rho0.0 Data"):
        # 生成 7 个合法的全局随机角度
        true_doas = get_valid_7_doas(min_sep=12)

        # 生成底层 IQ 矩阵
        iq_mat = generate_iq_sample(true_doas, T, snr, rho=rho)

        data_list.append(iq_mat.astype(np.float16))
        label_list.append(np.array(true_doas, dtype=np.int16))

    np.save(os.path.join(save_dir, "random_iq_data.npy"), np.array(data_list, dtype=np.float16))
    np.save(os.path.join(save_dir, "random_true_angles.npy"), np.array(label_list, dtype=np.int16))

    print(f">>> 全局随机测试集生成完毕，已保存至: {save_dir}")


if __name__ == "__main__":
    BASE_SAVE_PATH = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data"
    BASE_SAVE_PATH1 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Single_Source"
    BASE_SAVE_PATH2 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Two_Source"
    BASE_SAVE_PATH3 = r"D:\Python\Project\doa_estimation\Graduation\data\IQ_Data\Seven_Source"
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    # generate_test_single_fig5(BASE_SAVE_PATH1, rho=0.0)
    # snrs_to_test = [-10, -5, 0, 5, 10, 15, 20]
    # for snr in snrs_to_test:
    #     generate_test_single(BASE_SAVE_PATH1, target_snr=snr, samples_per_angle=50, rho=1.0)
    # generate_test_single(BASE_SAVE_PATH1, target_snr=0, samples_per_angle=50, rho=1.0)

    # generate_test_two_sep5(BASE_SAVE_PATH2, rho=0.0)
    # generate_test_two_sep4_snr4(BASE_SAVE_PATH2, rho=0.0)
    # generate_test_two_sep(BASE_SAVE_PATH2, rho=1.0)
    generate_seven_source_test(BASE_SAVE_PATH3, rho=0.0)
    generate_seven_source_random_test(BASE_SAVE_PATH3, rho=0.0)

    print("数据集生成完毕！")
