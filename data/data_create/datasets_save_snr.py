import sys
import os
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[3]
ext_lib = root / "Graduation" / "external" / "DOA_est_Master-master"

if str(ext_lib) not in sys.path:
    sys.path.insert(0, str(ext_lib))
    sys.path.insert(1, str(root))
if 'utils' in sys.modules:
    del sys.modules['utils']

from signal_datasets90 import ULA_dataset
from Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from Create_classic_test_dataset import Create_monte_carlo_theta

if __name__ == '__main__':
    # ================= 核心参数配置 =================
    snap = 50
    snrs = [-20, -15, -10, -5, 0, 5, 10]
    # snrs = [-10, -5, 0, 5, 10, 15, 20]
    k = 3
    M = 8
    signal_range = [-90, 90]
    grid_size = 1
    rho = 1.0
    theta = 5

    dir = fr"D:\Python\Project\doa_estimation\Graduation\data\ViT\ViT_M_{M}_k_{k}\M_{M}_k_{k}_rho{rho}_theta{theta}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    # ================= 1. random_input 测试集 =================
    print(">>> 正在生成 random_input 测试集...")
    test_theta_set = Create_random_k_input_theta(k, signal_range[0], signal_range[1], 5000, min_delta_theta=theta)
    for snr in snrs:
        test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
        Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
        test_dataset.cx_t = []
        test_dataset.save_all_data(os.path.join(dir, f"test_random_input_snr_{snr}"))
        print(f"  - random_input SNR={snr:3d} dB 生成完毕")

    # ================= 2. monte_carlo 测试集  =================
    # test_theta_set = Create_monte_carlo_theta([np.array([10, 20, 30])], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
    #     test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[10,20,30]_snr_{snr}"))
    #
    # test_theta_set = Create_monte_carlo_theta([np.array([10, 13, 16])], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
    #     test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[10,13,16]_snr_{snr}"))

    # # (针对 K=1 修改)
    # print("\n>>> 正在生成 monte_carlo 测试集 (固定角度: 10°)...")
    # test_theta_set_10 = Create_monte_carlo_theta([np.array([10.0])], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set_10, 100, snap, snr, snr_set=0)
    #     test_dataset.cx_t = []
    #     test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[10]_snr_{snr}"))
    #     print(f"  - monte_carlo[10°] SNR={snr:3d} dB 生成完毕")
    #
    # print("\n>>> 正在生成 monte_carlo 测试集 (固定角度: 30°)...")
    # test_theta_set_30 = Create_monte_carlo_theta([np.array([30.0])], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set_30, 100, snap, snr, snr_set=0)
    #     test_dataset.cx_t = []
    #     test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[30]_snr_{snr}"))
    #     print(f"  - monte_carlo[30°] SNR={snr:3d} dB 生成完毕")

    print(f"\n 所有测试数据生成完毕！文件保存在:\n{dir}")