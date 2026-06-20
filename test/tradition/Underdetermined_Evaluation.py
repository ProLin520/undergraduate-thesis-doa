import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import doatools.model as model
import doatools.estimation as estimation
from doatools.estimation.coarray import CoarrayACMBuilder1D
current_dir = os.path.dirname(os.path.abspath(__file__))
doatools_path = os.path.join(current_dir, 'doatools.py-master')
if doatools_path not in sys.path:
    sys.path.append(doatools_path)


save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.size'] = 11
wavelength = 1.0
d0 = 0.5
NUM_RX = 8
NUM_SNAPSHOTS = 500
NUM_REPEATS = 100
SNR_RANGE = np.arange(-5, 26, 5)
TEST_K_LIST = [9, 10]

ula = model.UniformLinearArray(NUM_RX, d0)
coprime = model.CoPrimeArray(2, 5, d0, mode='2m')
nested = model.NestedArray(4, 4, d0)

array_configs = {'ULA': ula, 'Coprime': coprime, 'Nested': nested}


# 数据生成函数
def generate_array_data(array, angles_deg, snr_db, num_snapshots):
    K = len(angles_deg)
    A = array.steering_matrix(model.FarField1DSourcePlacement(np.radians(angles_deg)), wavelength)
    S = (np.random.randn(K, num_snapshots) + 1j * np.random.randn(K, num_snapshots)) / np.sqrt(2)
    N = (np.random.randn(array.size, num_snapshots) + 1j * np.random.randn(array.size, num_snapshots)) / np.sqrt(2)
    signal_power = 1.0
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    X = A @ S + N * np.sqrt(noise_power)
    return (X @ X.conj().T) / num_snapshots


# ================= 4. 主循环：计算 MSE =================
grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)
esprit_est = estimation.Esprit1D(wavelength)

# 结构化存储：all_results[K][阵列名_算法名] = [mse_list]
all_results = {K: {} for K in TEST_K_LIST}

for K in TEST_K_LIST:
    print(f"\n>>> 正在计算 {K} 信源场景...")
    np.random.seed(123 + K)
    true_angles = np.sort(np.random.uniform(-50, 50, K))
    true_angles_rad = np.radians(true_angles)

    results = {f'{arr}_{algo}': [] for arr in array_configs.keys() for algo in ['MUSIC', 'ESPRIT']}

    for snr in SNR_RANGE:
        print(f"  SNR = {snr:2d} dB...", end="", flush=True)
        errs = {k: [] for k in results.keys()}

        for _ in range(NUM_REPEATS):
            for arr_name, arr_obj in array_configs.items():
                R_raw = generate_array_data(arr_obj, true_angles, snr, NUM_SNAPSHOTS)
                builder = CoarrayACMBuilder1D(arr_obj)
                R_ss = builder.transform(R_raw, 'ss')
                v_ula = builder.get_virtual_ula()

                # 检查自由度
                if K >= v_ula.size:
                    # 对于 ULA 这种必定失败的，填充较大误差值
                    errs[f'{arr_name}_MUSIC'].append([1.0] * K)
                    errs[f'{arr_name}_ESPRIT'].append([1.0] * K)
                    continue

                music_obj = estimation.MUSIC(v_ula, wavelength, grid)

                # MUSIC 估计
                res_m, est_m = music_obj.estimate(R_ss, K)
                if res_m and len(est_m.locations) == K:
                    errs[f'{arr_name}_MUSIC'].append(np.sort(est_m.locations) - true_angles_rad)
                else:
                    errs[f'{arr_name}_MUSIC'].append([1.0] * K)

                # ESPRIT 估计
                res_e, est_e = esprit_est.estimate(R_ss, K, d0)
                if res_e and len(est_e.locations) == K:
                    errs[f'{arr_name}_ESPRIT'].append(np.sort(est_e.locations) - true_angles_rad)
                else:
                    errs[f'{arr_name}_ESPRIT'].append([1.0] * K)

        for k in results.keys():
            results[k].append(np.mean(np.square(errs[k])))
        print("Done")
    all_results[K] = results


# ================= 5. 绘图逻辑 =================

def plot_group(K_val, include_ula=False):
    plt.figure(figsize=(9, 6))
    data = all_results[K_val]

    # 只有在需要时才画 ULA (黄色)
    if include_ula:
        plt.semilogy(SNR_RANGE, data['ULA_MUSIC'], 'y-x', label='ULA MUSIC')
        plt.semilogy(SNR_RANGE, data['ULA_ESPRIT'], 'y--x', label='ULA ESPRIT')

    # Coprime (红色系)
    plt.semilogy(SNR_RANGE, data['Coprime_MUSIC'], 'r-^', label='Coprime SS-MUSIC')
    plt.semilogy(SNR_RANGE, data['Coprime_ESPRIT'], 'm-^', label='Coprime ESPRIT')

    # Nested (蓝色系)
    plt.semilogy(SNR_RANGE, data['Nested_MUSIC'], 'b-o', label='Nested SS-MUSIC')
    plt.semilogy(SNR_RANGE, data['Nested_ESPRIT'], 'c-o', label='Nested ESPRIT')

    # title_suffix = " (with ULA Control)" if include_ula else " (Sparse Arrays Only)"
    # plt.title(f'Underdetermined DOA Estimation: {K_val} Sources\n{title_suffix}', fontweight='bold')
    # plt.title(f'Underdetermined DOA Estimation: {K_val} Sources', fontweight='bold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE (rad^2)')
    plt.grid(True, which='both', alpha=0.3)

    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()


# 生成第一组：只有嵌套和互质
for K in TEST_K_LIST:
    plot_group(K, include_ula=False)
    plt.savefig(os.path.join(save_dir, f"Underdetermined_{K}_source_Evaluation.png"), dpi=300)

# 生成第二组：包含 ULA 对比 (黄色)
for K in TEST_K_LIST:
    plot_group(K, include_ula=True)
    plt.savefig(os.path.join(save_dir, f"Underdetermined_{K}_source_Evaluation.png"), dpi=300)

plt.show()