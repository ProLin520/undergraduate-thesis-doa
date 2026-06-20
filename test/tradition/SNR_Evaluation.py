import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 确保能找到 doatools 包
current_dir = os.path.dirname(os.path.abspath(__file__))
doatools_path = os.path.join(current_dir, 'doatools.py-master')
if doatools_path not in sys.path:
    sys.path.append(doatools_path)

import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
from Graduation.data.data_create.simulation_generator import generate_ideal_data

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)

# ================= 配置参数 =================
plt.rcParams['font.size'] = 12

wavelength = 1.0
d0 = 0.5
NUM_RX = 8
NUM_SNAPSHOTS = 200
NUM_REPEATS = 100

TRUE_ANGLE = 20.0
TRUE_ANGLE_RAD = np.radians(TRUE_ANGLE)
SNR_RANGE = np.arange(-20, 11, 2)
NUM_SOURCES = 1

ula = model.UniformLinearArray(NUM_RX, d0)
sources = model.FarField1DSourcePlacement([TRUE_ANGLE_RAD])

grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)
music_estimator = estimation.MUSIC(ula, wavelength, grid)
rm_estimator = estimation.RootMUSIC1D(wavelength)
esprit_estimator = estimation.Esprit1D(wavelength)

l_subarrays = 2  # 设置划分为 2 个子阵
NUM_RX_SS = NUM_RX - l_subarrays + 1 # 子阵的有效阵元数变为 8 - 2 + 1 = 7
ula_ss = model.UniformLinearArray(NUM_RX_SS, d0)
ssmusic_estimator = estimation.MUSIC(ula_ss, wavelength, grid)

# 存放算法 MSE 的列表
mse_music, mse_rm, mse_es, mse_ss = [], [], [], []
crb_sto_list, crb_det_list, crb_stouc_list = [], [], []

print(f"Starting MSE vs CRB Evaluation ...")

# ================= 核心测试循环 =================
for snr in SNR_RANGE:
    print(f"Simulating SNR = {snr:3d} dB ... ", end="", flush=True)

    # 1. 计算理论极限 CRB
    Rs = np.array([[1.0]])
    power_noise = 10 ** (-snr / 10.0)
    B_sto = perf.crb_sto_farfield_1d(ula, sources, wavelength, Rs, power_noise, NUM_SNAPSHOTS)
    crb_sto_list.append(B_sto[0, 0])

    # 2. 蒙特卡罗实验跑算法
    err_music, err_rm, err_es, err_ss = [], [], [], []

    for r in range(NUM_REPEATS):
        X = generate_ideal_data([TRUE_ANGLE], snr_db=snr, num_rx=NUM_RX, num_snapshots=NUM_SNAPSHOTS)
        R = (X @ X.conj().T) / NUM_SNAPSHOTS

        # 常规算法估计
        res_m, est_m = music_estimator.estimate(R, NUM_SOURCES)
        if res_m: err_music.append(est_m.locations[0] - TRUE_ANGLE_RAD)

        res_rm, est_rm = rm_estimator.estimate(R, NUM_SOURCES, d0)
        if res_rm: err_rm.append(est_rm.locations[0] - TRUE_ANGLE_RAD)

        res_es, est_es = esprit_estimator.estimate(R, NUM_SOURCES, d0)
        if res_es: err_es.append(est_es.locations[0] - TRUE_ANGLE_RAD)

        R_ss = estimation.spatial_smooth(R, l_subarrays, fb=True)
        res_ss, est_ss = ssmusic_estimator.estimate(R_ss, NUM_SOURCES)
        if res_ss: err_ss.append(est_ss.locations[0] - TRUE_ANGLE_RAD)

    # 3. 计算 MSE
    mse_music.append(np.mean(np.square(err_music)) if err_music else np.nan)
    mse_rm.append(np.mean(np.square(err_rm)) if err_rm else np.nan)
    mse_es.append(np.mean(np.square(err_es)) if err_es else np.nan)
    mse_ss.append(np.mean(np.square(err_ss)) if err_ss else np.nan)

    print("Done!")

# ================= 绘图 =================
plt.figure(figsize=(9, 6))

plt.semilogy(SNR_RANGE, mse_music, '-^', label='MUSIC')
plt.semilogy(SNR_RANGE, mse_rm, '-o', label='Root-MUSIC')
plt.semilogy(SNR_RANGE, mse_es, '-s', label='ESPRIT')
plt.semilogy(SNR_RANGE, mse_ss, '-x', color='purple', label='SS-MUSIC (FB)')
plt.semilogy(SNR_RANGE, crb_sto_list, '--', label='Stochastic CRB')
plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel(r'MSE / $\mathrm{rad}^2$', fontsize=13)

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "SNR_Evaluation.png"), dpi=300)
plt.show()
print("\n图表生成完毕！")