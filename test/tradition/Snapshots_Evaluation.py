import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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
save_path = os.path.join(save_dir, "Snapshots_Evaluation.png")

# ================= 配置参数 =================
plt.rcParams['font.size'] = 12

wavelength = 1.0
d0 = 0.5
NUM_RX = 8
FIXED_SNR = 0
NUM_REPEATS = 500

TRUE_ANGLES = [-20.0, 0.0, 20.0]
TRUE_ANGLES_RAD = np.sort(np.radians(TRUE_ANGLES))
NUM_SOURCES = len(TRUE_ANGLES)

SNAPSHOTS_RANGE = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]

ula = model.UniformLinearArray(NUM_RX, d0)
# [核心修改 2]：告知模型这是 3 个信源
sources = model.FarField1DSourcePlacement(TRUE_ANGLES_RAD)

grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)
music_estimator = estimation.MUSIC(ula, wavelength, grid)
rm_estimator = estimation.RootMUSIC1D(wavelength)
esprit_estimator = estimation.Esprit1D(wavelength)

l_subarrays = 2
NUM_RX_SS = NUM_RX - l_subarrays + 1
ula_ss = model.UniformLinearArray(NUM_RX_SS, d0)
ssmusic_estimator = estimation.MUSIC(ula_ss, wavelength, grid)

mse_music, mse_rm, mse_es, mse_ss = [], [], [], []
crb_sto_list = []

print(f"Starting Multi-Source Snapshots Evaluation ({NUM_SOURCES} targets, SNR={FIXED_SNR}dB)...")

for snap in SNAPSHOTS_RANGE:
    print(f"Simulating Snapshots = {snap:3d} ... ", end="", flush=True)

    # [核心修改 3]：CRB 的输入矩阵必须是 3x3，且传入当前的 snap 变量
    Rs = np.eye(NUM_SOURCES)
    power_noise = 10 ** (-FIXED_SNR / 10.0)

    B_sto = perf.crb_sto_farfield_1d(ula, sources, wavelength, Rs, power_noise, snap)
    crb_sto_list.append(np.mean(np.diag(B_sto))) # 取三信源CRB的平均下限

    err_music, err_rm, err_es, err_ss = [], [], [], []

    for r in range(NUM_REPEATS):
        # [核心修改 4]：生成数据时，传入3个角度列表，以及当前循环的快拍数 snap
        X = generate_ideal_data(TRUE_ANGLES, snr_db=FIXED_SNR, num_rx=NUM_RX, num_snapshots=snap)
        R = (X @ X.conj().T) / snap

        # [核心修改 5]：估算出3个角度后，排序，再与排好序的真值做相减求均方误差
        res_m, est_m = music_estimator.estimate(R, NUM_SOURCES)
        if res_m and len(est_m.locations) == NUM_SOURCES:
            est_sorted = np.sort(est_m.locations)
            err_music.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        res_rm, est_rm = rm_estimator.estimate(R, NUM_SOURCES, d0)
        if res_rm and len(est_rm.locations) == NUM_SOURCES:
            est_sorted = np.sort(est_rm.locations)
            err_rm.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        res_es, est_es = esprit_estimator.estimate(R, NUM_SOURCES, d0)
        if res_es and len(est_es.locations) == NUM_SOURCES:
            est_sorted = np.sort(est_es.locations)
            err_es.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        R_ss = estimation.spatial_smooth(R, l_subarrays, fb=True)
        res_ss, est_ss = ssmusic_estimator.estimate(R_ss, NUM_SOURCES)
        if res_ss and len(est_ss.locations) == NUM_SOURCES:
            est_sorted = np.sort(est_ss.locations)
            err_ss.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

    mse_music.append(np.mean(err_music) if err_music else np.nan)
    mse_rm.append(np.mean(err_rm) if err_rm else np.nan)
    mse_es.append(np.mean(err_es) if err_es else np.nan)
    mse_ss.append(np.mean(err_ss) if err_ss else np.nan)

    print("Done!")

# ================= 绘图 =================
plt.figure(figsize=(9, 6))
plt.loglog(SNAPSHOTS_RANGE, mse_music, '-^', label='MUSIC')
plt.loglog(SNAPSHOTS_RANGE, mse_rm, '-o', label='Root-MUSIC')
plt.loglog(SNAPSHOTS_RANGE, mse_es, '-s', label='ESPRIT')
plt.loglog(SNAPSHOTS_RANGE, mse_ss, '-x', color='purple', label='SS-MUSIC (FB)')
plt.loglog(SNAPSHOTS_RANGE, crb_sto_list, '--k', label='CRB (Stochastic)')

plt.xlabel('Number of Snapshots', fontsize=13)
plt.ylabel(r'Average MSE / $\mathrm{rad}^2$', fontsize=13)

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

