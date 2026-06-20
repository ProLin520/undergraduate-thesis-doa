import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
from Graduation.data.data_create.simulation_generator import generate_dca1000_style_data
from Graduation.utils.radar_utils import process_radar_data, NUM_CHIRPS, NUM_FRAMES
current_dir = os.path.dirname(os.path.abspath(__file__))
doatools_path = os.path.join(current_dir, 'doatools.py-master')
if doatools_path not in sys.path:
    sys.path.append(doatools_path)

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "Snapshots_Evaluation_DCA.png")

plt.rcParams['font.size'] = 12
wavelength = 1.0
d0 = 0.5
NUM_RX = 8
FIXED_SNR = 0

# 3 个目标的真值
TRUE_ANGLES = [-20.0, 0.0, 20.0]
TRUE_ANGLES_RAD = np.sort(np.radians(TRUE_ANGLES))
NUM_SOURCES = len(TRUE_ANGLES)

# 严格遵守物理限制：快拍数最大只到 NUM_CHIRPS (即 60)
SNAPSHOTS_RANGE = [5, 10, 15, 20, 30, 40, 50, NUM_CHIRPS]

# ================= 初始化 doatools 对象 =================
ula = model.UniformLinearArray(NUM_RX, d0)
sources = model.FarField1DSourcePlacement(TRUE_ANGLES_RAD)
grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)

music_est = estimation.MUSIC(ula, wavelength, grid)
rm_est = estimation.RootMUSIC1D(wavelength)
esprit_est = estimation.Esprit1D(wavelength)

l_subarrays = 2
NUM_RX_SS = NUM_RX - l_subarrays + 1
ula_ss = model.UniformLinearArray(NUM_RX_SS, d0)
ssmusic_est = estimation.MUSIC(ula_ss, wavelength, grid)

mse_m, mse_rm, mse_es, mse_ss = [], [], [], []
crb_sto_list = []

print(f"开始 DCA 格式多信源快拍数评估 (信源数={NUM_SOURCES}, 物理最高快拍={NUM_CHIRPS}, SNR={FIXED_SNR}dB)...")

# 核心技巧：只生成一次 DCA 数据，保证公平
sim_cube = generate_dca1000_style_data(TRUE_ANGLES, snr_db=FIXED_SNR, sim_rx=NUM_RX)
X_frames, _ = process_radar_data(sim_cube, is_simulation=True)

# ================= 核心测试循环 =================
for snap in SNAPSHOTS_RANGE:
    print(f"Simulating Snapshots (Chirps) = {snap:2d} ... ", end="", flush=True)

    # 1. 计算理论 CRB
    Rs = np.eye(NUM_SOURCES)
    power_noise = 10 ** (-FIXED_SNR / 10.0)
    B_sto = perf.crb_sto_farfield_1d(ula, sources, wavelength, Rs, power_noise, snap)
    crb_sto_list.append(np.mean(np.diag(B_sto)))

    err_music, err_rm, err_es, err_ss = [], [], [], []

    for f in range(NUM_FRAMES):
        X_f = X_frames[f]
        if X_f.shape[0] != NUM_RX:
            X_f = X_f.T

        X_snap = X_f[:, :snap]
        R_snap = (X_snap @ X_snap.conj().T) / snap

        # (1) MUSIC
        res, est = music_est.estimate(R_snap, NUM_SOURCES)
        if res and len(est.locations) == NUM_SOURCES:
            est_sorted = np.sort(est.locations)
            err_music.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        # (2) Root-MUSIC
        res, est = rm_est.estimate(R_snap, NUM_SOURCES, d0)
        if res and len(est.locations) == NUM_SOURCES:
            est_sorted = np.sort(est.locations)
            err_rm.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        # (3) ESPRIT
        res, est = esprit_est.estimate(R_snap, NUM_SOURCES, d0)
        if res and len(est.locations) == NUM_SOURCES:
            est_sorted = np.sort(est.locations)
            err_es.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

        # (4) SS-MUSIC
        R_ss = estimation.spatial_smooth(R_snap, l_subarrays, fb=True)
        res, est = ssmusic_est.estimate(R_ss, NUM_SOURCES)
        if res and len(est.locations) == NUM_SOURCES:
            est_sorted = np.sort(est.locations)
            err_ss.append(np.mean(np.square(est_sorted - TRUE_ANGLES_RAD)))

    mse_m.append(np.mean(err_music) if err_music else np.nan)
    mse_rm.append(np.mean(err_rm) if err_rm else np.nan)
    mse_es.append(np.mean(err_es) if err_es else np.nan)
    mse_ss.append(np.mean(err_ss) if err_ss else np.nan)

    print("Done!")

# ================= 绘图 =================
plt.figure(figsize=(9, 6))
ax = plt.gca()

ax.loglog(SNAPSHOTS_RANGE, mse_m, '-^', label='MUSIC')
ax.loglog(SNAPSHOTS_RANGE, mse_rm, '-o', label='Root-MUSIC')
ax.loglog(SNAPSHOTS_RANGE, mse_es, '-s', label='ESPRIT')
ax.loglog(SNAPSHOTS_RANGE, mse_ss, '-x', color='purple', label='SS-MUSIC (FB)')
ax.loglog(SNAPSHOTS_RANGE, crb_sto_list, '--k', label='CRB (Effective SNR)')

# --- 强制接管 X 轴显示属性 ---
ax.set_xlim([0, 65])
ax.set_xticks(SNAPSHOTS_RANGE)  # 强制设置主刻度为你测试的那些点
ax.set_xticklabels([str(x) for x in SNAPSHOTS_RANGE])  # 把刻度强制转换为字符串显示

# 移除对数坐标默认的次级刻度（minor ticks），防止视觉干扰
ax.xaxis.set_minor_formatter(plt.NullFormatter())

plt.xlabel('Number of Snapshots / Chirps', fontsize=13)
plt.ylabel(r'Average MSE / $\mathrm{rad}^2$', fontsize=13)

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()