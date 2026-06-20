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
from Graduation.data.data_create.simulation_generator import generate_ideal_data

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)

def is_resolved_2(est, delta_rad):
    if est is None or len(est.locations) < 2: return False
    locs = np.sort(est.locations)
    if locs[0] >= 0 or locs[0] <= -delta_rad: return False
    if locs[1] <= 0 or locs[1] >= delta_rad: return False
    return True

def is_resolved_3(est, delta_rad):
    if est is None or len(est.locations) < 3: return False
    locs = np.sort(est.locations)
    true_locs = np.array([-delta_rad, 0.0, delta_rad])
    threshold = delta_rad / 2.0
    for i in range(3):
        if np.abs(locs[i] - true_locs[i]) >= threshold: return False
    return True

plt.rcParams['font.size'] = 12
wavelength = 1.0
d0 = 0.5
NUM_RX = 8
NUM_SNAPSHOTS = 200
NUM_REPEATS = 100

SNR_2_SOURCES = 10.0
SNR_3_SOURCES = 10.0

ula = model.UniformLinearArray(NUM_RX, d0)
grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)
music_est = estimation.MUSIC(ula, wavelength, grid)
rm_est = estimation.RootMUSIC1D(wavelength)
esprit_est = estimation.Esprit1D(wavelength)

l_sub = 2
ula_ss = model.UniformLinearArray(NUM_RX - l_sub + 1, d0)
ssmusic_est = estimation.MUSIC(ula_ss, wavelength, grid)

styles = {'MUSIC': '-^', 'RootMUSIC': '-o', 'ESPRIT': '-s', 'SSMUSIC': '-x'}
colors = {'MUSIC': 'C0', 'RootMUSIC': 'C1', 'ESPRIT': 'C2', 'SSMUSIC': 'purple'}

delta_2_range = np.arange(0, 21, 2)
res_2 = {k: [] for k in ['MUSIC', 'RootMUSIC', 'ESPRIT', 'SSMUSIC']}

print(f">>> 开始 2 信源分辨率测试 (SNR={SNR_2_SOURCES}dB)...")
for delta in delta_2_range:
    counts = np.zeros(4)
    delta_rad = np.radians(delta)
    # 当 delta=0 时，angles_deg 为 [0, 0]，完全重叠
    angles_deg = [-delta / 2, delta / 2]
    for _ in range(NUM_REPEATS):
        X = generate_ideal_data(angles_deg, snr_db=SNR_2_SOURCES, num_rx=NUM_RX, num_snapshots=NUM_SNAPSHOTS)
        R = (X @ X.conj().T) / NUM_SNAPSHOTS

        _, e_m = music_est.estimate(R, 2)
        if is_resolved_2(e_m, delta_rad): counts[0] += 1
        _, e_rm = rm_est.estimate(R, 2, d0)
        if is_resolved_2(e_rm, delta_rad): counts[1] += 1
        _, e_es = esprit_est.estimate(R, 2, d0)
        if is_resolved_2(e_es, delta_rad): counts[2] += 1
        R_ss = estimation.spatial_smooth(R, l_sub, fb=True)
        _, e_ss = ssmusic_est.estimate(R_ss, 2)
        if is_resolved_2(e_ss, delta_rad): counts[3] += 1

    for idx, key in enumerate(res_2.keys()):
        res_2[key].append(counts[idx] / NUM_REPEATS)
    print(f"Delta {delta:2d}° Done", end=" | ")

# 绘图 1
plt.figure(figsize=(8, 6))
for label, data in res_2.items():
    plt.plot(delta_2_range, data, styles[label], color=colors[label], label=label, linewidth=2)
# plt.title(f'2-Source Resolution (SNR={SNR_2_SOURCES}dB)')
plt.xlabel(r'Angular Separation $\Delta\theta$ (Deg)')
plt.xticks(np.arange(0, 21, 2))
plt.ylabel('Probability of Success')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Two-Source separated resolution.png"), dpi=300)
print(f"\n2信源图像已保存至: {save_dir}")


delta_3_range = np.arange(0, 21, 2)
res_3 = {k: [] for k in ['MUSIC', 'RootMUSIC', 'ESPRIT', 'SSMUSIC']}

print(f"\n>>> 开始 3 信源分辨率测试 (SNR={SNR_3_SOURCES}dB)...")
for delta in delta_3_range:
    counts = np.zeros(4)
    delta_rad = np.radians(delta)
    angles_deg = [-delta, 0.0, delta]
    for _ in range(NUM_REPEATS):
        X = generate_ideal_data(angles_deg, snr_db=SNR_3_SOURCES, num_rx=NUM_RX, num_snapshots=NUM_SNAPSHOTS)
        R = (X @ X.conj().T) / NUM_SNAPSHOTS

        _, e_m = music_est.estimate(R, 3)
        if is_resolved_3(e_m, delta_rad): counts[0] += 1
        _, e_rm = rm_est.estimate(R, 3, d0)
        if is_resolved_3(e_rm, delta_rad): counts[1] += 1
        _, e_es = esprit_est.estimate(R, 3, d0)
        if is_resolved_3(e_es, delta_rad): counts[2] += 1
        R_ss = estimation.spatial_smooth(R, l_sub, fb=True)
        _, e_ss = ssmusic_est.estimate(R_ss, 3)
        if is_resolved_3(e_ss, delta_rad): counts[3] += 1

    for idx, key in enumerate(res_3.keys()):
        res_3[key].append(counts[idx] / NUM_REPEATS)
    print(f"Delta {delta:2d}° Done", end=" | ")

# 绘图 2
plt.figure(figsize=(8, 6))
for label, data in res_3.items():
    plt.plot(delta_3_range, data, styles[label], color=colors[label], label=label, linewidth=2)
# plt.title(f'3-Source Resolution (SNR={SNR_3_SOURCES}dB)')
plt.xlabel(r'Angular Separation $\Delta\theta$ (Deg)')
plt.xticks(np.arange(0, 21, 2))
plt.ylabel('Probability of Success')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Three-Source separated resolution.png"), dpi=300)
plt.show()
print(f"\n3信源图像已保存至: {save_dir}\n")