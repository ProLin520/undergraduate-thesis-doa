import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_base = os.path.dirname(os.path.dirname(current_dir))

if project_base not in sys.path:
    sys.path.append(project_base)
    sys.path.append(os.path.join(project_base, 'Graduation', 'external'))

from Graduation.utils.radar_utils import process_radar_data, load_and_reshape, NUM_FRAMES
from doatools import model, estimation

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "music_esprit_real_result.png")

# ================= 配置参数 =================
TARGET_FRAME_IDX = 80
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

wavelength = 1.0
d0 = 0.5

def apply_fbss(R):
    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    R_fb = 0.5 * (R + J @ np.conj(R) @ J)
    return R_fb

def doatools_music(R, num_sources, num_rx, use_fbss=False):
    if use_fbss:
        R = apply_fbss(R)
    ula = model.UniformLinearArray(num_rx, d0)
    grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)
    music_estimator = estimation.MUSIC(ula, wavelength, grid)
    res, est, sp_val = music_estimator.estimate(R, num_sources, return_spectrum=True)
    P_linear = np.abs(sp_val)
    P_db = 10 * np.log10(P_linear / np.max(P_linear))
    theta_range = np.linspace(-90, 90, 1801)
    peaks = np.sort(np.degrees(est.locations)) if res else np.array([])
    return theta_range, P_db, peaks

def doatools_esprit(R, num_sources, num_rx, use_fbss=False):
    if use_fbss:
        R = apply_fbss(R)
    esprit_estimator = estimation.Esprit1D(wavelength)
    res, est = esprit_estimator.estimate(R, num_sources, d0)
    if res:
        angles = np.sort(np.degrees(est.locations))
        return angles
    return np.full(num_sources, np.nan)

# ================= 1. 数据加载与处理 =================
print("加载实测数据 ...")
bin_path = os.path.join(project_base, 'data', 'raw', 'cropped.bin')

if not os.path.exists(bin_path):
    print(f" 找不到文件: {bin_path}")
    sys.exit()

data_cube_real = load_and_reshape(bin_path, conjugate=True)
X_real, R_real = process_radar_data(data_cube_real, is_simulation=False)

mus_ang1, mus_spec1, peaks1 = doatools_music(R_real[TARGET_FRAME_IDX], 1, 8, use_fbss=False)
mus_peak1 = peaks1[0] if len(peaks1) > 0 else np.nan

esp_res_real = []
for i in range(NUM_FRAMES):
    esp_res_real.append(doatools_esprit(R_real[i], 1, 8, use_fbss=True)[0])

esp_mean1 = np.nanmean(esp_res_real)
val_80_real = esp_res_real[TARGET_FRAME_IDX]

# ================= 2. 绘制 (1, 2) 组合图 =================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- 图像 1: 实测 MUSIC ---
axes[0].plot(mus_ang1, mus_spec1, 'b-', label='MUSIC Spectrum')
axes[0].axvline(mus_peak1, color='r', linestyle='--', label=f'Peak: {mus_peak1:.1f}°')
axes[0].set_title(f'MUSIC - 实测单信源 (Frame {TARGET_FRAME_IDX})')
axes[0].set_xlabel('Angle(degrees)')
axes[0].set_ylabel('Normalized Power(dB)')
axes[0].set_xlim([-90, 45])
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc='upper right')

# --- 图像 2: 实测 ESPRIT ---
axes[1].plot(range(NUM_FRAMES), esp_res_real, 'b-o', markersize=3, alpha=0.4, label='Estimated DOA')
axes[1].axhline(esp_mean1, color='r', linestyle='--', label=f'Mean: {esp_mean1:.2f}°')
axes[1].vlines(TARGET_FRAME_IDX, axes[1].get_ylim()[0], val_80_real,
               colors='r', linestyles=':', alpha=0.7)
axes[1].scatter(TARGET_FRAME_IDX, val_80_real, s=150, facecolors='none', edgecolors='red', linewidth=2, zorder=5)
axes[1].annotate(f'{val_80_real:.2f}°',
                   xy=(TARGET_FRAME_IDX, val_80_real),
                   xytext=(TARGET_FRAME_IDX - 15, val_80_real + 0.5),
                   color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle="->", color='red'),
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8))

axes[1].set_title('ESPRIT - 实测单信源')
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Angle(degrees)')
axes[1].set_ylim([esp_mean1 - 5, esp_mean1 + 5])
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()