import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_base = os.path.dirname(os.path.dirname(current_dir))

if project_base not in sys.path:
    sys.path.append(project_base)
    sys.path.append(os.path.join(project_base, 'Graduation', 'external'))

from Graduation.utils.radar_utils import process_radar_data, NUM_FRAMES
from Graduation.data.data_create.simulation_generator import generate_dca1000_style_data
from Graduation.test.tradition.MUSIC_ESPRIT_real import doatools_music, doatools_esprit

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)

# ================= 配置参数 =================
TARGET_FRAME_IDX = 80
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

wavelength = 1.0
d0 = 0.5

# ================= 1. 生成仿真数据 =================
print("生成仿真数据 ...")
true_angle_single = [0]
true_angle_multi = [-30, 0, 30]

sim_data_single = generate_dca1000_style_data(angle_deg_list=true_angle_single, snr_db=0)
X_sim_single, R_sim_single = process_radar_data(sim_data_single, is_simulation=True)

sim_data_multi = generate_dca1000_style_data(angle_deg_list=true_angle_multi, snr_db=0)
X_sim_multi, R_sim_multi = process_radar_data(sim_data_multi, is_simulation=True)

# ================= 2. 算法处理 =================
print("处理算法并计算均值 ...")
# 单信源处理
mus_ang_1, mus_spec_1, peaks_1 = doatools_music(R_sim_single[TARGET_FRAME_IDX], 1, 8)
esp_res_sim_1 = np.array([doatools_esprit(R_sim_single[i], 1, 8) for i in range(NUM_FRAMES)])

# 三信源处理
mus_ang_3, mus_spec_3, peaks_3 = doatools_music(R_sim_multi[TARGET_FRAME_IDX], 3, 8)
esp_res_sim_3 = np.array([doatools_esprit(R_sim_multi[i], 3, 8) for i in range(NUM_FRAMES)])


# ================= 3. 分别绘制并保存四张图 =================
print("生成并保存图表 ...")

# ----------------- 图 1: 单信源 MUSIC -----------------
plt.figure(figsize=(8, 6))
plt.plot(mus_ang_1, mus_spec_1, 'g-', label='MUSIC Spectrum')
plt.axvline(true_angle_single[0], color='orange', label='真值')
plt.text(true_angle_single[0], -0.5, f'{true_angle_single[0]}°', color='orange', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.title(f'MUSIC - 仿真单信源 (Frame {TARGET_FRAME_IDX})')
plt.xlabel('Angle(degrees)')
plt.ylabel('Normalized Power(dB)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
save_path_1 = os.path.join(save_dir, "MUSIC_sim_single.png")
plt.savefig(save_path_1, dpi=300)
plt.show()

# ----------------- 图 2: 单信源 ESPRIT -----------------
plt.figure(figsize=(8, 6))
mean_1 = np.nanmean(esp_res_sim_1[:, 0])
plt.plot(range(NUM_FRAMES), esp_res_sim_1[:, 0], color='#1f77b4', marker='.', markersize=3, alpha=0.4, label='Estimated DOA')
plt.axhline(mean_1, color='r', linestyle='--', label=f'Mean: {mean_1:.2f}°')
plt.axhline(true_angle_single[0], color='gray', linestyle=':', alpha=0.6, label=f'Truth: {true_angle_single[0]}°')

val_80_1 = esp_res_sim_1[TARGET_FRAME_IDX, 0]
plt.scatter(TARGET_FRAME_IDX, val_80_1, s=150, facecolors='none', edgecolors='red', linewidth=2, zorder=5)
plt.annotate(f'Frame 80: {val_80_1:.2f}°',
             xy=(TARGET_FRAME_IDX, val_80_1),
             xytext=(TARGET_FRAME_IDX - 25, val_80_1 + 1.0),
             color='red', fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='red'),
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8))

plt.title('ESPRIT - 仿真单信源稳定性分析')
plt.xlabel('Frame Index')
plt.ylabel('Angle(degrees)')
plt.ylim([true_angle_single[0] - 5, true_angle_single[0] + 5])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
save_path_2 = os.path.join(save_dir, "ESPRIT_sim_single.png")
plt.savefig(save_path_2, dpi=300)
plt.show()

# ----------------- 图 3: 三信源 MUSIC -----------------
plt.figure(figsize=(8, 6))
plt.plot(mus_ang_3, mus_spec_3, 'g-', label='MUSIC Spectrum')
for i, ang in enumerate(true_angle_multi):
    plt.axvline(ang, color='orange', label='真值' if i == 0 else None)
    plt.text(ang, -0.5, f'{ang}°', color='orange', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.title(f'MUSIC - 仿真三信源 (Frame {TARGET_FRAME_IDX})')
plt.xlabel('Angle(degrees)')
plt.ylabel('Normalized Power(dB)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
save_path_3 = os.path.join(save_dir, "MUSIC_sim_three.png")
plt.savefig(save_path_3, dpi=300)
plt.show()

# ----------------- 图 4: 三信源 ESPRIT -----------------
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for k in range(3):
    mean_k = np.nanmean(esp_res_sim_3[:, k])
    plt.plot(range(NUM_FRAMES), esp_res_sim_3[:, k], color=colors[k], marker='.', markersize=3, alpha=0.4)
    plt.axhline(mean_k, color=colors[k], linestyle='--', alpha=0.8)
    plt.axhline(true_angle_multi[k], color=colors[k], linestyle=':', alpha=0.4)
    # 直接在曲线旁标注真值与均值，避免图例过载
    plt.text(5, mean_k + 0.8, f'Truth: {true_angle_multi[k]}°\nMean: {mean_k:.2f}°', color=colors[k], fontsize=10, fontweight='bold')

    val_80_3 = esp_res_sim_3[TARGET_FRAME_IDX, k]
    plt.scatter(TARGET_FRAME_IDX, val_80_3, s=150, facecolors='none', edgecolors='red', linewidth=1.5, zorder=5)

    offset_y = 1.8 if k % 2 == 0 else -2.5
    offset_x = -15 if k % 2 == 0 else 5
    plt.annotate(f'{val_80_3:.2f}°',
                 xy=(TARGET_FRAME_IDX, val_80_3),
                 xytext=(TARGET_FRAME_IDX + offset_x, val_80_3 + offset_y),
                 color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='red'),
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7))

plt.title('ESPRIT - 仿真三信源稳定性分析')
plt.xlabel('Frame Index')
plt.ylabel('Angle(degrees)')
plt.ylim([min(true_angle_multi) - 10, max(true_angle_multi) + 10])
plt.grid(True, linestyle=':', alpha=0.6)

# 自定义底部图例
custom_legend = [Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Mean'),
                 Line2D([0], [0], linestyle=':', color='gray', label='Truth'),
                 Line2D([0], [0], marker='o', color='w', markeredgecolor='red', markerfacecolor='none', label=f'Frame {TARGET_FRAME_IDX}')]
plt.legend(handles=custom_legend, loc='lower right')

plt.tight_layout()
save_path_4 = os.path.join(save_dir, "ESPRIT_sim_three.png")
plt.savefig(save_path_4, dpi=300)
plt.show()

print(f"四张图像均已成功保存至: \n{save_dir}")