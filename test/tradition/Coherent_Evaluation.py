import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= 1. 环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
doatools_path = os.path.join(current_dir, 'doatools.py-master')
if doatools_path not in sys.path:
    sys.path.append(doatools_path)

import doatools.model as model
import doatools.estimation as estimation

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "Coherent_Evaluation.png")

# ================= 2. 参数配置 =================
plt.rcParams['font.size'] = 11
NUM_RX = 8
d0 = 0.5
L = 200
SNR = 15
angles = [10.0, 30.0]
K = len(angles)
wavelength = 1.0

# ================= 3. 生成完全相干信号 =================
s1 = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
s2 = s1 * np.exp(1j * np.pi / 4) * 0.95  # 完全相干：信号2是信号1的相移和缩放
S = np.vstack((s1, s2))
A = np.exp(1j * 2 * np.pi * d0 * np.arange(NUM_RX)[:, None] * np.sin(np.radians(angles)))
X_pure = A @ S
noise_p = np.mean(np.abs(X_pure) ** 2) / (10 ** (SNR / 10.0))
N = (np.random.randn(NUM_RX, L) + 1j * np.random.randn(NUM_RX, L)) / np.sqrt(2) * np.sqrt(noise_p)
X = X_pure + N
R = (X @ X.conj().T) / L


# ================= 4. 空间谱手动计算函数 =================
def compute_spectra(R_mat, ula_obj, num_src):
    """
    手动计算并归一化 MUSIC 和 MVDR 空间谱
    """
    theta_range = np.linspace(-90, 90, 1801)
    A_search = ula_obj.steering_matrix(model.FarField1DSourcePlacement(np.radians(theta_range)), wavelength)

    # --- MUSIC ---
    eigvals, eigvecs = np.linalg.eigh(R_mat)
    idx = np.argsort(eigvals)[::-1]
    En = eigvecs[:, idx[num_src:]]  # 噪声子空间

    spec_music = np.zeros(1801)
    # --- MVDR ---
    R_inv = np.linalg.pinv(R_mat)
    spec_mvdr = np.zeros(1801)

    for i in range(1801):
        a = A_search[:, i]
        # MUSIC: 1 / (a^H * En * En^H * a)
        spec_music[i] = 1.0 / np.abs(a.conj().T @ En @ En.conj().T @ a)
        # MVDR: 1 / (a^H * R_inv * a)
        spec_mvdr[i] = 1.0 / np.abs(a.conj().T @ R_inv @ a)

    return theta_range, 10 * np.log10(spec_music / np.max(spec_music)), 10 * np.log10(spec_mvdr / np.max(spec_mvdr))


# ================= 5. 执行对比 =================
ula_full = model.UniformLinearArray(NUM_RX, d0)
# 空间平滑设置
l_sub = 4
R_ss = estimation.spatial_smooth(R, l_sub, fb=True)
ula_ss = model.UniformLinearArray(NUM_RX - l_sub + 1, d0)

# 计算谱
theta, m_orig, v_orig = compute_spectra(R, ula_full, K)
_, m_ss, v_ss = compute_spectra(R_ss, ula_ss, K)

# ================= 6. 绘图 =================
plt.figure(figsize=(9, 6))

# Top: Standard
plt.plot(theta, m_orig, 'r-', label='MUSIC')
plt.plot(theta, v_orig, 'g--', label='MVDR')
plt.plot(theta, m_ss, 'b-', label='SS-MUSIC')

for ang in angles: plt.axvline(ang, color='k', linestyle=':', alpha=0.5)
plt.ylim([-20, 5])
plt.xlabel('Angle (°)')
plt.ylabel('dB')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()