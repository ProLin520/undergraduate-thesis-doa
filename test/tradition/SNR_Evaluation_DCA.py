import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= 环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
doatools_path = os.path.join(current_dir, 'doatools.py-master')
if doatools_path not in sys.path:
    sys.path.append(doatools_path)

import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
from Graduation.data.data_create.simulation_generator import generate_dca1000_style_data
from Graduation.utils.radar_utils import process_radar_data, NUM_CHIRPS, NUM_FRAMES

save_dir = r"D:\Python\Project\doa_estimation\Graduation\result\plot\tradition"
os.makedirs(save_dir, exist_ok=True)

# ================= 配置参数 =================
plt.rcParams['font.size'] = 12
wavelength = 1.0
d0 = 0.5
NUM_RX = 8

TRUE_ANGLE = 20.0
TRUE_ANGLE_RAD = np.radians(TRUE_ANGLE)
SNR_RANGE = np.arange(-20, 11, 2)
NUM_SOURCES = 1

ula = model.UniformLinearArray(NUM_RX, d0)
sources = model.FarField1DSourcePlacement([TRUE_ANGLE_RAD])
grid = estimation.FarField1DSearchGrid(start=-np.pi / 2, stop=np.pi / 2, size=1801)

music_est = estimation.MUSIC(ula, wavelength, grid)
rm_est = estimation.RootMUSIC1D(wavelength)
esprit_est = estimation.Esprit1D(wavelength)

l_subarrays = 2
NUM_RX_SS = NUM_RX - l_subarrays + 1
ula_ss = model.UniformLinearArray(NUM_RX_SS, d0)
ssmusic_est = estimation.MUSIC(ula_ss, wavelength, grid)

mse_m, mse_rm, mse_es, mse_ss, crb_sto = [], [], [], [], []

print("开始包含 SS-MUSIC 的 DCA 仿真性能评估...")

for snr in SNR_RANGE:
    print(f"Simulating SNR = {snr:3d} dB ... ", end="", flush=True)

    power_noise = 10 ** (-snr / 10.0)
    Rs = np.array([[1.0]])
    B = perf.crb_sto_farfield_1d(ula, sources, wavelength, Rs, power_noise, NUM_CHIRPS)
    crb_sto.append(B[0, 0])

    sim_cube = generate_dca1000_style_data([TRUE_ANGLE], snr_db=snr, sim_rx=NUM_RX)
    _, R_frames = process_radar_data(sim_cube, is_simulation=True)

    err_m, err_rm, err_es, err_ss = [], [], [], []
    for f in range(NUM_FRAMES):
        R = R_frames[f]

        res, est = music_est.estimate(R, NUM_SOURCES)
        if res: err_m.append(est.locations[0] - TRUE_ANGLE_RAD)

        res, est = rm_est.estimate(R, NUM_SOURCES, d0)
        if res: err_rm.append(est.locations[0] - TRUE_ANGLE_RAD)

        res, est = esprit_est.estimate(R, NUM_SOURCES, d0)
        if res: err_es.append(est.locations[0] - TRUE_ANGLE_RAD)

        R_ss = estimation.spatial_smooth(R, l_subarrays, fb=True)
        res, est = ssmusic_est.estimate(R_ss, NUM_SOURCES)
        if res: err_ss.append(est.locations[0] - TRUE_ANGLE_RAD)

    mse_m.append(np.mean(np.square(err_m)))
    mse_rm.append(np.mean(np.square(err_rm)))
    mse_es.append(np.mean(np.square(err_es)))
    mse_ss.append(np.mean(np.square(err_ss)))
    print("Done!")

# ================= 绘图 =================
plt.figure(figsize=(9, 6))

plt.semilogy(SNR_RANGE, mse_m, '-^', label='MUSIC')
plt.semilogy(SNR_RANGE, mse_rm, '-o', label='Root-MUSIC')
plt.semilogy(SNR_RANGE, mse_es, '-s', label='ESPRIT')
plt.semilogy(SNR_RANGE, mse_ss, '-x', color='purple', label='SS-MUSIC (FB)')
plt.semilogy(SNR_RANGE, crb_sto, '--k', label='CRB (Stochastic)')

plt.xlabel('Input SNR (dB)')
plt.ylabel(r'MSE / $\mathrm{rad}^2$')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "SNR_Evaluation_DCA.png"), dpi=300)
plt.show()