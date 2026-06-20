import numpy as np
import doatools.model as model
from Graduation.utils.radar_utils import NUM_FRAMES, NUM_CHIRPS, NUM_SAMPLES

# --- 1. 雷达参数配置 ---
RADAR_CONFIG = {
    'c': 3e8,
    'f0': 77e9,
    'd_lambda': 0.5,
    'fs': 5e6,
    'slope': 64.985e12,
}


def generate_dca1000_style_data(angle_deg_list, snr_db, target_bin_index=20, sim_rx=8):

    if not isinstance(angle_deg_list, list):
        angle_deg_list = [angle_deg_list]

    num_samples = NUM_SAMPLES  # 256
    num_chirps = NUM_CHIRPS  # 60
    num_frames = NUM_FRAMES  # 100

    num_rx = sim_rx

    sim_cube = np.zeros((num_frames, num_chirps, num_rx, num_samples), dtype=complex)

    # 1. 基础的时间/频率信号
    t_idx = np.arange(num_samples)
    freq_norm = target_bin_index / num_samples
    base_signal = np.exp(1j * 2 * np.pi * freq_norm * t_idx).reshape(1, 1, 1, num_samples)

    # 2. 使用 doatools 生成完美导向矢量
    wavelength = 1.0
    d0 = 0.5
    ula = model.UniformLinearArray(num_rx, d0)
    sources = model.FarField1DSourcePlacement(np.radians(angle_deg_list))
    A = ula.steering_matrix(sources, wavelength)

    # 3. 循环叠加每个角度的独立信号
    for i, angle in enumerate(angle_deg_list):
        steering_vec = A[:, i].reshape(1, 1, num_rx, 1)
        random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(num_frames, num_chirps, 1, 1)))
        signal_component = base_signal * steering_vec * random_phase
        sim_cube += signal_component

    # 4. 添加环境噪声
    noise = (np.random.randn(*sim_cube.shape) + 1j * np.random.randn(*sim_cube.shape)) / np.sqrt(2)
    fft_gain_db = 10 * np.log10(num_samples)
    actual_snr_to_add = snr_db - fft_gain_db

    signal_power = 1.0
    noise_power = signal_power / (10 ** (actual_snr_to_add / 10))
    sim_cube = sim_cube + noise * np.sqrt(noise_power)
    return sim_cube

def generate_ideal_data(angle_deg_list, snr_db=10, num_rx=8, num_snapshots=200):

    if not isinstance(angle_deg_list, list):
        angle_deg_list = [angle_deg_list]

    wavelength = 1.0
    d0 = 0.5
    ula = model.UniformLinearArray(num_rx, d0)
    sources = model.FarField1DSourcePlacement(np.radians(angle_deg_list))
    A = ula.steering_matrix(sources, wavelength)

    K = len(angle_deg_list)
    S = (np.random.randn(K, num_snapshots) + 1j * np.random.randn(K, num_snapshots)) / np.sqrt(2)
    Noise = (np.random.randn(num_rx, num_snapshots) + 1j * np.random.randn(num_rx, num_snapshots)) / np.sqrt(2)

    X_pure = A @ S
    signal_power = np.mean(np.abs(X_pure) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    X = X_pure + Noise * np.sqrt(noise_power)
    return X


# --- 验证代码 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Graduation.utils.radar_utils import process_radar_data, NUM_FRAMES
    print("正在生成仿真数据 (Angle=30°, Bin=20)...")
    sim_data = generate_dca1000_style_data([-30, 0, 30], snr_db=15, target_bin_index=20)

    # 使用 radar_utils 处理
    X_out, R_out = process_radar_data(sim_data, is_simulation=True)

    print(f"生成的 Cube 维度: {sim_data.shape}")
    if R_out is not None:
        print(f"提取的 R 矩阵维度: {R_out.shape}")

    # 验证 1: 看看 FFT 峰值是不是真的在第 20 个点
    frame_0_chirp_0_rx_0 = sim_data[0, 0, 0, :]
    fft_spectrum = np.fft.fft(frame_0_chirp_0_rx_0)
    peak_loc = np.argmax(np.abs(fft_spectrum))
    print(f"FFT 峰值位置 (预期 20): {peak_loc}")

    # 验证 2: 看看协方差矩阵 R 是否包含了角度信息
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(fft_spectrum))
    plt.title("Range-FFT Profile (Peak at 20?)")
    plt.grid()

    if R_out is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(R_out[0]), cmap='viridis')
        plt.title("Covariance Matrix Magnitude")
        plt.colorbar()

    plt.show()
