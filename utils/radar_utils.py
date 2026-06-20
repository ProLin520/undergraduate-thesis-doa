import numpy as np
import os

# 全局参数
NUM_CHIRPS = 60
NUM_RX = 4
NUM_SAMPLES = 256
NUM_FRAMES = 100


def load_and_reshape(file_path, conjugate=True):
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None

    file_size = os.path.getsize(file_path)
    bytes_per_frame = NUM_CHIRPS * NUM_RX * NUM_SAMPLES * 4
    total_frames = file_size // bytes_per_frame

    start_frame = (total_frames - NUM_FRAMES) // 2
    if start_frame < 0: start_frame = 0
    seek_offset = start_frame * bytes_per_frame
    bytes_to_read = NUM_FRAMES * bytes_per_frame

    with open(file_path, 'rb') as f:
        f.seek(seek_offset)
        adc_data = np.fromfile(f, dtype=np.int16, count=bytes_to_read // 2)

    adc_data = adc_data.reshape(-1, 2)
    complex_data = adc_data[:, 0] + 1j * adc_data[:, 1]

    if conjugate: complex_data = np.conj(complex_data)
    try:
        return complex_data.reshape(NUM_FRAMES, NUM_CHIRPS, NUM_RX, NUM_SAMPLES)
    except:
        return None


def process_radar_data(data_input, is_simulation=False):
    if data_input is None: return None, None

    if data_input.ndim == 2:
        X = data_input
        M, N = X.shape
        R = (X @ X.conj().T) / N
        return X[np.newaxis, :, :], R[np.newaxis, :, :]

    data_cube = data_input
    num_frames, num_chirps, num_rx, num_samples = data_cube.shape

    # 【关键1】加 Hanning 窗，防止相位模糊
    window = np.hanning(num_samples)

    if is_simulation:
        eff_chirps = num_chirps
        X_final = np.zeros((num_frames, num_rx, eff_chirps), dtype=complex)
        R_final = np.zeros((num_frames, num_rx, num_rx), dtype=complex)

        for i in range(num_frames):
            frame_data = data_cube[i] * window
            range_fft = np.fft.fft(frame_data, axis=2)

            range_profile = np.abs(range_fft).mean(axis=(0, 1))
            target_idx = np.argmax(range_profile[5:]) + 5

            X_i = range_fft[:, :, target_idx].T
            X_i = X_i / (np.max(np.abs(X_i)) + 1e-8)  # 底层归一化

            R_i = (X_i @ X_i.conj().T) / eff_chirps
            X_final[i] = X_i
            R_final[i] = R_i

    else:
        eff_chirps = num_chirps // 3
        virtual_rx = 8
        X_final = np.zeros((num_frames, virtual_rx, eff_chirps), dtype=complex)
        R_final = np.zeros((num_frames, virtual_rx, virtual_rx), dtype=complex)

        for i in range(num_frames):
            frame_data = data_cube[i] * window
            range_fft = np.fft.fft(frame_data, axis=2)

            data_tx0 = range_fft[0::3, :, :]
            data_tx2 = range_fft[2::3, :, :]

            # 用 TX0 寻找目标，避免多径干扰
            range_profile = np.abs(data_tx0).mean(axis=(0, 1))
            target_idx = np.argmax(range_profile[5:]) + 5

            X_tx0 = data_tx0[:, :, target_idx].T  # [4, 20]
            X_tx2 = data_tx2[:, :, target_idx].T  # [4, 20]

            phase_diffs = X_tx0[1:, :] * np.conj(X_tx0[:-1, :])
            delta_phi = np.angle(np.mean(phase_diffs))

            expected_X4 = X_tx0[3, :] * np.exp(1j * delta_phi)

            correction_phasor = np.mean(expected_X4 * np.conj(X_tx2[0, :]))
            phase_corr = np.angle(correction_phasor)

            X_tx2_comp = X_tx2 * np.exp(1j * phase_corr)

            # 完美拼接为 8 阵元 ULA
            X_i = np.concatenate([X_tx0, X_tx2_comp], axis=0)

            # 底层幅度归一化
            X_i = X_i / (np.max(np.abs(X_i)) + 1e-8)

            # 直接求协方差矩阵 R
            R_i = (X_i @ X_i.conj().T) / eff_chirps

            X_final[i] = X_i
            R_final[i] = R_i

    return X_final, R_final