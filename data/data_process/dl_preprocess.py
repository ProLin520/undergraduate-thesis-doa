import numpy as np
import torch

def preprocess_for_cnn(R, num_rx=8):
    R_cnn_base = np.conj(R.T)

    input_tensor = np.zeros((1, num_rx, num_rx, 3), dtype=np.float32)
    max_val = np.max(np.abs(R_cnn_base))

    max_val = max_val if max_val > 0 else 1e-8

    input_tensor[0, :, :, 0] = np.real(R_cnn_base) / max_val
    input_tensor[0, :, :, 1] = np.imag(R_cnn_base) / max_val
    input_tensor[0, :, :, 2] = np.angle(R_cnn_base)

    return input_tensor



def preprocess_for_vit(R):
    R = np.conj(R)

    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    R_fbss = 0.5 * (R + J @ np.conj(R) @ J)

    scm_real = np.real(R_fbss)
    scm_imag = np.imag(R_fbss)
    scm_pytorch = np.stack([scm_real, scm_imag], axis=0).astype(np.float32)
    scm_tensor = torch.from_numpy(scm_pytorch).unsqueeze(0)

    batch_size = scm_tensor.shape[0]
    max_vals = torch.max(torch.abs(scm_tensor.view(batch_size, -1)), dim=1)[0].view(batch_size, 1, 1, 1)
    scm_tensor = scm_tensor / (max_vals + 1e-8)

    return scm_tensor
