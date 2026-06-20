import torch
import torch.nn as nn

from models.dl_model.grid_based_network import Grid_Based_network


class MLP(Grid_Based_network):
    def __init__(self, model_size, drop_out_ratio=0, sp_mode=False, **kwargs):
        out_dims = model_size[-1]
        if sp_mode:
            Grid_Based_network.__init__(self, **kwargs)
            assert self._grid.shape[0] == out_dims, 'error grid_size or model output dims'
            self.sp_to_doa = self.grid_to_theta  # 重命名 空间谱估计->角度的函数
        else:
            # super(VisionTransformer, self).__init__()
            nn.Module.__init__(self)

        layers = []

        for input_size, output_size in zip(model_size[:-2], model_size[1:-1]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(drop_out_ratio))
        layers.append(nn.Linear(model_size[-2], model_size[-1]))  # output layer 不需要bn,relu
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LearningSPICE_MLP(torch.nn.Module):
    def __init__(self, M=8):
        super().__init__()
        in_dim = M * (M + 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, in_dim)
        )
    def forward(self, x): return self.net(x)


class LearningSPICE_SP_MLP(torch.nn.Module):
    """直接拟合空间谱(181维)的 MLP，用于 K=3 场景"""
    def __init__(self, M=8, out_dim=181):
        super().__init__()
        in_dim = M * (M + 1) # 72维
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024), torch.nn.BatchNorm1d(1024), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024), torch.nn.BatchNorm1d(1024), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, out_dim) # 输出 181
        )
    def forward(self, x):
        return self.net(x)


def vec72_to_scm(v, M=8):
    """将 MLP 输出的 72 维向量完美还原为 8x8 埃尔米特协方差矩阵"""
    B = v.shape[0]
    R = torch.zeros((B, M, M), dtype=torch.complex64, device=v.device)
    triu_idx = torch.triu_indices(M, M)
    R[:, triu_idx[0], triu_idx[1]] = torch.complex(v[:, :36], v[:, 36:])
    # 转换为共轭对称矩阵 (Hermitian)
    R_herm = R + R.conj().transpose(1, 2)
    diag_idx = torch.arange(M)
    R_herm[:, diag_idx, diag_idx] /= 2.0  # 对角线被加了两次，除以 2 还原
    return R_herm

def scm_to_vec72(R):
    B, M, _ = R.shape
    triu_idx = torch.triu_indices(M, M, device=R.device)
    R_triu = R[:, triu_idx[0], triu_idx[1]]
    return torch.cat([R_triu.real, R_triu.imag], dim=1)
