# 基于深度学习的波达方向估计算法研究

本仓库是本科毕业设计《基于深度学习的波达方向估计算法研究》的代码工程，主要用于论文中的 DOA（Direction of Arrival，波达方向）估计算法仿真、深度学习模型训练与毫米波雷达实测验证。

论文 PDF 已放在：

[docs/undergraduate-thesis-doa.pdf](docs/undergraduate-thesis-doa.pdf)

## 项目简介

DOA 估计是阵列信号处理中的基础问题，可用于雷达探测、无线通信、目标定位和波束控制等场景。传统 MUSIC、ESPRIT、Root-MUSIC、SS-MUSIC 等子空间算法在低信噪比、少快拍、小角度间隔、相干信源和阵列误差条件下容易出现性能下降。

本项目围绕复杂场景下的 DOA 估计展开，比较传统算法与多种深度学习方法在 BPSK 调制信号域、高斯窄带信号域和毫米波雷达实测数据上的表现。

主要研究内容包括：

- 构建均匀线阵远场窄带信号模型和阵列误差模型。
- 对 MUSIC、ESPRIT、Root-MUSIC、SS-MUSIC 等传统算法进行仿真分析。
- 构建 BPSK 原始 I/Q 输入和 SCM 协方差矩阵输入两类深度学习建模路线。
- 测试 IQ-ResNet、ViT、CNN、REG-CNN、SPE-CNN、MLP、Learning-SPICE 等模型。
- 使用 IWR1443BOOST 与 DCA1000 搭建毫米波雷达实测平台，验证模型迁移效果。

## 主要结论

论文实验表明，深度学习 DOA 模型的性能与数据域、输入表征和输出方式密切相关。

- 在 BPSK 数据域中，原始 I/Q 输入的 IQ-ResNet 能较好利用调制信号的幅相特征。
- 在高斯数据域中，基于协方差矩阵输入的 ViT 和 REG-CNN 表现更稳定。
- 在七信源等复杂场景中，深度学习模型仍保持一定分辨能力，而 MUSIC 难以完成全源估计。
- 在毫米波雷达实测实验中，部分深度学习模型相对 MUSIC(FBSS) 工程参考结果具有较稳定的估计表现。

## 目录结构

```text
Graduation/
├── data/
│   ├── data_create/      # BPSK、高斯窄带、DCA1000风格仿真数据生成
│   └── data_process/     # 深度学习输入预处理
├── dl_models/            # IQ-ResNet、CNN、ViT、SPE-CNN、MLP等模型
├── train/
│   ├── bpsk/             # BPSK数据域训练脚本
│   └── gauss/            # 高斯窄带数据域训练脚本
├── test/
│   ├── tradition/        # MUSIC、ESPRIT等传统算法测试
│   └── *.py              # 深度学习模型仿真与实测测试脚本
├── utils/                # 指标计算、雷达数据读取、实测模型推理工具
├── external/             # doatools和参考DOA深度学习框架
├── docs/                 # 论文与项目文档
└── result/               # 模型权重与实验输出，本目录不提交Git
```

## 运行环境

本项目主要在以下环境中运行：

- 操作系统：Windows 11
- Python：3.11
- GPU：NVIDIA GPU，训练建议使用 CUDA
- 主要依赖：`numpy`、`scipy`、`matplotlib`、`pandas`、`scikit-learn`、`h5py`、`tqdm`、`torch`、`tensorflow/keras`、`doatools`

注意：部分脚本包含本机绝对路径。迁移到其他电脑或目录后，需要先检查并修改路径。

## 数据与模型文件

仓库只保存源码、文档和轻量配置文件。以下内容不提交到 GitHub：

- `data/IQ_Data/`
- `data/ViT/`
- `data/raw/`
- `result/`
- `*.npy`、`*.npz`、`*.bin`
- `*.pth`、`*.pt`、`*.h5`、`*.pkl`、`*.ckpt`

原因是这些文件通常是训练数据、原始雷达数据、模型权重或实验结果，体积较大，适合通过本地生成、网盘、Hugging Face 或 GitHub Release 等方式单独管理。

## 推荐复现实验顺序

### 1. 传统算法仿真

传统算法不需要训练模型，可直接运行 `test/tradition/` 下的脚本。
这些脚本用于测试 MUSIC、ESPRIT 等算法在不同 SNR、快拍数、信源角度间隔、相干信源和欠定场景下的性能。

### 2. BPSK 数据域实验

BPSK 数据域通常先运行`data\data_create\Generate_IQ_Data.py`生成离线数据，再运行`train/bpsk/`训练模型和`test`测试。
BPSK 数据域主要包含单信源、双信源和七信源场景，模型包括 IQ-ResNet、ViT、CNN、MLP 及迁移学习模型。

### 3. 高斯数据域实验

高斯数据域多数脚本采用在线重采样方式其代码在`data/data_create/`，不需要提前保存完整训练集。再运行`train/gauss/`训练模型和`test`测试。
高斯数据域主要测试 IQ-ResNet、ViT、REG-CNN、SPE-CNN、Learning-SPICE 和 MUSIC 在单信源、三信源、七信源场景下的表现。

### 4. 毫米波雷达实测实验

实测实验对应论文第五章。采集到 DCA1000 原始 `.bin` 文件后，需要先进行`radar_utils.py`的数据预处理，再运行测试代码
其中 `test_frame80.py` 用于单帧可视化对比，`test_frame100_summary.py` 用于统计 100 帧平均偏差和标准差。

## 实测硬件平台

实测平台由以下部分组成：
| 硬件/软件 | 作用 |
| --- | --- |
| IWR1443BOOST ES3.0 EVM | 毫米波雷达前端，完成信号发射、接收和 ADC 采样 |
| DCA1000 EVM | 高速采集 IWR1443 输出的 LVDS 原始 ADC 数据 |
| mmWave Studio | 配置 Profile、Chirp、Frame、ADC 输出和发射天线参数 |
| 千兆以太网 | DCA1000 向 PC 传输原始采样数据 |
| USB/UART | PC 与 IWR1443BOOST 通信，完成固件下载和雷达配置 |

DCA1000 采集前需要将 PC 网卡设置为：
- 静态 IP：`192.168.33.30`
- 子网掩码：`255.255.255.0`
- DCA1000 FPGA 默认 IP：`192.168.33.180`

实测数据处理流程为：
1. 读取 DCA1000 生成的 `.bin` 原始数据。
2. 重排为 `frame/chirp/rx/sample` 数据立方体。
3. 执行 Range FFT。
4. 选择目标距离单元。
5. 拼接 8 阵元虚拟 ULA。
6. 构造 SCM 协方差矩阵。
7. 输入 MUSIC、ESPRIT 和深度学习模型进行 DOA 估计。

## 常见注意事项

- 建议直接打开 `Graduation/` 作为 VS Code 工作区，避免看到外层仓库的 Git 状态。
- 运行脚本前确认当前工作目录为 `Graduation/`注意训练和测试的路径。
- 如果找不到数据或权重，先检查 `data/` 和 `result/` 下是否已经生成对应文件。
- 如果 CUDA 不可用，可将脚本中的 `device` 改为 `cpu`，但训练速度会明显变慢。
- 迁移学习脚本依赖已有基模型权重，不能在没有基模型的情况下单独运行。
- 重新训练时，随机种子、数据量和训练轮数变化可能导致结果与论文中略有差异。

## 论文信息

- 题目：基于深度学习的波达方向估计算法研究
- 英文题目：Direction-of-Arrival Estimation Algorithms Based on Deep Learning
- 作者：林宏锦
- 专业：电子信息工程
- 学校：福州大学

## 许可与说明

本仓库主要用于毕业设计代码整理、实验复现和学习交流。外部参考代码位于 `external/` 目录，其许可请以对应项目原始说明为准。
