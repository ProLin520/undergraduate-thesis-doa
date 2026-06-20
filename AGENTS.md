# AGENTS.md

## 项目概览

本项目是本科毕业设计相关的 DOA（Direction of Arrival，波达方向）估计实验工程，主题集中在阵列信号处理与深度学习 DOA 估计。代码覆盖数据生成、模型训练、算法评估、传统方法对比和实测雷达数据处理。

项目当前作为独立 Git 仓库管理，根目录是 `Graduation/`，远程仓库为 `https://github.com/ProLin520/undergraduate-thesis-doa.git`。大规模数据、模型权重和实验结果被 `.gitignore` 排除，不应直接提交到 GitHub。

## 目录结构

- `data/data_create/`：仿真数据生成。包含 BPSK IQ 数据、SCM 协方差矩阵数据、Gaussian/ULA 在线生成数据、角度集合生成和 DCA1000 风格雷达仿真数据。
- `data/data_process/`：深度学习输入预处理。把协方差矩阵转换为 CNN/ViT 所需张量格式。
- `dl_models/`：核心深度学习模型定义。包括 CNN、IQ-ResNet、ViT、SPE-CNN、Learning-SPICE MLP 和多源后处理工具。
- `train/bpsk/`：基于离线 BPSK/IQ/SCM 数据集的训练脚本。
- `train/gauss/`：基于 Gaussian/ULA 在线生成数据的训练脚本，常用于三信源、七信源和迁移学习实验。
- `test/`：深度学习模型测试、对比实验、表格汇总和论文图表生成。
- `test/tradition/`：传统 DOA 方法评估，如 MUSIC、ESPRIT、FBSS、DCA 场景、SNR/snapshot/分辨率/欠定场景评估。
- `utils/`：通用工具，包括早停、误差指标、CSV 保存、实测雷达数据读取和统一模型推理。
- `external/doatools/`：第三方 doatools.py，用于 MUSIC、ESPRIT、阵列模型、CRB 等传统 DOA 工具。
- `external/DOA_est_Master-master/`：参考框架，提供 ULA/UCA 数据生成、SPE-CNN、ASL、Learning-SPICE、SubspaceNet、ViT 迁移学习等实现。
- `result/`：模型权重、图表、CSV 和实验输出目录。该目录被 Git 忽略。
- `data/IQ_Data/`、`data/ViT/`、`data/raw/`：生成数据和原始数据目录。均被 Git 忽略。

## 核心任务与数据流

项目主要支持两类实验线：

1. BPSK/IQ 离线数据实验
   - `data/data_create/Generate_IQ_Data.py` 生成 IQ 时域数据，底层使用 BPSK 调制、升余弦滤波、ULA 导向矢量、可选阵列误差 `rho` 和加性复高斯噪声。
   - `data/data_create/Generate_IQ_SCMData.py` 基于 IQ 数据生成 SCM 表示，分别保存 CNN 输入、ViT 输入和标签。
   - `train/bpsk/` 读取 `.npy` 数据训练 IQ-ResNet、CNN、MLP、ViT 和迁移学习模型。

2. Gaussian/ULA 在线生成实验
   - `data/data_create/signal_datasets90.py` 和 `Create_k_source_dataset90.py` 在线生成 ULA 多信源样本，可输出 `y_t`、SCM、DOA 标签和空间谱标签。
   - `train/gauss/` 中的脚本通常每个 epoch 重新采样角度集合和 SNR，适合三信源、七信源、低 SNR、近间隔和阵列误差实验。
   - `test/test_snr_three_gauss.py`、`test/test_snr_seven_gauss.py` 等脚本统一加载多个模型，与 MUSIC 对比并保存 RMSE/成功率结果。

角度网格通常是 `-90` 到 `90` 度，步长 `1` 度，对应 181 维 one-hot/multi-hot 空间谱标签。默认阵列多为 `M=8` 的 ULA，信源数常见为 `K=1`、`K=2`、`K=3`、`K=7`。

## 主要模型

- `dl_models/CNN_model.py`
  - `CNN_Classify`：2 通道输入，输出 181 类角度网格。
  - `CNN_Regression`：复用 CNN 特征提取器，输出连续角度，可通过 `out_dim` 支持多信源回归。

- `dl_models/IQ_ResNet_model.py`
  - `IQ_ResNet`：面向 IQ 时域输入。典型输入形状为 `(batch, 1, 16, T)`，其中 16 来自 8 阵元的 I/Q 拼接。
  - 使用 1x3 残差块和全局平均池化，输出 181 维 logits。

- `dl_models/vit_model.py`
  - `VisionTransformer`：基于 SCM patch embedding 的 ViT。常与 `scm_embeding(M, 768)` 搭配。
  - 部分代码从 timm ViT 思路改写，并支持空间谱模式 `sp_mode`。
  - 注意该文件中存在 `from models.dl_model.grid_based_network import Grid_Based_network` 这类来自参考项目的导入，运行时需要保证 `external/DOA_est_Master-master` 或项目兼容路径在 `PYTHONPATH` 中。

- `dl_models/SPE_CNN.py`
  - `std_CNN` 和 `modified_CNN` 用于空间谱估计，继承 `Grid_Based_network`，输出 181 维空间谱。

- `dl_models/MLP.py`
  - `LearningSPICE_MLP` 和 `LearningSPICE_SP_MLP` 处理 SCM 向量化输入。
  - `scm_to_vec72` / `vec72_to_scm` 用于 8x8 Hermitian SCM 与 72 维上三角实虚部向量之间转换。

- `dl_models/embeding_layer.py`
  - `scm_embeding` 将 SCM 转换为 ViT token。
  - 包含 MUSIC 基线、批量 MUSIC、IQ 复数转换、单/三/七信源连续角度提取和 RMSE 计算。

## 评估与结果

- `utils/metrics_utils.py` 定义统一结果路径和指标：
  - `TRADITION_DATA_DIR = result/test_data/tradition`
  - `BPSK_DATA_DIR = result/test_data/bpsk`
  - `GAUSS_DATA_DIR = result/test_data/gauss`
  - 常用指标包括 RMSE、MAE、recall、full success、误差分位数。

- `utils/real_test.py` 封装实测数据上的模型加载与推理：
  - 支持 BPSK 组和 Gaussian 组模型规格。
  - 可统一调用 ViT、MLP、IQ-ResNet、CNN、SPE-CNN、Learning-SPICE、MUSIC/ESPRIT。

- `utils/radar_utils.py` 处理 DCA1000 风格雷达数据：
  - 默认参数：`NUM_CHIRPS=60`、`NUM_RX=4`、`NUM_SAMPLES=256`、`NUM_FRAMES=100`。
  - 实测模式会从 3 发射通道中取 TX0 和 TX2，拼接为 8 阵元虚拟 ULA。
  - 仿真模式和实测模式的数据处理路径不同，调用时注意 `is_simulation`。

- `test/build_test_summary_tables.py` 汇总 BPSK/Gaussian 实验 CSV，生成可用于论文表格的数据。

## 运行注意事项

- 建议在 VS Code 中打开 `D:\Python\Project\doa_estimation\Graduation` 作为工作区；如果脚本使用 `Graduation.xxx` 导入，则需要把父目录 `D:\Python\Project\doa_estimation` 加入 `PYTHONPATH`。
- 很多脚本包含 Windows 绝对路径，例如 `D:\Python\Project\doa_estimation\Graduation\...`。迁移机器或目录后，需要先检查并改成当前路径。
- 部分脚本手动把 `external/DOA_est_Master-master` 插入 `sys.path`，这是为了复用参考框架中的数据生成和 grid-based 网络代码。
- 训练脚本默认优先使用 CUDA；无 GPU 时会退回 CPU，但大规模训练会非常慢。
- 训练和评估通常依赖被 Git 忽略的数据集和权重文件。若报找不到 `.npy`、`.pth`、`.h5`，先确认对应数据或权重是否已经生成或放在 `data/`、`result/` 下。
- 项目中部分中文注释显示为乱码，原因应是历史文件编码不一致。改动这些文件时避免无关的大规模重编码，除非明确要做编码清理。

## Git 与文件管理约定

- 当前仓库只提交源码、配置、文档和轻量脚本。
- 不提交以下内容：
  - `data/IQ_Data/`
  - `data/ViT/`
  - `data/raw/`
  - `result/`
  - `*.npy`、`*.npz`、`*.bin`
  - `*.pth`、`*.pt`、`*.h5`、`*.pkl`、`*.onnx`、`*.ckpt`
  - `__pycache__/`、`*.pyc`
- 如需分享大数据或权重，应使用网盘、Hugging Face、Release artifact 或其他外部存储，不要直接推到 GitHub。
- 该仓库嵌套在外层 `doa_estimation` 目录中；VS Code 如果打开外层目录，会看到外层仓库的 Git 状态，不代表 `Graduation` 仓库状态。检查本项目状态请在 `Graduation/` 中运行 `git status`。

## 常见命令

```powershell
# 查看本项目 Git 状态
cd D:\Python\Project\doa_estimation\Graduation
git status -sb

# 推送代码到 GitHub
git push

# 运行单信源 BPSK IQ-ResNet 训练示例
python train\bpsk\train_IQ_ResNet.py

# 运行三信源 Gaussian SNR 对比示例
python test\test_snr_three_gauss.py

# 生成论文表格汇总数据
python test\build_test_summary_tables.py
```

## 后续 agent 工作建议

- 修改训练或测试脚本前，先确认当前实验线是 BPSK 离线数据还是 Gaussian 在线生成数据。
- 修改模型输入形状时，同时检查数据生成、预处理、训练脚本和测试脚本中的归一化逻辑。
- 多信源评估时通常需要先排序预测角度和真实角度，再计算 RMSE 或成功率。
- 任何涉及 `result/`、`data/IQ_Data/`、`data/ViT/` 的改动都应视为本地实验产物处理，不应直接纳入 Git。
- 如果新增脚本，优先使用相对项目根目录的路径，减少硬编码绝对路径。
