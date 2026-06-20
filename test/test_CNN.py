import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

print("==========================================")
print(" 正在加载 4 个模型 (旧版 vs IQ对齐新版)...")
print("==========================================")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_base = os.path.dirname(os.path.dirname(current_dir))

# 1. 模型路径定义
MODEL_CLA_OLD_PATH = os.path.join(project_base, 'Graduation', 'result', 'CNN', 'SingleSource', 'Model_CNN_Classification_8ULA_K1.h5')
MODEL_REG_OLD_PATH = os.path.join(project_base, 'Graduation', 'result', 'CNN', 'SingleSource', 'Model_CNN_Regression_8ULA_K1.h5')
MODEL_CLA_NEW_PATH = os.path.join(project_base, 'Graduation', 'result', 'CNN', 'SingleSource', 'Model_CNN_ClassificationIQ_8ULA_K1_rho0.0.h5')
MODEL_REG_NEW_PATH = os.path.join(project_base, 'Graduation', 'result', 'CNN', 'SingleSource', 'Model_CNN_RegressionIQ_8ULA_K1_rho0.0.h5')

# 使用 compile=False 避免加载自定义 Loss 时报错
model_cla_old = load_model(MODEL_CLA_OLD_PATH, compile=False)
model_reg_old = load_model(MODEL_REG_OLD_PATH, compile=False)
model_cla_new = load_model(MODEL_CLA_NEW_PATH, compile=False)
model_reg_new = load_model(MODEL_REG_NEW_PATH, compile=False)

# ==========================================
# 2. 读取测试数据
# ==========================================
filename2 = r'D:\Python\Project\doa_estimation\Graduation\data\CNN\CNN_M8_K1\TEST_DATA_8ULA_K1_min10dBSNR_T2000_3D_90deg_snr0.h5'
print(f"正在加载测试数据: {os.path.basename(filename2)}")
f2 = h5py.File(filename2, 'r')

GT_angles = np.transpose(np.array(f2['angles']))
Ry_sam_test = np.array(f2['sam'])

# ==========================================
# 3. 数据预处理 (构建双输入流 & 修复相位翻转)
# ==========================================
X_raw = Ry_sam_test.swapaxes(1, 3)
B = GT_angles.T  # 真实角度标签

# 为旧模型和新模型准备独立的数据副本
X_old = np.copy(X_raw)
X_new = np.copy(X_raw)

print("正在执行物理级归一化与阵列流形对齐...")
for i in range(X_raw.shape[0]):
    complex_mag = np.sqrt(X_raw[i, :, :, 0] ** 2 + X_raw[i, :, :, 1] ** 2)
    max_val = np.max(complex_mag)
    if max_val > 0:
        # ---- 旧模型预处理 ----
        X_old[i, :, :, 0] /= max_val
        X_old[i, :, :, 1] /= max_val
        # 相位保持原样

        # ---- 新模型预处理 (修复 MATLAB -> Python 的相位反转) ----
        # 1. 归一化实部 (实部不变)
        X_new[i, :, :, 0] /= max_val
        # 2. 取共轭虚部并归一化 (虚部取反)
        X_new[i, :, :, 1] = -X_new[i, :, :, 1] / max_val
        # 3. 取共轭相位并缩放至 [-1, 1] (相位取反)
        X_new[i, :, :, 2] = -X_new[i, :, :, 2] / np.pi

    # ==========================================
# 4. 模型预测
# ==========================================
print("正在执行预测推理...")
angles_grid = np.arange(-90, 91).astype(float)

# --- 4.1 旧模型推理 ---
A_reg_old = model_reg_old.predict(X_old, verbose=0)
cls_probs_old = model_cla_old.predict(X_old, verbose=0)
A_cla_old = np.sum(cls_probs_old * angles_grid, axis=1).reshape(-1, 1)

# --- 4.2 新模型推理 ---
A_reg_new = model_reg_new.predict(X_new, verbose=0) * 90.0
cls_probs_new = model_cla_new.predict(X_new, verbose=0)
A_cla_new = np.sum(cls_probs_new * angles_grid, axis=1).reshape(-1, 1)

# ==========================================
# 5. 误差计算与对比
# ==========================================
RMSE_reg_old = round(np.sqrt(mean_squared_error(A_reg_old, B)), 4)
RMSE_cla_old = round(np.sqrt(mean_squared_error(A_cla_old, B)), 4)
RMSE_reg_new = round(np.sqrt(mean_squared_error(A_reg_new, B)), 4)
RMSE_cla_new = round(np.sqrt(mean_squared_error(A_cla_new, B)), 4)

print(f"\n=====================================")
print(f" 🎯 评估结果 (Test RMSE):")
print(f" -------------------------------------")
print(f" [旧版模型 - 纯高斯噪声训练]")
print(f" 📉 Regression   : {RMSE_reg_old} degrees")
print(f" 📊 Classification : {RMSE_cla_old} degrees")
print(f" \n [新版模型 - IQ-ResNet波形对齐训练]")
print(f" 📉 Regression (IQ): {RMSE_reg_new} degrees")
print(f" 📊 Classification(IQ): {RMSE_cla_new} degrees")
print(f"=====================================\n")

# ==========================================
# 6. 可视化 (论文级对比图)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------
# 绘图 1：四模型误差对比子图
# ------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True, sharey=True)

# 子图 1：分类模型误差 (旧版 vs 新版)
axes[0].scatter(np.arange(B.shape[0]), A_cla_old[:, 0] - B[:, 0],
                label=f'Old Classify (RMSE: {RMSE_cla_old})', color='orange', alpha=0.6, marker='o', s=20)
axes[0].scatter(np.arange(B.shape[0]), A_cla_new[:, 0] - B[:, 0],
                label=f'New Classify IQ (RMSE: {RMSE_cla_new})', color='blue', alpha=0.7, marker='s', s=20)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[0].set_title('Estimation Error: Classification Models (Old vs IQ-Aligned)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Error (Degrees)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)

# 子图 2：回归模型误差 (旧版 vs 新版)
axes[1].scatter(np.arange(B.shape[0]), A_reg_old[:, 0] - B[:, 0],
                label=f'Old Regression (RMSE: {RMSE_reg_old})', color='lightcoral', alpha=0.6, marker='+', s=40)
axes[1].scatter(np.arange(B.shape[0]), A_reg_new[:, 0] - B[:, 0],
                label=f'New Regression IQ (RMSE: {RMSE_reg_new})', color='darkred', alpha=0.8, marker='*', s=40)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[1].set_title('Estimation Error: Regression Models (Old vs IQ-Aligned)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Error (Degrees)', fontsize=12)
axes[1].set_xlabel('Sample Index (Angle Sweep)', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.ylim(-15, 15)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 绘图 2：真实值 vs 预测值 信号跟踪图 (拆分上下子图)
# ------------------------------------------
fig2, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex=True, sharey=True)

# ================= 子图 1：分类模型跟踪对比 =================
axes2[0].plot(np.arange(B.shape[0]), B[:, 0], label='True Signal (Ground Truth)', color='black', linewidth=2.5, zorder=1)

# 旧分类：橙色空心圆
axes2[0].scatter(np.arange(B.shape[0]), A_cla_old[:, 0], label='Old Classify (Gaussian Trained)',
                 facecolors='none', edgecolors='orange', s=45, linewidths=1.5, zorder=2)
# 新分类：蓝色实心方块 (高对比)
axes2[0].scatter(np.arange(B.shape[0]), A_cla_new[:, 0], label='New Classify IQ (BPSK Trained)',
                 color='blue', marker='s', s=30, zorder=3)

axes2[0].set_title('True vs Estimated DoA: Classification Models', fontsize=14, fontweight='bold')
axes2[0].set_ylabel('DoA (Degrees)', fontsize=12)
axes2[0].legend(loc='upper left', framealpha=0.9, edgecolor='black')
axes2[0].grid(True, linestyle='--', alpha=0.6)

# ================= 子图 2：回归模型跟踪对比 =================
axes2[1].plot(np.arange(B.shape[0]), B[:, 0], label='True Signal (Ground Truth)', color='black', linewidth=2.5, zorder=1)

# 旧回归：粉红色加号
axes2[1].scatter(np.arange(B.shape[0]), A_reg_old[:, 0], label='Old Regression (Gaussian Trained)',
                 color='hotpink', marker='+', s=60, linewidths=1.5, zorder=2)
# 新回归：深红色星星 (高对比)
axes2[1].scatter(np.arange(B.shape[0]), A_reg_new[:, 0], label='New Regression IQ (BPSK Trained)',
                 color='darkred', marker='*', s=50, zorder=3)

axes2[1].set_title('True vs Estimated DoA: Regression Models', fontsize=14, fontweight='bold')
axes2[1].set_ylabel('DoA (Degrees)', fontsize=12)
axes2[1].set_xlabel('Sample Index (Angle Sweep)', fontsize=12)
axes2[1].legend(loc='upper left', framealpha=0.9, edgecolor='black')
axes2[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()