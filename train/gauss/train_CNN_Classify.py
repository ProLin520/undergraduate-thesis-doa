import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, ReLU
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os

# ==========================================
# 1. 加载数据
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
data_file = os.path.join(root_path, 'data', 'CNN', 'CNN_M8_K1', 'TRAIN_DATA_8ULA_K1_low_SNR_res1_3D_90deg_Snapshots.h5')

save_folder = os.path.join(root_path, 'result', 'CNN', 'SingleSource')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

f1 = h5py.File(data_file, 'r')

angles = np.transpose(np.array(f1['angles']))
Ry_sam = np.array(f1['sam'])
[SNRs, n, chan, M, N] = Ry_sam.shape

X_data0 = Ry_sam.swapaxes(2, 4)
X_data = X_data0.reshape([SNRs * n, N, M, chan])

# 1. 使用高斯软标签 (Gaussian Soft Labels)
# 不再使用非黑即白的 One-Hot，而是生成服从高斯分布的概率标签
Y_angles_flat = np.tile(angles, reps=(SNRs, 1)).flatten()
classes = np.arange(-90, 91)  # 物理角度类别网格 (-90 到 90)

# 设置高斯分布的宽容度 sigma (标准差)。
# sigma 越大，标签越平滑。对于 1 度分辨率，sigma 设为 2.0 左右效果最好。
sigma = 2.0

# 利用广播机制计算所有样本到所有 181 个类别的物理距离平方
diff = classes[np.newaxis, :] - Y_angles_flat[:, np.newaxis]
# 计算高斯概率
Y_soft = np.exp(-(diff**2) / (2 * sigma**2))
# 归一化，确保每个样本的 181 个概率加起来等于 1
Y_soft = Y_soft / np.sum(Y_soft, axis=1, keepdims=True)

# 2. 物理级归一化
print("物理级最大值归一化...")
for i in range(X_data.shape[0]):
    complex_mag = np.sqrt(X_data[i, :, :, 0]**2 + X_data[i, :, :, 1]**2)
    max_val = np.max(complex_mag)
    if max_val > 0:
        X_data[i, :, :, 0] /= max_val
        X_data[i, :, :, 1] /= max_val

xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_soft, test_size=0.1, random_state=42)

# 3. 构建分类 CNN 模型
model = Sequential([
    Conv2D(64, 3, activation=None, input_shape=xTrain.shape[1:], padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(64, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Flatten(),
    Dense(512, activation="relu"), Dropout(0.3),
    Dense(256, activation="relu"), Dropout(0.3),
    Dense(128, activation="relu"), Dropout(0.3),
    Dense(181, activation="softmax")  # 输出依然是 181 个类别的概率
])

# 4. 编译与训练
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# 使用 KL 散度 (KLDivergence) 或 CategoricalCrossentropy
# 因为现在的目标标签 Y_soft 也是一个概率分布，用 KL 散度拟合两个分布是最严谨的数学做法
print("开始训练高斯软标签分类模型...")
model.compile(optimizer=opt, loss=tf.keras.losses.KLDivergence(), metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=60, batch_size=64, shuffle=True, validation_data=(xVal, yVal), callbacks=[rlr])

save_path = os.path.join(save_folder, 'Model_CNN_Classification_8ULA_K1.h5')
model.save(save_path)
print(f">>> 分类模型已保存至: {save_path}")