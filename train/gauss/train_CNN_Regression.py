#!/usr/bin/env python
# coding: utf-8

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, ReLU
from tensorflow.keras.initializers import GlorotNormal as glorot_normal
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import os

# 1. 加载包含 Snapshots 的新训练数据
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
Y_Labels = np.tile(angles, reps=(SNRs, 1))

print("正在执行物理级幅度归一化...")
for i in range(X_data.shape[0]):
    # 重构复数幅度大小，寻找矩阵最大值
    complex_mag = np.sqrt(X_data[i, :, :, 0]**2 + X_data[i, :, :, 1]**2)
    max_val = np.max(complex_mag)
    if max_val > 0:
        X_data[i, :, :, 0] /= max_val  # 缩放实部
        X_data[i, :, :, 1] /= max_val  # 缩放虚部


xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_Labels, test_size=0.1, random_state=42)

# 2. 构建回归 CNN 模型
input_shape = xTrain.shape[1:]
model = Sequential([
    Conv2D(64, 3, activation=None, input_shape=input_shape, padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(64, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"), BatchNormalization(), ReLU(),
    Flatten(),
    Dense(512, activation="relu"), Dropout(0.3),
    Dense(256, activation="relu"), Dropout(0.3),
    Dense(128, activation="relu"), Dropout(0.3),
    Dense(1, activation="linear", kernel_initializer=glorot_normal())
])

# 3. 编译与训练
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])

print("开始训练彻底净化的回归模型...")
model.fit(xTrain, yTrain, epochs=60, batch_size=64, shuffle=True, validation_data=(xVal, yVal), callbacks=[rlr])

save_path = os.path.join(save_folder, 'Model_CNN_Regression_8ULA_K1.h5')
model.save(save_path)
print(f">>> 回归模型已保存至: {save_path}")