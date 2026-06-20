import os

import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, ReLU
from keras.models import Sequential
from tensorflow.keras.initializers import GlorotNormal as glorot_normal
from tensorflow.keras.layers import GlobalAveragePooling2D


current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))

rho = 1.0
data_dir = os.path.join(root_path, "data", "IQ_Data", "Single_Source", f"SCM_Single_Source_Rho{rho}")

save_folder = os.path.join(root_path, "result", "CNN", "SingleSource")
os.makedirs(save_folder, exist_ok=True)

print("Loading single-source SCM data...")
x_train = np.load(os.path.join(data_dir, "Train", "cnn_train_data.npy"))
y_train_onehot = np.load(os.path.join(data_dir, "Train", "train_labels.npy"))
x_val = np.load(os.path.join(data_dir, "Val", "cnn_val_data.npy"))
y_val_onehot = np.load(os.path.join(data_dir, "Val", "val_labels.npy"))

print("Converting one-hot labels to normalized angles...")
y_train = (np.argmax(y_train_onehot, axis=1) - 90).astype(np.float32) / 90.0
y_val = (np.argmax(y_val_onehot, axis=1) - 90).astype(np.float32) / 90.0

model = Sequential([
    Conv2D(64, 3, activation=None, input_shape=x_train.shape[1:], padding="same"),
    BatchNormalization(),
    ReLU(),
    Conv2D(64, 3, activation=None, padding="same"),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, 3, activation=None, padding="same"),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, 3, activation=None, padding="same"),
    BatchNormalization(),
    ReLU(),
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="tanh", kernel_initializer=glorot_normal()),
])

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=10, verbose=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"Training single-source CNN regressor with rho={rho}...")
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])
model.fit(
    x_train,
    y_train,
    epochs=60,
    batch_size=64,
    shuffle=True,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler],
)

save_path = os.path.join(save_folder, f"Model_CNN_RegressionIQ_8ULA_K1_rho{rho}.h5")
model.save(save_path)
print(f"Saved regressor to: {save_path}")
