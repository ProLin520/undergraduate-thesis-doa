import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.models import Sequential


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

angles_train = np.argmax(y_train_onehot, axis=1) - 90
angles_val = np.argmax(y_val_onehot, axis=1) - 90

classes = np.arange(-90, 91)
sigma = 4.0

diff_train = classes[np.newaxis, :] - angles_train[:, np.newaxis]
y_train_soft = np.exp(-(diff_train ** 2) / (2 * sigma ** 2))
y_train_soft /= np.sum(y_train_soft, axis=1, keepdims=True)

diff_val = classes[np.newaxis, :] - angles_val[:, np.newaxis]
y_val_soft = np.exp(-(diff_val ** 2) / (2 * sigma ** 2))
y_val_soft /= np.sum(y_val_soft, axis=1, keepdims=True)

model = Sequential([
    Conv2D(64, 3, activation=None, input_shape=x_train.shape[1:], padding="valid"),
    BatchNormalization(),
    ReLU(),
    Conv2D(64, 2, activation=None, padding="valid"),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, 2, activation=None, padding="valid"),
    BatchNormalization(),
    ReLU(),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(181, activation="softmax"),
])

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=10, verbose=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"Training single-source CNN classifier with rho={rho}...")
model.compile(optimizer=optimizer, loss=tf.keras.losses.KLDivergence(), metrics=["accuracy"])
model.fit(
    x_train,
    y_train_soft,
    epochs=60,
    batch_size=64,
    shuffle=True,
    validation_data=(x_val, y_val_soft),
    callbacks=[lr_scheduler],
)

save_path = os.path.join(save_folder, f"Model_CNN_ClassificationIQ_8ULA_K1_rho{rho}.h5")
model.save(save_path)
print(f"Saved classifier to: {save_path}")
