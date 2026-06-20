import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.models import Sequential
from tqdm import tqdm


def load_cnn_sevensource_data(base_dir, split_name):
    snrs = [0, 5, 10, 15, 20]
    x_list, y_list = [], []

    print(f"Loading {split_name} seven-source Article data...")
    for snr in tqdm(snrs):
        x_path = os.path.join(base_dir, split_name, f"cnn_{split_name.lower()}_data_snr{snr}.npy")
        y_path = os.path.join(base_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")
        x_list.append(np.load(x_path))
        y_list.append(np.load(y_path))

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0).astype(np.float32)


current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))

rho = 0.0
data_dir = os.path.join(root_path, "data", "IQ_Data", "Seven_Source", f"SCM_Seven_Source_Article_Rho{rho}")
save_folder = os.path.join(root_path, "result", "CNN", "SevenSource")
os.makedirs(save_folder, exist_ok=True)

x_train, y_train = load_cnn_sevensource_data(data_dir, "Train")
x_val, y_val = load_cnn_sevensource_data(data_dir, "Val")

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
    Dense(181, activation="sigmoid"),
])

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=10, verbose=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"Training seven-source Article CNN with rho={rho}...")
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"])
model.fit(x_train, y_train, epochs=60, batch_size=128, shuffle=True, validation_data=(x_val, y_val), callbacks=[lr_scheduler])

save_path = os.path.join(save_folder, f"Model_CNN_ClassifyIQ_SevenSource_Article_rho{rho}.h5")
model.save(save_path)
print(f"Saved Article CNN to: {save_path}")
