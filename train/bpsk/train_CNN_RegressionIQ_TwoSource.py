import os

import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, ReLU
from keras.models import Sequential
from tensorflow.keras.initializers import GlorotNormal as glorot_normal
from tqdm import tqdm


def load_cnn_twosource_data(base_dir, split_name):
    snrs = np.arange(-25, 26, 5)
    x_list, y_list = [], []

    print(f"Loading {split_name} data across SNRs...")
    for snr in tqdm(snrs):
        x_path = os.path.join(base_dir, split_name, f"cnn_{split_name.lower()}_data_snr{snr}.npy")
        y_path = os.path.join(base_dir, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")
        x_list.append(np.load(x_path))
        y_list.append(np.load(y_path))

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def pit_sep_loss(y_true, y_pred):
    """PIT angle loss with extra center and spacing constraints."""

    def calc_pair_mse(y_t, y_p):
        return tf.reduce_mean(tf.square(y_t - y_p), axis=1)

    loss1 = calc_pair_mse(y_true, y_pred)
    y_pred_reversed = tf.reverse(y_pred, axis=[1])
    loss2 = calc_pair_mse(y_true, y_pred_reversed)
    pred_best = tf.where(tf.expand_dims(loss1 <= loss2, axis=1), y_pred, y_pred_reversed)

    true_center = tf.reduce_mean(y_true, axis=1)
    pred_center = tf.reduce_mean(pred_best, axis=1)
    true_sep = tf.abs(y_true[:, 1] - y_true[:, 0])
    pred_sep = tf.abs(pred_best[:, 1] - pred_best[:, 0])
    close_weight = 1.0 + 3.0 * tf.cast(true_sep <= (10.0 / 90.0), tf.float32)
    edge_weight = 1.0 + 1.5 * tf.cast(tf.reduce_max(tf.abs(y_true), axis=1) >= (75.0 / 90.0), tf.float32)
    pair_loss = tf.minimum(loss1, loss2)
    center_loss = tf.square(true_center - pred_center)
    sep_loss = tf.square(true_sep - pred_sep)
    bound_loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(y_pred) - 1.0)), axis=1)
    return edge_weight * (pair_loss + 0.2 * center_loss + 0.2 * close_weight * sep_loss + 0.1 * bound_loss)


def pit_rmse_deg(y_true, y_pred):
    loss1 = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    y_pred_reversed = tf.reverse(y_pred, axis=[1])
    loss2 = tf.reduce_mean(tf.square(y_true - y_pred_reversed), axis=1)
    return tf.sqrt(tf.reduce_mean(tf.minimum(loss1, loss2))) * 90.0


def sep_mae_deg(y_true, y_pred):
    y_pred_sorted = tf.sort(y_pred, axis=1)
    true_sep = tf.abs(y_true[:, 1] - y_true[:, 0]) * 90.0
    pred_sep = tf.abs(y_pred_sorted[:, 1] - y_pred_sorted[:, 0]) * 90.0
    return tf.reduce_mean(tf.abs(true_sep - pred_sep))


current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))

rho = 0.0
data_dir = os.path.join(root_path, 'data', 'IQ_Data', 'Two_Source', f'SCM_Two_Source_Rho{rho}')
save_folder = os.path.join(root_path, 'result', 'CNN', 'TwoSource')
os.makedirs(save_folder, exist_ok=True)

xTrain, yTrain_multihot = load_cnn_twosource_data(data_dir, 'Train')
xVal, yVal_multihot = load_cnn_twosource_data(data_dir, 'Val')

print("Converting multi-hot labels to sorted normalized angles...")


def extract_and_scale_angles(multihot_labels):
    indices = np.argsort(multihot_labels, axis=1)[:, -2:]
    angles = indices - 90
    angles = np.sort(angles, axis=1)
    return angles.astype(np.float32) / 90.0


def make_sample_weights(y_scaled):
    angles = y_scaled * 90.0
    sep = np.abs(angles[:, 1] - angles[:, 0])
    edge = (np.abs(angles[:, 0]) >= 75) | (np.abs(angles[:, 1]) >= 75)
    weights = np.ones(len(y_scaled), dtype=np.float32)
    weights += (sep <= 15).astype(np.float32) * 2.0
    weights += (sep <= 10).astype(np.float32) * 3.0
    weights += (sep <= 5).astype(np.float32) * 5.0
    weights += edge.astype(np.float32) * 1.5
    return weights / np.mean(weights)


yTrain = extract_and_scale_angles(yTrain_multihot)
yVal = extract_and_scale_angles(yVal_multihot)
train_weights = make_sample_weights(yTrain)
val_weights = make_sample_weights(yVal)

input_shape = xTrain.shape[1:]
model = Sequential([
    Conv2D(64, 3, activation=None, input_shape=input_shape, padding="same"), BatchNormalization(), ReLU(),
    Conv2D(64, 3, activation=None, padding="same"), BatchNormalization(), ReLU(),
    Conv2D(128, 3, activation=None, padding="same"), BatchNormalization(), ReLU(),
    Conv2D(128, 3, activation=None, padding="same"), BatchNormalization(), ReLU(),
    Flatten(),
    Dense(512, activation="relu"), Dropout(0.1),
    Dense(256, activation="relu"), Dropout(0.1),
    Dense(128, activation="relu"),
    Dense(2, activation="linear", kernel_initializer=glorot_normal())
])

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=opt, loss=pit_sep_loss, metrics=[pit_rmse_deg, sep_mae_deg], weighted_metrics=[])

print("Training improved two-source CNN regressor...")
model.fit(xTrain, yTrain, sample_weight=train_weights, epochs=80, batch_size=128, shuffle=True, validation_data=(xVal, yVal, val_weights), callbacks=[rlr])

save_path = os.path.join(save_folder, f'Model_CNN_RegressionIQ_TwoSource_rho{rho}.h5')
model.save(save_path)
print(f">>> Saved improved two-source regressor to: {save_path}")
