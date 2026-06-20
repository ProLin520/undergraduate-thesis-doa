import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRADITION_DATA_DIR = PROJECT_ROOT / "result" / "test_data" / "tradition"
BPSK_DATA_DIR = PROJECT_ROOT / "result" / "test_data" / "bpsk"
GAUSS_DATA_DIR = PROJECT_ROOT / "result" / "test_data" / "gauss"


def ensure_dir(path):
    path = Path(os.fspath(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(records, path):
    path = Path(os.fspath(path))
    ensure_dir(path.parent)
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    df.to_csv(path, index=False)
    return path


def sort_angles(x):
    arr = np.asarray(x)
    return np.sort(arr, axis=-1)


def align_pred_true(pred, true):
    pred_sorted = sort_angles(pred)
    true_sorted = sort_angles(true)
    return pred_sorted, true_sorted


def abs_error(pred, true):
    pred_sorted, true_sorted = align_pred_true(pred, true)
    return np.abs(pred_sorted - true_sorted)


def rmse_deg(pred, true):
    errors = abs_error(pred, true)
    return float(np.sqrt(np.mean(errors ** 2)))


def mae_deg(pred, true):
    errors = abs_error(pred, true)
    return float(np.mean(errors))


def recall_at(pred, true, threshold):
    errors = abs_error(pred, true)
    return float(np.mean(errors < threshold))


def full_success_at(pred, true, threshold):
    errors = abs_error(pred, true)
    if errors.ndim <= 1:
        return float(np.all(errors < threshold))
    return float(np.mean(np.all(errors < threshold, axis=1)))


def error_quantiles(pred, true):
    errors = abs_error(pred, true).reshape(-1)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "p50": float(np.percentile(errors, 50)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max_error": float(np.max(errors)),
    }


def nearest_value(values, target):
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return None
    value = arr[np.argmin(np.abs(arr - target))]
    return value.item() if hasattr(value, "item") else value


def first_x_reach_threshold(x_list, y_list, threshold):
    for x, y in zip(x_list, y_list):
        if y >= threshold:
            return x
    return None
