"""
Shared inference pipeline for Chapter 6 real-data evaluation.
Centralizes model loading, preprocessing, and prediction.
Supports BPSK-domain and Gauss-domain model groups.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import tensorflow as tf
except Exception:
    tf = None

# --- path setup ---
_GRADUATION_DIR = Path(__file__).resolve().parents[1]
_PROJECT_BASE = _GRADUATION_DIR.parent
for _p in [_PROJECT_BASE, _GRADUATION_DIR, _GRADUATION_DIR / "external"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.radar_utils import load_and_reshape, process_radar_data
from data.data_process.dl_preprocess import preprocess_for_vit
from doatools import model as doa_model
from doatools import estimation

from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding
from dl_models.IQ_ResNet_model import IQ_ResNet
from dl_models.CNN_model import CNN_Regression
from dl_models.SPE_CNN import std_CNN
from dl_models.embeding_layer import get_continuous_angle
from dl_models.MLP import LearningSPICE_SP_MLP

# ============================================================
# Model specification base
# ============================================================


class ModelSpec:
    name: str = ""
    weight_relpath: str = ""

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def preprocess(self, X_i: np.ndarray, R_i: np.ndarray, device: torch.device) -> Any:
        raise NotImplementedError

    def postprocess(self, output: Any) -> float:
        raise NotImplementedError


# ============================================================
# Shared helpers
# ============================================================

def _get_weight_path(relpath: str) -> str:
    return os.path.join(_GRADUATION_DIR, "result", relpath)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _make_scm_features(R_i: np.ndarray):
    """Build 3ch and 2ch SCM feature arrays from covariance matrix."""
    R_c = np.conj(R_i)
    R_n = R_c / (np.max(np.abs(R_c)) + 1e-8)
    feat3 = np.stack([R_n.real, R_n.imag, np.angle(R_n) / np.pi], axis=-1).astype(np.float32)
    feat2 = np.stack([R_n.real, R_n.imag], axis=-1).astype(np.float32)
    return feat3, feat2


def _match_input_shape(model, feat3, feat2):
    """Return the feature variant that matches the Keras model's input_shape."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    expected = tuple(input_shape[1:])

    candidates = {
        feat3.shape: feat3[np.newaxis, ...],
        np.transpose(feat3, (2, 0, 1)).shape: np.transpose(feat3, (2, 0, 1))[np.newaxis, ...],
        feat2.shape: feat2[np.newaxis, ...],
        np.transpose(feat2, (2, 0, 1)).shape: np.transpose(feat2, (2, 0, 1))[np.newaxis, ...],
        (feat3.size,): feat3.reshape(1, -1),
        (feat2.size,): feat2.reshape(1, -1),
    }
    return candidates.get(expected, feat3[np.newaxis, ...])


def _continuous_angle_from_scores(scores: np.ndarray, radius: int = 2) -> float:
    peak_idx = int(np.argmax(scores))
    left = max(0, peak_idx - radius)
    right = min(len(scores), peak_idx + radius + 1)
    weights = scores[left:right]
    angles = np.arange(left, right) - 90.0
    if np.sum(np.abs(weights)) < 1e-12:
        return float(peak_idx - 90)
    return float(np.sum(angles * weights) / np.sum(weights))


# ============================================================
# BPSK-domain model specs
# ============================================================

class ViT_IQ_Spec(ModelSpec):
    """ViT trained on BPSK IQ data (TransIQ) — 181-class classification."""
    name: str = "TransIQ"
    weight_relpath: str = "vit/vit_M_8_k_1_base/weight_base_bestIQ_rho0.0.pth"
    num_rx: int = 8
    embed_dim: int = 768

    def build_model(self):
        return VisionTransformer(
            embed_layer=scm_embeding(self.num_rx, self.embed_dim),
            embed_dim=self.embed_dim,
            out_dims=181,
            drop_ratio=0,
            attn_drop_ratio=0,
        )

    def preprocess(self, X_i, R_i, device):
        scm_input = preprocess_for_vit(R_i)
        if isinstance(scm_input, torch.Tensor):
            scm_input = scm_input.to(device)
        return scm_input

    def postprocess(self, output):
        if isinstance(output, torch.Tensor):
            return safe_float(torch.argmax(output, dim=1).item() - 90)
        return safe_float(np.argmax(np.asarray(output)) - 90)


class MLP_ClassifyIQ_Spec(ModelSpec):
    """Keras MLP classification on BPSK SCM data."""
    name: str = "MLP"
    weight_relpath: str = "MLP/SingleSource/Model_MLP_ClassificationIQ_8ULA_K1_rho0.0.h5"

    def build_model(self):
        if tf is None:
            return None
        return tf.keras.models.load_model(_get_weight_path(self.weight_relpath), compile=False)

    def preprocess(self, X_i, R_i, device):
        feat3, feat2 = _make_scm_features(R_i)
        model = getattr(self, "_cached_model", None)
        if model is None:
            return feat3[np.newaxis, ...]
        return _match_input_shape(model, feat3, feat2)

    def postprocess(self, output):
        return _continuous_angle_from_scores(np.asarray(output).reshape(-1))


class IQ_ResNet_BPSK_Spec(ModelSpec):
    """IQ-ResNet classification on BPSK raw IQ data."""
    name: str = "IQ-ResNet (BPSK)"
    weight_relpath: str = "IQ_ResNet/SingleSource/IQ_ResNet_SingleSource_rho0.0.pth"

    def build_model(self):
        return IQ_ResNet(num_classes=181)

    def preprocess(self, X_i, R_i, device):
        X_i_conj = np.conj(X_i)
        X_i_power_norm = X_i_conj / (np.sqrt(np.mean(np.abs(X_i_conj) ** 2)) + 1e-8)
        iq_matrix = np.vstack((np.real(X_i_power_norm), np.imag(X_i_power_norm)))
        return torch.tensor(iq_matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    def postprocess(self, output):
        return safe_float(torch.argmax(output, dim=1).item() - 90)


class CNN_ClassifyIQ_Spec(ModelSpec):
    """Keras CNN classification on BPSK SCM data."""
    name: str = "CNN-ClassifyIQ"
    weight_relpath: str = "CNN/SingleSource/Model_CNN_ClassificationIQ_8ULA_K1_rho0.0.h5"

    def build_model(self):
        if tf is None:
            return None
        return tf.keras.models.load_model(_get_weight_path(self.weight_relpath), compile=False)

    def preprocess(self, X_i, R_i, device):
        feat3, feat2 = _make_scm_features(R_i)
        model = getattr(self, "_cached_model", None)
        if model is None:
            return feat3[np.newaxis, ...]
        return _match_input_shape(model, feat3, feat2)

    def postprocess(self, output):
        return _continuous_angle_from_scores(np.asarray(output).reshape(-1))


# ============================================================
# Gauss-domain model specs
# ============================================================

class REG_CNN_Spec(ModelSpec):
    name: str = "REG-CNN"
    weight_relpath: str = "CNN/SingleSource/CNN_Regression_Gaussian_rho0.0.pth"

    def build_model(self):
        return CNN_Regression()

    def preprocess(self, X_i, R_i, device):
        M = R_i.shape[0]
        R_tensor = torch.tensor(R_i, dtype=torch.complex64, device=device)
        X = torch.zeros(1, 2, M, M, device=device)
        X[0, 0] = R_tensor.real
        X[0, 1] = -R_tensor.imag
        max_val = torch.max(torch.abs(X.view(1, -1)), dim=1)[0].view(1, 1, 1, 1)
        return X / (max_val + 1e-8)

    def postprocess(self, output):
        return safe_float(output.item())


class SPE_CNN_Spec(ModelSpec):
    name: str = "SPE-CNN"
    weight_relpath: str = "CNN/SingleSource/SPE_CNN_Gaussian_8ULA_K1_rho0.0.pth"
    num_rx: int = 8

    def build_model(self):
        return std_CNN(3, self.num_rx, 181, sp_mode=True, start_angle=-90, end_angle=90)

    def preprocess(self, X_i, R_i, device):
        M = R_i.shape[0]
        R_tensor = torch.tensor(R_i, dtype=torch.complex64, device=device).conj()
        X_spe = torch.zeros(1, 3, M, M, device=device)
        X_spe[0, 0] = R_tensor.real
        X_spe[0, 1] = R_tensor.imag
        X_spe[0, 2] = R_tensor.angle() / torch.pi
        max_spe = torch.max(torch.abs(R_tensor.view(-1)))
        X_spe[0, 0] /= (max_spe + 1e-8)
        X_spe[0, 1] /= (max_spe + 1e-8)
        return X_spe

    def postprocess(self, output):
        return safe_float(get_continuous_angle(output, radius=2).item())


class ViT_Spec(ModelSpec):
    name: str = "ViT"
    weight_relpath: str = "vit/vit_M_8_k_1_base/weight_base_best_snr020.pth"
    num_rx: int = 8
    embed_dim: int = 768

    def build_model(self):
        return VisionTransformer(
            embed_layer=scm_embeding(self.num_rx, self.embed_dim),
            embed_dim=self.embed_dim,
            out_dims=1,
            drop_ratio=0,
            attn_drop_ratio=0,
        )

    def preprocess(self, X_i, R_i, device):
        scm_input = preprocess_for_vit(R_i)
        if isinstance(scm_input, torch.Tensor):
            scm_input = scm_input.to(device)
        return scm_input

    def postprocess(self, output):
        if isinstance(output, torch.Tensor):
            return safe_float(output.detach().cpu().numpy().squeeze())
        return safe_float(np.squeeze(output))


class IQ_ResNet_Gauss_Spec(ModelSpec):
    name: str = "IQ-ResNet (Gauss)"
    weight_relpath: str = "IQ_ResNet/SingleSource/IQ_ResNet_Gaussian_rho0.0.pth"

    def build_model(self):
        return IQ_ResNet(num_classes=181)

    def preprocess(self, X_i, R_i, device):
        X_i_tensor = torch.tensor(X_i, dtype=torch.complex64, device=device).unsqueeze(0).conj()
        inputs = torch.cat([X_i_tensor.real, X_i_tensor.imag], dim=1).unsqueeze(1).float()
        rms_val = torch.sqrt(torch.mean(inputs ** 2, dim=(2, 3), keepdim=True))
        return inputs / (rms_val + 1e-8)

    def postprocess(self, output):
        return safe_float(get_continuous_angle(output, radius=2).item())


class LearningSPICE_Spec(ModelSpec):
    name: str = "Learning-SPICE"
    weight_relpath: str = "MLP/SingleSource/LearningSPICE_Gaussian_rho0.0.pth"
    num_rx: int = 8

    def build_model(self):
        return LearningSPICE_SP_MLP(M=self.num_rx, out_dim=181)

    def preprocess(self, X_i, R_i, device):
        M = self.num_rx
        R_tensor = torch.tensor(np.conj(R_i), dtype=torch.complex64, device=device).unsqueeze(0)
        max_val = torch.max(torch.abs(R_tensor.reshape(1, -1)), dim=1)[0].reshape(1, 1, 1)
        R_tensor = R_tensor / (max_val + 1e-8)
        B_val, M_val, _ = R_tensor.shape
        triu_idx = torch.triu_indices(M_val, M_val, device=device)
        R_triu = R_tensor[:, triu_idx[0], triu_idx[1]]
        return torch.cat([R_triu.real, R_triu.imag], dim=1)

    def postprocess(self, output):
        return safe_float(get_continuous_angle(output, radius=2).item())


# ============================================================
# Group definitions
# ============================================================

def get_bpsk_specs() -> List[ModelSpec]:
    return [
        ViT_IQ_Spec(),
        MLP_ClassifyIQ_Spec(),
        IQ_ResNet_BPSK_Spec(),
        CNN_ClassifyIQ_Spec(),
    ]


def get_gauss_specs() -> List[ModelSpec]:
    return [
        REG_CNN_Spec(),
        SPE_CNN_Spec(),
        IQ_ResNet_Gauss_Spec(),
        LearningSPICE_Spec(),
        ViT_Spec(),
    ]


# ============================================================
# Traditional methods (shared)
# ============================================================

def apply_fbss(R: np.ndarray) -> np.ndarray:
    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    return 0.5 * (R + J @ np.conj(R) @ J)


def get_music_peak_and_spectrum(
    R: np.ndarray, num_rx: int = 8, use_fbss: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    if use_fbss:
        R = apply_fbss(R)
    ula = doa_model.UniformLinearArray(num_rx, 0.5)
    grid = estimation.FarField1DSearchGrid(
        start=np.deg2rad(-90), stop=np.deg2rad(70), size=1601
    )
    music_estimator = estimation.MUSIC(ula, 1.0, grid)
    res, est, sp_val = music_estimator.estimate(R, 1, return_spectrum=True)
    angle = float(np.degrees(est.locations[0])) if res else np.nan
    theta_range = np.linspace(-90, 70, 1601)
    return angle, theta_range, sp_val


def get_esprit_peak(R: np.ndarray, use_fbss: bool = False) -> float:
    if use_fbss:
        R = apply_fbss(R)
    try:
        esprit_estimator = estimation.Esprit1D(1.0)
        res, est = esprit_estimator.estimate(R, 1, 0.5)
        return float(np.degrees(est.locations[0])) if res else np.nan
    except Exception:
        return np.nan


# ============================================================
# Model loading and inference
# ============================================================

def load_models(
    specs: List[ModelSpec], device: torch.device = None
) -> Dict[str, Any]:
    """Load models from specs. Returns dict of name -> (model, spec)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded: Dict[str, Tuple[Any, ModelSpec]] = {}
    for spec in specs:
        weight_path = _get_weight_path(spec.weight_relpath)
        if not os.path.exists(weight_path):
            print(f"  [跳过] {spec.name}: 权重不存在 {weight_path}")
            continue
        try:
            model = spec.build_model()
            if model is None:
                print(f"  [跳过] {spec.name}: 模型构建失败")
                continue

            if isinstance(spec, (MLP_ClassifyIQ_Spec, CNN_ClassifyIQ_Spec)):
                spec._cached_model = model
            else:
                model.load_state_dict(
                    torch.load(weight_path, map_location=device, weights_only=True)
                )
                model.to(device)
                model.eval()

            loaded[spec.name] = (model, spec)
            print(f"  [OK] {spec.name}")
        except Exception as e:
            print(f"  [失败] {spec.name}: {e}")

    return loaded


def predict_dl_models(
    X_i: np.ndarray,
    R_i: np.ndarray,
    loaded_models: Dict[str, Tuple[Any, ModelSpec]],
    device: torch.device,
) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    for name, (model, spec) in loaded_models.items():
        try:
            if isinstance(spec, (MLP_ClassifyIQ_Spec, CNN_ClassifyIQ_Spec)):
                mlp_input = spec.preprocess(X_i, R_i, device)
                raw_output = model.predict(mlp_input, verbose=0)
                preds[name] = spec.postprocess(raw_output[0])
            else:
                with torch.no_grad():
                    model_input = spec.preprocess(X_i, R_i, device)
                    raw_output = model(model_input)
                    preds[name] = spec.postprocess(raw_output)
        except Exception as e:
            print(f"  {name} 推理失败: {e}")
            preds[name] = np.nan
    return preds


def estimate_one_frame(
    X_i: np.ndarray,
    R_i: np.ndarray,
    loaded_models: Dict[str, Tuple[Any, ModelSpec]],
    device: torch.device,
    num_rx: int = 8,
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    for name, use_fbss in [("MUSIC (Raw)", False), ("MUSIC (FBSS)", True)]:
        try:
            results[name], _, _ = get_music_peak_and_spectrum(
                R_i, num_rx=num_rx, use_fbss=use_fbss
            )
        except Exception as e:
            print(f"  {name} 失败: {e}")
            results[name] = np.nan

    for name, use_fbss in [("ESPRIT (Raw)", False), ("ESPRIT (FBSS)", True)]:
        results[name] = get_esprit_peak(R_i, use_fbss=use_fbss)

    results.update(predict_dl_models(X_i, R_i, loaded_models, device))
    return results
