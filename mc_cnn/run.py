#!/usr/bin/env python
# coding: utf8
"""
CPU-only execution for MC-CNN fast with frameworks and variants.
Supports automatic architecture detection for window sizes 7/11/13/15.

Profiling markers (stdout):
  - PROFILING_LIBRARY_IMPORT: time, mem_peak
  - PROFILING_MODEL_INIT: time, mem_peak
  - PROFILING_IA_FEATURES: time, mem_peak
  - PROFILING_NON_IA_LOOP: time, mem_peak

Structured metrics (JSON):
  - metrics_stages.json: per-stage time and memory
"""

import os
import re
import time
import json
import threading
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import psutil


def get_memory_usage_bytes() -> int:
    """Get current process RSS in bytes."""
    return psutil.Process().memory_info().rss


def bytes_to_mb(b: int) -> float:
    """Convert bytes to megabytes."""
    return b / (1024.0 * 1024.0)


class MemorySampler:
    """
    Background thread to capture peak RSS memory usage.
    
    Configurable via MCCNN_MEM_SAMPLE_SEC environment variable (default: 0.005s).
    """
    def __init__(self, interval_sec: float = None):
        if interval_sec is None:
            try:
                interval_sec = float(os.getenv("MCCNN_MEM_SAMPLE_SEC", "0.005"))
            except Exception:
                interval_sec = 0.005
                
        self.interval = max(0.0005, interval_sec)
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0

    def _run(self):
        """Background sampling loop."""
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """Start memory sampling."""
        self._peak = get_memory_usage_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mem_sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop memory sampling."""
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def peak_mb(self) -> float:
        """Get peak memory in MB."""
        return bytes_to_mb(self._peak)


def _num_threads() -> int:
    """
    Unified threading control.
    
    Default: 1 thread for reproducibility.
    Override with MCCNN_THREADS environment variable.
    """
    try:
        return max(1, int(os.getenv("MCCNN_THREADS", "1")))
    except Exception:
        return 1


def _resolve_onnx_path(model_path: str, model_name: Optional[str] = None) -> str:
    """
    Locate ONNX model file with fallback search.
    
    Priority:
    1. MCCNN_ONNX_PATH environment variable
    2. Next to model_path with .onnx suffix
    3. model_path directory / mc_cnn_fast.onnx
    4. Current directory / mc_cnn_fast.onnx
    5. Module directory / mc_cnn_fast.onnx
    """
    env_path = os.getenv("MCCNN_ONNX_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    mp = Path(model_path)
    if model_name is None:
        candidates = [
            mp.with_suffix(".onnx"),
            mp.parent / "mc_cnn_fast.onnx",
            Path.cwd() / "mc_cnn_fast.onnx",
            Path(__file__).resolve().parent / "mc_cnn_fast.onnx",
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
    else:
        return str(os.path.join(os.path.dirname(mp), model_name))

    return "mc_cnn_fast.onnx"  # Fallback


def _resolve_openvino_path(model_path: str, model_name: Optional[str] = None) -> str:
    """
    Locate OpenVINO IR (.xml) with fallback search.
    
    Priority:
    1. MCCNN_OPENVINO_PATH environment variable
    2. Next to model_path with .xml suffix
    3. model_path directory / mc_cnn_fast.xml
    4. Current directory / mc_cnn_fast.xml
    5. Module directory / mc_cnn_fast.xml
    """
    env_path = os.getenv("MCCNN_OPENVINO_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    mp = Path(model_path)
    if model_name is None:
        candidates = [
            mp.with_suffix(".xml"),
            mp.parent / "mc_cnn_fast.xml",
            Path.cwd() / "mc_cnn_fast.xml",
            Path(__file__).resolve().parent / "mc_cnn_fast.xml",
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
    else:
        return str(os.path.join(os.path.dirname(mp), model_name))

    return "mc_cnn_fast.xml"  # Fallback


def _ov_set_cpu_properties(core: "ov.Core", nt: int) -> None:
    """
    Configure OpenVINO CPU plugin properties.
    
    Tries multiple property sets for version compatibility.
    """
    candidates = [
        {"INFERENCE_NUM_THREADS": nt, "NUM_STREAMS": "1", "INFERENCE_PRECISION_HINT": "f32"},
        {"INFERENCE_NUM_THREADS": nt, "NUM_STREAMS": "1"},
        {"INFERENCE_NUM_THREADS": nt},
    ]
    
    for props in candidates:
        try:
            core.set_property("CPU", props)
            return
        except Exception:
            continue


def _ov_compile_for_shape(core: "ov.Core", model_path: str, h: int, w: int) -> "ov.CompiledModel":
    """
    Compile OpenVINO model for specific input shape.
    
    Handles input rank: 2 (HW), 3 (CHW), or 4 (NCHW).
    """
    m = core.read_model(model_path)
    rank = m.inputs[0].get_partial_shape().rank
    rank_len = rank.get_length() if rank.is_static else 2

    if rank_len == 2:
        target_shape = [h, w]
    elif rank_len == 3:
        target_shape = [1, h, w]
    elif rank_len == 4:
        target_shape = [1, 1, h, w]
    else:
        target_shape = [h, w]

    m.reshape({m.inputs[0]: target_shape})
    return core.compile_model(m, "CPU")


def _write_metrics_stages(framework: str, variant: str, model_path: str, data: dict) -> None:
    """Write per-stage metrics to metrics_stages.json."""
    out_dir = os.getenv("PANDORA_RUN_OUTPUT_DIR", "")
    if not out_dir:
        return
        
    try:
        p = Path(out_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        
        payload = dict(data)
        payload["framework"] = framework
        payload["variant"] = variant
        payload["model_path"] = model_path
        
        with open(p / "metrics_stages.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def import_libraries(framework: str, variant: str):
    """Import required libraries for specified framework."""
    modules: Dict[str, Any] = {}

    # PyTorch needed for PyTorch framework and torch-based variants
    need_torch = (framework == "pytorch") or (variant in ["baseline", "opt1", "opt2", "cpp", "cpp2"])
    
    if need_torch:
        import torch
        modules["torch"] = torch
        import torch.nn as nn
        modules["nn"] = nn
        import torch.nn.functional as F
        modules["F"] = F

    if framework == "onnx":
        import onnxruntime as ort
        modules["ort"] = ort
    elif framework == "openvino":
        import openvino as ov
        modules["ov"] = ov

    return modules


# -------------------------
# Dynamic MC-CNN network
# -------------------------

def _build_conv_blocks(nn_mod, n_convs: int, in_channels: int = 1, out_channels: int = 64):
    """
    Build sequential conv stack: Conv2d(1->64, k=3) + ReLU, repeated n_convs times.
    
    No ReLU after final conv layer.
    """
    layers: List[Any] = []
    for i in range(n_convs):
        layers.append(nn_mod.Conv2d(
            in_channels if i == 0 else out_channels, 
            out_channels, 
            kernel_size=3
        ))
        if i != n_convs - 1:
            layers.append(nn_mod.ReLU())
            
    return nn_mod.Sequential(*layers)


def _make_dynamic_net(nn_mod, F_mod, n_convs_unused: int):
    """
    Create dynamic MC-CNN network class.
    
    Returns: Class (not instance) that can be instantiated with conv count.
    """
    class FastMcCnnDynamic(nn_mod.Module):
        def __init__(self, n: int):
            super().__init__()
            self.num_conv_feature_maps = 64
            self.conv_blocks = _build_conv_blocks(nn_mod, n_convs=n, in_channels=1, out_channels=64)
            self._n = n

        def forward(self, sample, training: bool):
            if training:
                # Training mode: process left, positive, and negative patches
                left = self.conv_blocks(sample[:, 0:1, :, :])
                left = F_mod.normalize(left, p=2, dim=1)
                
                pos = self.conv_blocks(sample[:, 1:2, :, :])
                pos = F_mod.normalize(pos, p=2, dim=1)
                
                neg = self.conv_blocks(sample[:, 2:3, :, :])
                neg = F_mod.normalize(neg, p=2, dim=1)
                
                return left, pos, neg
            else:
                # Inference mode: process single image
                features = self.conv_blocks(sample.unsqueeze(0).unsqueeze(0))
                features = F_mod.normalize(features, p=2, dim=1)
                return features.squeeze(0).squeeze(0)

    return FastMcCnnDynamic


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, Any]:
    """
    Extract state dict from checkpoint with format handling.
    
    Supports:
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - state_dict directly
    """
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
            
    if isinstance(ckpt_obj, dict):
        return ckpt_obj  # Already state dict
        
    raise ValueError("Unsupported checkpoint format")


def _infer_conv_count_from_state_dict(state_dict: Dict[str, Any]) -> Optional[int]:
    """
    Infer number of conv layers from state dict.
    
    Counts conv_blocks.*.weight keys.
    Mapping: 3->window 7, 5->window 11, 6->window 13, 7->window 15
    """
    pat = re.compile(r"^conv_blocks\.(\d+)\.weight$")
    conv_indices = []
    
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            conv_indices.append(int(m.group(1)))
            
    if not conv_indices:
        return None
        
    return len(conv_indices)


def _window_from_conv_count(n_convs: int) -> int:
    """Calculate window size from conv count: window = 2 * n_convs + 1."""
    return 2 * n_convs + 1


def _build_pytorch_infer(modules: Dict[str, Any], model_path: str):
    """
    Build PyTorch inference function.
    
    Auto-detects conv count from checkpoint, with optional override via MCCNN_WINDOW_SIZE.
    Returns: (inference_function, conv_count)
    """
    torch = modules["torch"]
    nn = modules["nn"]
    F = modules["F"]

    # Set threading
    nt = _num_threads()
    try:
        torch.set_num_threads(nt)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
        
    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    state = _extract_state_dict(ckpt)

    # Optional window size override
    ws_hint = os.getenv("MCCNN_WINDOW_SIZE")
    conv_hint = None
    
    if ws_hint is not None:
        try:
            ws_hint = int(ws_hint)
            if ws_hint in (7, 11, 13, 15):
                conv_hint = ws_hint // 2
        except Exception:
            pass

    # Determine conv count to try
    conv_count = _infer_conv_count_from_state_dict(state)
    candidates = []
    
    if conv_hint is not None:
        candidates.append(conv_hint)
        
    if conv_count is not None and conv_count not in candidates:
        candidates.append(conv_count)
        
    # Try conv counts in priority order (7 for window 15 support)
    for c in (7, 6, 5, 3):
        if c not in candidates:
            candidates.append(c)

    # Try each candidate until one loads successfully
    last_error = None
    net = None
    chosen = None
    
    for c in candidates:
        try:
            NetCls = _make_dynamic_net(nn, F, c)
            net = NetCls(c)
            net.load_state_dict(state, strict=True)
            chosen = c
            break
        except Exception as e:
            last_error = e
            net = None
            continue

    if net is None:
        raise RuntimeError(
            f"Failed to load checkpoint (tried conv_counts={candidates}). "
            f"Last error: {last_error}"
        )

    net.to(device)
    net.eval()

    # Report detected window size
    ws = _window_from_conv_count(chosen)
    print(f"[INFO] Loaded PyTorch MC-CNN fast with {chosen} convs (window_size={ws}) from: {model_path}")

    def inference_func(img_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img_np.astype(np.float32, copy=False)).to(device=device)
        with torch.no_grad():
            feats = net(x, training=False)
        return feats.cpu().numpy()

    return inference_func, chosen


def _onnx_prepare_input(session: "ort.InferenceSession", x2d: np.ndarray) -> Tuple[str, np.ndarray]:
    """Prepare input array matching model's expected rank."""
    in0 = session.get_inputs()[0]
    name = in0.name
    shape = in0.shape
    rank = len(shape)

    if rank == 2:
        arr = x2d.astype(np.float32, copy=False)
    elif rank == 3:
        arr = x2d[np.newaxis, :, :].astype(np.float32, copy=False)
    elif rank == 4:
        arr = x2d[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
    else:
        # Fallback to 4D
        arr = x2d[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
        
    return name, arr


def _to_chw(feats: np.ndarray) -> np.ndarray:
    """
    Convert feature tensor to (C, H, W) layout.
    
    Accepts: (C, H, W), (1, C, H, W), (H, W, C), (1, H, W, C)
    """
    f = feats
    
    if f.ndim == 4:
        if f.shape[1] == 64:  # (1, C, H, W)
            return np.squeeze(f, axis=0)
        elif f.shape[-1] == 64:  # (1, H, W, C)
            f = np.squeeze(f, axis=0)
            return np.transpose(f, (2, 0, 1))
        else:  # Unknown 4D
            return np.squeeze(f, axis=0)
    elif f.ndim == 3:
        if f.shape[0] == 64:  # (C, H, W)
            return f
        elif f.shape[-1] == 64:  # (H, W, C)
            return np.transpose(f, (2, 0, 1))
        else:  # Unknown 3D
            return f
    else:
        raise ValueError(f"Unexpected feature tensor shape: {f.shape}")


# -------------------------
# Cost volume implementations
# -------------------------

def computes_cost_volume_mc_cnn_fast(
    modules: Dict[str, Any],
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Baseline cost volume: cosine similarity across channel dimension.
    
    Returns: (H, W, D)
    """
    torch = modules["torch"]
    nn = modules["nn"]

    disparity_range = np.arange(disp_min, disp_max + 1, dtype=int)

    lf = torch.from_numpy(left_features)   # (C, H, W)
    rf = torch.from_numpy(right_features)  # (C, H, W)
    H = lf.shape[1]
    W = lf.shape[2]
    cv = np.empty((len(disparity_range), W, H), dtype=np.float32)
    cv.fill(np.nan)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)  # Cosine over channel

    def point_interval(left_features_t: "torch.Tensor", right_features_t: "torch.Tensor", disp: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        _, _, nx_left = left_features_t.shape
        _, _, nx_right = right_features_t.shape
        left = (max(0 - disp, 0), min(nx_left - disp, nx_left))
        right = (max(0 + disp, 0), min(nx_right + disp, nx_right))
        return left, right

    with torch.no_grad():
        for disp in disparity_range:
            left_int, right_int = point_interval(lf, rf, int(disp))
            ind_d = int(disp - disp_min)
            sim = cos(
                lf[:, :, left_int[0] : left_int[1]],
                rf[:, :, right_int[0] : right_int[1]],
            )  # (H, valid_W)
            cv[ind_d, left_int[0] : left_int[1], :] = sim.cpu().numpy().T

    cv *= -1.0  # Similarity -> cost
    return np.swapaxes(cv, 0, 2)  # (H, W, D)


def computes_cost_volume_mc_cnn_fast_opt1(
    modules: Dict[str, Any],
    left_features: np.ndarray,   # (C, H, W)
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Cosine similarity: normalize features once, then dot product.
    
    Cost = -cosine
    """
    torch = modules["torch"]

    with torch.no_grad():
        lf = torch.from_numpy(left_features).permute(1, 2, 0).contiguous()   # (H, W, C)
        rf = torch.from_numpy(right_features).permute(1, 2, 0).contiguous()  # (H, W, C)
        eps = 1e-6
        lf = lf / torch.clamp(torch.linalg.vector_norm(lf, dim=2, keepdim=True), min=eps)
        rf = rf / torch.clamp(torch.linalg.vector_norm(rf, dim=2, keepdim=True), min=eps)

        H, W, C = lf.shape
        D = disp_max - disp_min + 1
        out = torch.full((D, H, W), float("nan"), dtype=lf.dtype, device=lf.device)

        for d in range(disp_min, disp_max + 1):
            di = d - disp_min
            l0 = max(0, -d)
            r0 = max(0,  d)
            width = W - abs(d)
            if width <= 0:
                continue
            sim = (lf[:, l0:l0 + width, :] * rf[:, r0:r0 + width, :]).sum(dim=2).neg_()  # (H, width)
            out[di, :, l0:l0 + width] = sim

    return out.permute(1, 2, 0).contiguous().numpy()


def computes_cost_volume_mc_cnn_fast_opt1_notorch(
    left_features: np.ndarray,   # (C, H, W)
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Cosine similarity (NumPy implementation).
    
    Normalize once, then dot product. Cost = -cosine.
    """
    lf = np.transpose(left_features, (1, 2, 0)).copy(order="C")   # (H, W, C)
    rf = np.transpose(right_features, (1, 2, 0)).copy(order="C")  # (H, W, C)

    eps = 1e-6
    lf = lf / np.clip(np.linalg.norm(lf, axis=2, keepdims=True), a_min=eps, a_max=None)
    rf = rf / np.clip(np.linalg.norm(rf, axis=2, keepdims=True), a_min=eps, a_max=None)

    H, W, C = lf.shape
    D = disp_max - disp_min + 1
    out = np.full((D, H, W), np.nan, dtype=np.float32)

    for d in range(disp_min, disp_max + 1):
        di = d - disp_min
        l0 = max(0, -d)
        r0 = max(0,  d)
        width = W - abs(d)
        if width <= 0:
            continue
        sim = -np.sum(lf[:, l0:l0 + width, :] * rf[:, r0:r0 + width, :], axis=2)  # (H, width)
        out[di, :, l0:l0 + width] = sim

    return np.transpose(out, (1, 2, 0)).copy(order="C")


def computes_cost_volume_mc_cnn_fast_opt2(
    modules: Dict[str, Any],
    left_features: np.ndarray,   # (C, H, W)
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Dot product assuming unit-norm features.
    
    Cost = -dot(left, right)
    """
    torch = modules["torch"]

    with torch.no_grad():
        lf = torch.from_numpy(left_features).permute(1, 2, 0).contiguous()   # (H, W, C)
        rf = torch.from_numpy(right_features).permute(1, 2, 0).contiguous()  # (H, W, C)

        H, W, C = lf.shape
        D = disp_max - disp_min + 1
        out = torch.full((D, H, W), float("nan"), dtype=lf.dtype, device=lf.device)

        for d in range(disp_min, disp_max + 1):
            di = d - disp_min
            l0 = max(0, -d)
            r0 = max(0,  d)
            width = W - abs(d)
            if width <= 0:
                continue
            sim = (lf[:, l0:l0 + width, :] * rf[:, r0:r0 + width, :]).sum(dim=2).neg_()  # (H, width)
            out[di, :, l0:l0 + width] = sim

    return out.permute(1, 2, 0).contiguous().numpy()


def computes_cost_volume_mc_cnn_fast_opt2_notorch(
    left_features: np.ndarray,   # (C, H, W)
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Dot product (NumPy implementation), assuming unit-norm features.
    
    Cost = -dot(left, right)
    """
    lf = np.transpose(left_features, (1, 2, 0)).copy(order="C")   # (H, W, C)
    rf = np.transpose(right_features, (1, 2, 0)).copy(order="C")  # (H, W, C)

    H, W, C = lf.shape
    D = disp_max - disp_min + 1
    out = np.full((D, H, W), np.nan, dtype=np.float32)

    for d in range(disp_min, disp_max + 1):
        di = d - disp_min
        l0 = max(0, -d)
        r0 = max(0,  d)
        width = W - abs(d)
        if width <= 0:
            continue
        sim = -np.sum(lf[:, l0:l0 + width, :] * rf[:, r0:r0 + width, :], axis=2)  # (H, width)
        out[di, :, l0:l0 + width] = sim

    return np.transpose(out, (1, 2, 0)).copy(order="C")


def computes_cost_volume_mc_cnn_fast_cpp(
    modules: Dict[str, Any],
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int):
    """C++ single kernel implementation (torch-based)."""
    torch = modules["torch"]
    from .cv_opt2_single_loader import computes_cost_volume_mc_cnn_fast_opt2_single_cpp

    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)

    with torch.no_grad():
        out = computes_cost_volume_mc_cnn_fast_opt2_single_cpp(lf, rf, disp_min, disp_max)  # (D,H,W)
        return out.permute(1, 2, 0).contiguous().numpy()  # (H,W,D)


def computes_cost_volume_mc_cnn_fast_cpp2(
    modules: Dict[str, Any],
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int):
    """C++ pixel-major kernel implementation (torch-based)."""
    torch = modules["torch"]
    from .cv_opt2_pixelmajor_loader import computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp

    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)

    with torch.no_grad():
        out = computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp(lf, rf, disp_min, disp_max)  # (H,W,D)
    return out.numpy()


def computes_cost_volume_mc_cnn_fast_cpp_notorch(left_features, right_features, disp_min, disp_max):
    """C++ single kernel implementation (numpy-only)."""
    from .cv_opt2_single_loader_notorch import computes_cost_volume_mc_cnn_fast_opt2_single_cpp as cpp_single_notorch

    if left_features.ndim != 3 or right_features.ndim != 3:
        raise ValueError("Features must be 3D (C,H,W)")
    if left_features.shape != right_features.shape:
        raise ValueError("Feature shapes must match")

    lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
    rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

    out_dhw = cpp_single_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return np.transpose(out_dhw, (1, 2, 0))  # (H,W,D)


def computes_cost_volume_mc_cnn_fast_cpp2_notorch(left_features, right_features, disp_min, disp_max):
    """C++ pixel-major kernel implementation (numpy-only)."""
    from .cv_opt2_pixelmajor_loader_notorch import computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp as cpp_pixelmajor_notorch

    if left_features.ndim != 3 or right_features.ndim != 3:
        raise ValueError("Features must be 3D (C,H,W)")
    if left_features.shape != right_features.shape:
        raise ValueError("Feature shapes must match")

    lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
    rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

    out_hwd = cpp_pixelmajor_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return out_hwd


# -------------------------
# Main execution
# -------------------------

def run_mc_cnn_fast(
    img_left: np.ndarray,
    img_right: np.ndarray,
    disp_min: int,
    disp_max: int,
    model_path: str,
    framework: str = "pytorch",
    variant: str = "baseline",
    provider: Optional[str] = "cpu_base",
    model_name: Optional[str] = None
) -> np.ndarray:
    """
    Compute cost volume using MC-CNN fast features.
    
    Args:
        img_left: Left image (H, W)
        img_right: Right image (H, W)
        disp_min: Minimum disparity (typically negative)
        disp_max: Maximum disparity (typically 0)
        model_path: Path to model weights
        framework: "pytorch", "onnx", or "openvino"
        variant: Implementation variant
        provider: ONNX provider ("cpu_base" or "openvino")
        model_name: Specific model file name
        
    Returns:
        Cost volume (H', W', D) as float32
    """
    # ---------------- Stage: Library import ----------------
    ms = MemorySampler().start()
    start_import = time.perf_counter()
    modules = import_libraries(framework, variant)
    time_import = time.perf_counter() - start_import
    ms.stop()
    
    print(f"PROFILING_LIBRARY_IMPORT: time={time_import:.4f}s, mem_peak={ms.peak_mb:.2f}MB, framework={framework}, variant={variant}")

    # ---------------- Stage: Model initialization ----------------
    ms = MemorySampler().start()
    start_init = time.perf_counter()

    H_in, W_in = int(img_left.shape[0]), int(img_left.shape[1])

    if framework == "pytorch":
        inference_func, conv_count = _build_pytorch_infer(modules, model_path)
        # conv_count logged in _build_pytorch_infer
        
    elif framework == "onnx":
        ort = modules["ort"]
        if ort is None:
            raise ImportError("onnxruntime not installed")

        model_path = _resolve_onnx_path(model_path, model_name)

        nt = _num_threads()
        so = ort.SessionOptions()
        so.intra_op_num_threads = nt
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if provider == "cpu_base":
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif provider == "openvino":
            providers = ["OpenVINOExecutionProvider"]
            provider_options = [{"num_of_threads": str(nt)}]
        else:
            warnings.warn(f"Provider {provider} not implemented; using CPUExecutionProvider")
            providers, provider_options = ["CPUExecutionProvider"], [{}]

        session = ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=provider_options)

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            name, arr = _onnx_prepare_input(session, img_np.astype(np.float32, copy=False))
            outs = session.run(None, {name: arr})
            return _to_chw(outs[0]).astype(np.float32, copy=False)

    elif framework == "openvino":
        ov = modules["ov"]
        if ov is None:
            raise ImportError("openvino not installed")

        xml_path = _resolve_openvino_path(model_path, model_name)
        onnx_path = _resolve_onnx_path(model_path, model_name)
        model_path_eff = xml_path if Path(xml_path).exists() else onnx_path

        nt = _num_threads()
        core = ov.Core()
        _ov_set_cpu_properties(core, nt)

        cm: "ov.CompiledModel" = _ov_compile_for_shape(core, model_path_eff, H_in, W_in)
        input_port = cm.inputs[0]
        input_ps = cm.input(0).get_partial_shape()
        r = input_ps.rank
        rlen = r.get_length() if r.is_static else 2

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            x = img_np.astype(np.float32, copy=False)

            if rlen == 2:
                arr = x
            elif rlen == 3:
                arr = x[np.newaxis, :, :]
            elif rlen == 4:
                arr = x[np.newaxis, np.newaxis, :, :]
            else:
                arr = x

            infer_request = cm.create_infer_request()
            infer_request.infer({input_port: arr})
            feats_np = infer_request.get_output_tensor(0).data
            
            return _to_chw(np.array(feats_np, copy=False)).astype(np.float32, copy=False)

    else:
        ms.stop()
        raise ValueError(f"Unsupported framework: {framework}")

    time_init = time.perf_counter() - start_init
    ms.stop()
    
    print(f"PROFILING_MODEL_INIT: time={time_init:.4f}s, mem_peak={ms.peak_mb:.2f}MB, framework={framework}, variant={variant}")

    # ---------------- Stage: Feature extraction ----------------
    def normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image to zero mean, unit std."""
        img = img.astype(np.float32, copy=False)
        mean = float(img.mean())
        std = float(img.std())
        
        if std == 0.0:
            std = 1.0
            
        return (img - mean) / std

    ms = MemorySampler().start()
    start_inf = time.perf_counter()
    
    left = normalize(img_left)
    right = normalize(img_right)
    left_features = inference_func(left)    # (64, H', W')
    right_features = inference_func(right)  # (64, H', W')
    
    time_inf = time.perf_counter() - start_inf
    ms.stop()
    
    print(f"PROFILING_IA_FEATURES: time={time_inf:.4f}s, mem_peak={ms.peak_mb:.2f}MB, framework={framework}, variant={variant}")

    # ---------------- Stage: Cost volume computation ----------------
    ms = MemorySampler().start()
    start_loop = time.perf_counter()
    
    # Route to appropriate cost volume implementation
    if variant == "opt1":
        cv = computes_cost_volume_mc_cnn_fast_opt1(modules, left_features, right_features, disp_min, disp_max)
    elif variant == "opt1_notorch":
        cv = computes_cost_volume_mc_cnn_fast_opt1_notorch(left_features, right_features, disp_min, disp_max)
    elif variant == "opt2":
        cv = computes_cost_volume_mc_cnn_fast_opt2(modules, left_features, right_features, disp_min, disp_max)
    elif variant == "opt2_notorch":
        cv = computes_cost_volume_mc_cnn_fast_opt2_notorch(left_features, right_features, disp_min, disp_max)
    elif variant == "cpp":
        cv = computes_cost_volume_mc_cnn_fast_cpp(modules, left_features, right_features, disp_min, disp_max)
    elif variant == "cpp2":
        cv = computes_cost_volume_mc_cnn_fast_cpp2(modules, left_features, right_features, disp_min, disp_max)
    elif variant == "cpp_notorch":
        cv = computes_cost_volume_mc_cnn_fast_cpp_notorch(left_features, right_features, disp_min, disp_max)
    elif variant == "cpp2_notorch":
        cv = computes_cost_volume_mc_cnn_fast_cpp2_notorch(left_features, right_features, disp_min, disp_max)
    else:
        cv = computes_cost_volume_mc_cnn_fast(modules, left_features, right_features, disp_min, disp_max)
        
    time_loop = time.perf_counter() - start_loop
    ms.stop()
    
    print(f"PROFILING_NON_IA_LOOP: time={time_loop:.4f}s, mem_peak={ms.peak_mb:.2f}MB")

    # ---------------- Write metrics ----------------
    _write_metrics_stages(
        framework=framework,
        variant=variant,
        model_path=model_path,
        data={
            "library_import_time": time_import,
            "model_init_time": time_init,
            "library_import_mem": ms.peak_mb,
            "model_init_mem": ms.peak_mb,
            "ia_features_time": time_inf,
            "ia_features_mem": ms.peak_mb,
            "non_ia_loop_time": time_loop,
            "non_ia_loop_mem": ms.peak_mb,
        },
    )

    return cv.astype(np.float32, copy=False)