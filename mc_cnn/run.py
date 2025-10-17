#!/usr/bin/env python
# coding: utf8
#
# CPU-only execution for MC-CNN fast with frameworks and variants.
# Emits (stdout for fallback parsing):
#   - PROFILING_MODEL_INIT: time=...s, mem_peak=...MB
#   - PROFILING_IA_FEATURES: time=...s, mem_peak=...MB
#   - PROFILING_NON_IA_LOOP: time=...s, mem_peak=...MB
#
# Also writes structured per-stage metrics to:
#   <PANDORA_RUN_OUTPUT_DIR>/metrics_stages.json
# with:
#   - model_init_time, ia_features_time, non_ia_loop_time
#   - model_init_mem, ia_features_mem, non_ia_loop_mem
#   - framework, variant
#
# Notes:
# - All paths are CPU-only regardless of hardware availability.
# - Single-thread by default for stability (override with env MCCNN_THREADS).
# - ONNX and OpenVINO sessions are configured to avoid affinity issues and keep runs comparable.
# - ONNX model is resolved next to the weights by default (<weights>.onnx).
# - OpenVINO IR is resolved next to the weights by default (<weights>.xml). Prefer IR if present.
#
import os
import time
import json
import glob
import threading
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import onnxruntime
import psutil
import torch
from torch import nn

# Optional imports (loaded only when needed)
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

try:
    import openvino.runtime as ov  # type: ignore
except Exception:  # pragma: no cover
    ov = None

from mc_cnn.model.mc_cnn_fast import FastMcCnn

def get_memory_usage_bytes() -> int:
    """Get current process RSS in bytes."""
    return psutil.Process().memory_info().rss


def bytes_to_mb(b: int) -> float:
    return b / (1024.0 * 1024.0)


class MemorySampler:
    """
    Background sampler to capture true peak RSS during a stage.
    Sampling interval can be tuned with env MCCNN_MEM_SAMPLE_SEC (default 0.005s).
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
        self._peak = get_memory_usage_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mem_sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def peak_mb(self) -> float:
        return bytes_to_mb(self._peak)


def point_interval(left_features: torch.Tensor, right_features: torch.Tensor, disp: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute the horizontal intervals over which similarity is applied for a given disparity.
    left_features/right_features shape: (C=64, H, W)
    """
    _, _, nx_left = left_features.shape
    _, _, nx_right = right_features.shape

    # Range in the left image
    left = (max(0 - disp, 0), min(nx_left - disp, nx_left))
    # Range in the right image
    right = (max(0 + disp, 0), min(nx_right + disp, nx_right))

    return left, right


def _num_threads() -> int:
    """
    Unified threading knob.
    Default to 1 for reproducibility; override by setting env MCCNN_THREADS.
    """
    try:
        return max(1, int(os.getenv("MCCNN_THREADS", "1")))
    except Exception:
        return 1


def _resolve_onnx_path(model_path: str, img_shape: Optional[int] = None) -> str:
    """
    Locate the ONNX file.
    Priority:
      1) MCCNN_ONNX_PATH env var
      2) next to model_path with .onnx suffix
      3) model_path directory / mc_cnn_fast.onnx
      4) CWD / mc_cnn_fast.onnx
      5) this module folder / mc_cnn_fast.onnx
    """
    env_path = os.getenv("MCCNN_ONNX_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    mp = Path(model_path)
    if img_shape is None:
        candidates = [
            mp.with_name(mp.stem + f"_dynamo.onnx"),
            # mp.with_suffix(".onnx"),
            mp.parent / "mc_cnn_fast.onnx",
            Path.cwd() / "mc_cnn_fast.onnx",
            Path(__file__).resolve().parent / "mc_cnn_fast.onnx",
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
    else:
        candidates = glob.glob(os.path.join(mp.parent, f"*_{img_shape}.onnx"))
        return candidates[0]

    # Fallback (will raise later if missing)
    return "mc_cnn_fast.onnx"


def _resolve_openvino_path(model_path: str) -> str:
    """
    Locate the OpenVINO IR (.xml).
    Priority:
      1) MCCNN_OPENVINO_PATH env var
      2) next to model_path with .xml suffix
      3) model_path directory / mc_cnn_fast.xml
      4) CWD / mc_cnn_fast.xml
      5) this module folder / mc_cnn_fast.xml
    """
    env_path = os.getenv("MCCNN_OPENVINO_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    mp = Path(model_path)
    candidates = [
        mp.with_suffix(".xml"),
        mp.parent / "mc_cnn_fast.xml",
        Path.cwd() / "mc_cnn_fast.xml",
        Path(__file__).resolve().parent / "mc_cnn_fast.xml",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    # Fallback (will raise later if missing)
    return "mc_cnn_fast.xml"


def _ov_set_cpu_properties(core: "ov.Core", nt: int) -> None:
    """
    Set CPU plugin properties robustly across OpenVINO versions.
    Try progressively smaller property sets; ignore unsupported keys.
    """
    candidates = [
        # {"INFERENCE_NUM_THREADS": nt, "NUM_STREAMS": "1", "AFFINITY": "NONE", "INFERENCE_PRECISION_HINT": "f32"},
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
    # If all attempts fail, proceed with defaults


def _ov_compile_for_shape(core: "ov.Core", model_path: str, h: int, w: int) -> "ov.CompiledModel":
    """
    Read an OpenVINO model and compile it for a specific (H, W) by reshaping the input.
    Pick the reshape target from the model's input rank:
      - rank 2 -> [H, W]
      - rank 3 -> [1, H, W]
      - rank 4 -> [1, 1, H, W]
    """
    m = core.read_model(model_path)
    rank = m.inputs[0].get_partial_shape().rank
    # Default to 2D if unknown
    rank_len = rank.get_length() if rank.is_static else 2

    if rank_len == 2:
        target_shape = [h, w]
    elif rank_len == 3:
        target_shape = [1, h, w]
    elif rank_len == 4:
        target_shape = [1, 1, h, w]
    else:
        # Fallback: try 2D
        target_shape = [h, w]

    m.reshape({m.inputs[0]: target_shape})
    return core.compile_model(m, "CPU")


def _write_metrics_stages(framework: str, variant: str, model_path: str, data: dict) -> None:
    """
    Write per-stage metrics JSON to PANDORA_RUN_OUTPUT_DIR if set.
    """
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


def run_mc_cnn_fast(
    img_left: np.ndarray,
    img_right: np.ndarray,
    disp_min: int,
    disp_max: int,
    model_path: str,
    framework: str = "pytorch",
    variant: str = "baseline",
    provider: Optional[str] = "cpu_base",
) -> np.ndarray:
    """
    Compute the cost volume for a pair of images with MC-CNN fast (CPU-only).

    :param img_left: left image, shape (H, W), dtype float or uint8
    :param img_right: right image, shape (H, W)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)
    :param model_path: path to the trained network weights (.pth)
    :param framework: {"pytorch", "onnx", "openvino"}
    :param variant: {"baseline", "opt1"} selects the CV implementation
    :return: cost volume as numpy array of shape (H, W, D), float32
    """
    device = torch.device("cpu")  # Force CPU

    # We'll use input shape to pre-compile OV in model init
    H_in, W_in = int(img_left.shape[0]), int(img_left.shape[1])

    # ---------------- Stage: Model init ----------------
    ms = MemorySampler().start()
    start_init = time.perf_counter()

    if framework == "pytorch":
        # Cap PyTorch threads for reproducibility
        nt = _num_threads()
        try:
            torch.set_num_threads(nt)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        net = FastMcCnn()
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state["model"])
        net.to(device)
        net.eval()

        def inference_func(img_np: np.ndarray) -> torch.Tensor:
            # Expect img_np shape (H, W)
            x = torch.from_numpy(img_np.astype(np.float32, copy=False)).to(device=device)
            with torch.no_grad():
                feats = net(x, training=False)  # (64, H, W)
            return feats

    elif framework == "onnx":
        if ort is None:
            raise ImportError("onnxruntime is not installed but framework='onnx' was selected.")
        
        model_path = _resolve_onnx_path(model_path)

        nt = _num_threads()
        so = ort.SessionOptions()
        so.intra_op_num_threads = nt
        so.inter_op_num_threads = 1
        # Sequential mode to avoid extra thread pools
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if provider == "cpu_base":
            providers="CPUExecutionProvider"
            provider_options={}
        elif provider == "openvino":
            # so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            providers="OpenVINOExecutionProvider"
            provider_options={"device": "CPU_FP32"}
        else:
            warnings.warn(f"Provider {provider} is not implemented cpu_base selected then.")
            providers="CPUExecutionProvider"
            provider_options={}

        session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=[providers],
            provider_options=[provider_options]
        )

        def inference_func(img_np: np.ndarray) -> torch.Tensor:
            x = img_np.astype(np.float32, copy=False)
            outs = session.run(None, {"input": x})
            feats_np = outs[0]  # Expect (64, H, W)
            return torch.from_numpy(feats_np)

    elif framework == "openvino":
        if ov is None:
            raise ImportError("openvino is not installed but framework='openvino' was selected.")
        # Prefer IR (.xml) if present; fallback to ONNX
        xml_path = _resolve_openvino_path(model_path)
        onnx_path = _resolve_onnx_path(model_path)
        model_path = xml_path if Path(xml_path).exists() else onnx_path

        nt = _num_threads()
        core = ov.Core()
        _ov_set_cpu_properties(core, nt)

        # Compile once here for (H_in, W_in) so IA time is pure inference
        cm: "ov.CompiledModel" = _ov_compile_for_shape(core, model_path, H_in, W_in)
        input_port = cm.inputs[0]
        input_ps = cm.input(0).get_partial_shape()
        r = input_ps.rank
        rlen = r.get_length() if r.is_static else 2

        def inference_func(img_np: np.ndarray) -> torch.Tensor:
            x = img_np.astype(np.float32, copy=False)

            # Match input rank expected by compiled model
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
            feats_np = np.squeeze(feats_np)  # (64, H, W)
            return torch.from_numpy(feats_np)

    else:
        ms.stop()
        raise ValueError(f"Unsupported framework: {framework}")

    time_init = time.perf_counter() - start_init
    ms.stop()
    mem_init_peak = ms.peak_mb
    print(f"PROFILING_MODEL_INIT: time={time_init:.4f}s, mem_peak={mem_init_peak:.2f}MB, framework={framework}, variant={variant}")

    # ---------------- Stage: Feature extraction ----------------
    def normalize(img: np.ndarray) -> np.ndarray:
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
    left_features = inference_func(left)   # torch.Tensor on CPU, shape (64, H, W)
    right_features = inference_func(right) # torch.Tensor on CPU, shape (64, H, W)
    time_inf = time.perf_counter() - start_inf
    ms.stop()
    mem_inf_peak = ms.peak_mb
    print(f"PROFILING_IA_FEATURES: time={time_inf:.4f}s, mem_peak={mem_inf_peak:.2f}MB, framework={framework}, variant={variant}")

    # ---------------- Stage: Cost volume (non-IA loop) ----------------
    ms = MemorySampler().start()
    start_loop = time.perf_counter()
    if variant == "opt1":
        cv = computes_cost_volume_mc_cnn_fast_opt1(left_features, right_features, disp_min, disp_max)
    if variant == "opt2":
        cv = computes_cost_volume_mc_cnn_fast_opt2(left_features, right_features, disp_min, disp_max)
    else:
        cv = computes_cost_volume_mc_cnn_fast(left_features, right_features, disp_min, disp_max)
    time_loop = time.perf_counter() - start_loop
    ms.stop()
    mem_loop_peak = ms.peak_mb
    print(f"PROFILING_NON_IA_LOOP: time={time_loop:.4f}s, mem_peak={mem_loop_peak:.2f}MB")

    # ---------------- Write structured per-stage metrics ----------------
    _write_metrics_stages(
        framework=framework,
        variant=variant,
        model_path=model_path,
        data={
            "model_init_time": time_init,
            "model_init_mem": mem_init_peak,
            "ia_features_time": time_inf,
            "ia_features_mem": mem_inf_peak,
            "non_ia_loop_time": time_loop,
            "non_ia_loop_mem": mem_loop_peak,
        },
    )

    return cv


def computes_cost_volume_mc_cnn_fast(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Baseline cost volume: cosine similarity across channel dimension.
    Returns numpy array (H, W, D).
    """
    disparity_range = np.arange(disp_min, disp_max + 1, dtype=int)

    # Allocate cost volume as (D, W, H) for intermediate fill, initialized with NaN
    H = left_features.shape[1]
    W = left_features.shape[2]
    cv = np.empty((len(disparity_range), W, H), dtype=np.float32)
    cv.fill(np.nan)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)  # cosine over channel dimension C

    with torch.no_grad():
        for disp in disparity_range:
            left_int, right_int = point_interval(left_features, right_features, int(disp))
            ind_d = int(disp - disp_min)

            # Compute cosine similarity for the valid interval, then move to numpy
            sim = cos(
                left_features[:, :, left_int[0] : left_int[1]],
                right_features[:, :, right_int[0] : right_int[1]],
            )  # shape: (H, valid_W)

            # Place into cv (transpose to (valid_W, H))
            cv[ind_d, left_int[0] : left_int[1], :] = sim.cpu().numpy().T

    # Convert similarity to cost (negate), then return as (H, W, D)
    cv *= -1.0
    return np.swapaxes(cv, 0, 2)

def computes_cost_volume_mc_cnn_fast_opt1(left_features, right_features, disp_min, disp_max):
    """
    Optimized cost volume for MC-CNN fast, matching the original exactly:
    - Orientation: right-invalid for d > 0 (i.e., valid x ∈ [0, W-d))
    - Cost: -cosine_similarity between L2-normalized feature vectors
    - Implementation: channels-last for better memory access on CPU
    - Returns: np.ndarray (H, W, D) float32 with NaN where invalid
    """
    import torch as _torch
    with _torch.no_grad():
        lf = left_features.permute(1, 2, 0).contiguous()
        rf = right_features.permute(1, 2, 0).contiguous()
        H, W, C = lf.shape
        eps = 1e-6

        lf_norm = _torch.linalg.vector_norm(lf, dim=2, keepdim=True).clamp_min(eps)
        rf_norm = _torch.linalg.vector_norm(rf, dim=2, keepdim=True).clamp_min(eps)
        lf_n = lf / lf_norm
        rf_n = rf / rf_norm

        out = _torch.full((disp_max - disp_min + 1, H, W), float('nan'), dtype=lf.dtype, device=lf.device)

        for d in range(disp_min, disp_max + 1):
            di = d - disp_min
            if d >= 0:
                width = W - d
                if width <= 0:
                    continue
                sim = (lf_n[:, 0:width, :] * rf_n[:, d:W, :]).sum(dim=2).neg_()
                out[di, :, 0:width] = sim
            else:
                width = W + d
                if width <= 0:
                    continue
                sim = (lf_n[:, -d:W, :] * rf_n[:, 0:width, :]).sum(dim=2).neg_()
                out[di, :, -d:W] = sim

        cv_np = out.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return cv_np


def computes_cost_volume_mc_cnn_fast_opt2(
    left_features: torch.Tensor,   # (C, H, W), float32, CPU
    right_features: torch.Tensor,  # (C, H, W)
    disp_min: int,
    disp_max: int,
    assume_unit_norm: bool = True,  # True: trust model's F.normalize
) -> np.ndarray:
    """
    Faster MC-CNN CV:
    - Uses dot product across channels (features are L2-normalized by the model)
    - Channels-last for better CPU locality
    - Single numpy conversion at the end
    - Returns (H, W, D) float32 with NaN in invalid columns
    """
    with torch.no_grad():
        # (H, W, C) for contiguous channel access
        lf = left_features.permute(1, 2, 0).contiguous()
        rf = right_features.permute(1, 2, 0).contiguous()

        if not assume_unit_norm:
            # Only if features are not unit-norm (e.g., custom model)
            eps = 1e-6
            lf = lf / torch.clamp(torch.linalg.vector_norm(lf, dim=2, keepdim=True), min=eps)
            rf = rf / torch.clamp(torch.linalg.vector_norm(rf, dim=2, keepdim=True), min=eps)

        H, W, C = lf.shape
        D = disp_max - disp_min + 1

        out = torch.full((D, H, W), float("nan"), dtype=lf.dtype, device=lf.device)

        for d in range(disp_min, disp_max + 1):
            di = d - disp_min
            l0 = max(0, -d)
            r0 = max(0, d)
            width = W - abs(d)
            if width <= 0:
                continue
            # Dot product across channels; negate to convert similarity -> cost
            sim = (lf[:, l0:l0+width, :] * rf[:, r0:r0+width, :]).sum(dim=2).neg_()  # (H, width)
            out[di, :, l0:l0+width] = sim

        return out.permute(1, 2, 0).cpu().numpy().astype(np.float32)