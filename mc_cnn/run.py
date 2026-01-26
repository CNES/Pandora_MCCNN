# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA_MCCNN
#
#     https://github.com/CNES/Pandora_MCCNN
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
CPU-only execution for MC-CNN fast with frameworks and variants.
Notes:
- All paths are CPU-only regardless of hardware availability.
- Single-thread by default for stability (override with env MCCNN_THREADS).
- ONNX and OpenVINO sessions are configured to avoid affinity issues and keep runs comparable.
- ONNX model is resolved next to the weights by default (<weights>.onnx).
- OpenVINO IR is resolved next to the weights by default (<weights>.xml). Prefer IR if present.
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np


def import_libraries(framework: str, variant: str):
    modules = {}
    if (variant in ["baseline", "opt1", "opt2", "cpp", "cpp2"]) or (framework == "pytorch"):
        import torch

        modules["torch"] = torch

        if variant == "baseline":
            import torch.nn as nn

            modules["nn"] = nn

    if framework == "onnx":
        import onnxruntime as ort

        modules["ort"] = ort
    elif framework == "openvino":
        import openvino as ov

        modules["ov"] = ov

    return modules


def _resolve_onnx_path(model_path: str, model_name: Optional[str] = None) -> str:
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

    # Fallback (will raise later if missing)
    return "mc_cnn_fast.onnx"


def _resolve_openvino_path(model_path: str, model_name: Optional[str] = None) -> str:
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

    # Fallback (will raise later if missing)
    return "mc_cnn_fast.xml"


def _ov_set_cpu_properties(core: "ov.Core", nt: int) -> None:
    """
    Set CPU plugin properties robustly across OpenVINO versions.
    Try progressively smaller property sets; ignore unsupported keys.
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


def _num_threads() -> int:
    """
    Unified threading knob.
    Default to 1 for reproducibility; override by setting env MCCNN_THREADS.
    """
    try:
        return max(1, int(os.getenv("MCCNN_THREADS", "1")))
    except Exception:
        return 1


def run_mc_cnn_fast(
    img_left: np.ndarray,
    img_right: np.ndarray,
    disp_min: int,
    disp_max: int,
    model_path: str,
    framework: str = "pytorch",
    variant: str = "baseline",
    provider: Optional[str] = "cpu_base",
    model_name: Optional[str] = None,
    window_size: Optional[int] = None,  # configured window size (Pandora); used by PyTorch
) -> np.ndarray:
    """
    Compute the cost volume for a pair of images with MC-CNN fast (CPU-only).

    :param img_left: left image, shape (H, W), dtype float or uint8
    :param img_right: right image, shape (H, W)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)
    :param model_path: path to the trained network weights (.pt/.onnx/.xml)
    :param framework: {"pytorch", "onnx", "openvino"}
    :param variant: {"baseline", "opt1", "opt2", "cpp", "cpp2", ...} for CV loop
    :param provider: ONNX/OpenVINO provider selection
    :param model_name: optional override filename for ONNX/OV next to model_path
    :param window_size: odd patch size 7/11/13/15; required for PyTorch to build exact-depth net
    :return: cost volume as numpy array of shape (H, W, D), float32
    """
    # We'll use input shape to pre-compile OV in model init
    H_in, W_in = int(img_left.shape[0]), int(img_left.shape[1])

    # ---------------- Stage: Import library ----------------
    modules = import_libraries(framework, variant)

    # ---------------- Stage: Model init ----------------
    if framework == "pytorch":
        if window_size is None:
            # Pandora must pass window_size; choose safe default but will likely mismatch
            window_size = 11
        L = max(1, (int(window_size) - 1) // 2)  # number of 3x3 valid conv layers

        # Cap PyTorch threads for reproducibility
        nt = _num_threads()

        torch = modules["torch"]
        import torch.nn as nn

        try:
            torch.set_num_threads(nt)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        device = torch.device("cpu")  # Force CPU

        # Dynamic MC-CNN fast with N conv layers (3x3, valid), ReLU after each conv except the last
        class FastMcCnnDyn(nn.Module):
            def __init__(self, num_layers: int):
                super().__init__()
                layers = []
                in_ch = 1
                out_ch = 64
                for i in range(num_layers):
                    layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3))
                    if i < num_layers - 1:
                        layers.append(nn.ReLU())
                    in_ch = out_ch
                self.conv_blocks = nn.Sequential(*layers)

            def forward(self, sample, training):
                if training:
                    left = self.conv_blocks(sample[:, 0:1, :, :])
                    left = torch.nn.functional.normalize(left, p=2, dim=1)

                    pos = self.conv_blocks(sample[:, 1:2, :, :])
                    pos = torch.nn.functional.normalize(pos, p=2, dim=1)

                    neg = self.conv_blocks(sample[:, 2:3, :, :])
                    neg = torch.nn.functional.normalize(neg, p=2, dim=1)

                    return left, pos, neg
                else:
                    with torch.no_grad():
                        feats = self.conv_blocks(sample.unsqueeze(0).unsqueeze(0))
                        return torch.squeeze(torch.nn.functional.normalize(feats, p=2, dim=1))

        # Build, then load weights strictly
        net = FastMcCnnDyn(num_layers=L)
        state = torch.load(model_path, map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        # strip DataParallel 'module.' if present
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd)  # strict=True by default
        net.to(device)
        net.eval()

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            # Expect img_np shape (H, W)
            x = torch.from_numpy(img_np.astype(np.float32, copy=False)).to(device=device)
            with torch.no_grad():
                feats = net(x, training=False)  # (64, H', W')
            return feats.numpy()

    elif framework == "onnx":
        ort = modules["ort"]
        if ort is None:
            raise ImportError("onnxruntime is not installed but framework='onnx' was selected.")

        model_path = _resolve_onnx_path(model_path, model_name)

        nt = _num_threads()
        so = ort.SessionOptions()
        so.intra_op_num_threads = nt
        so.inter_op_num_threads = 1
        # Sequential mode to avoid extra thread pools
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if provider == "cpu_base":
            providers = "CPUExecutionProvider"
            provider_options = {}
        elif provider == "openvino":
            providers = "OpenVINOExecutionProvider"
            provider_options = {
                "num_of_threads": str(nt),
            }
        else:
            warnings.warn(f"Provider {provider} is not implemented; falling back to CPUExecutionProvider.")
            providers = "CPUExecutionProvider"
            provider_options = {}

        session = ort.InferenceSession(
            model_path, sess_options=so, providers=[providers], provider_options=[provider_options]
        )

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            x = img_np.astype(np.float32, copy=False)
            outs = session.run(None, {"input": x})
            feats_np = outs[0]  # Expect (64, H, W)
            return feats_np

    elif framework == "openvino":
        ov = modules["ov"]
        if ov is None:
            raise ImportError("openvino is not installed but framework='openvino' was selected.")

        # Prefer IR (.xml) if present; fallback to ONNX
        xml_path = _resolve_openvino_path(model_path, model_name)
        onnx_path = _resolve_onnx_path(model_path, model_name)
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

        def inference_func(img_np: np.ndarray) -> np.ndarray:
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
            return feats_np

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # ---------------- Stage: Feature extraction ----------------
    def normalize(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        mean = float(img.mean())
        std = float(img.std())
        if std == 0.0:
            std = 1.0
        return (img - mean) / std

    left = normalize(img_left)
    right = normalize(img_right)
    left_features = inference_func(left)  # (64, H', W') depending on model depth
    right_features = inference_func(right)  # (64, H', W')

    # ---------------- Stage: Cost volume (non-IA loop) ----------------
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
    elif variant == "cpp2_notorch_int32":
        cv = computes_cost_volume_mc_cnn_fast_cpp2_notorch_int32(left_features, right_features, disp_min, disp_max)
    else:
        cv = computes_cost_volume_mc_cnn_fast(modules, left_features, right_features, disp_min, disp_max)

    return cv


def computes_cost_volume_mc_cnn_fast(
    modules: Dict[str, Any],
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    Baseline cost volume: cosine similarity across channel dimension.
    Returns numpy array (H, W, D).
    """
    torch = modules["torch"]
    nn = modules["nn"]
    # Construct the cost volume
    disparity_range = np.arange(disp_min, disp_max + 1).astype(np.int32)

    # Allocate cost volume as (D, W, H) for intermediate fill, initialized with NaN
    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)
    H = lf.shape[1]
    W = lf.shape[2]
    cv = np.empty((len(disparity_range), W, H), dtype=np.float32)
    cv.fill(np.nan)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)  # cosine over channel dimension C

    def point_interval(
        left_features: "torch.Tensor", right_features: "torch.Tensor", disp: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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

    with torch.no_grad():
        for disp in disparity_range:
            left_int, right_int = point_interval(lf, rf, int(disp))
            ind_d = int(disp - disp_min)

            # Compute cosine similarity for the valid interval, then move to numpy
            sim = cos(
                lf[:, :, left_int[0] : left_int[1]],
                rf[:, :, right_int[0] : right_int[1]],
            )  # shape: (H, valid_W)

            # Place into cv (transpose to (valid_W, H))
            cv[ind_d, left_int[0] : left_int[1], :] = sim.cpu().numpy().T

    # Convert similarity to cost (negate), then return as (H, W, D)
    cv *= -1.0
    return np.swapaxes(cv, 0, 2)


def computes_cost_volume_mc_cnn_fast_opt1(
    modules: Dict[str, Any],
    left_features: np.ndarray,  # (C, H, W), float32, CPU
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    MC-CNN fast cost volume (cosine): normalize features once, then dot product.
    - Orientation: right-invalid for d > 0 (valid x ∈ [0, W-d))
    - Cost = -cosine_similarity (i.e., -dot of L2-normalized features)
    - Returns: np.ndarray (H, W, D) float32 with NaN in invalid columns
    """
    torch = modules["torch"]

    with torch.no_grad():
        # Channels-last for better memory locality
        lf = torch.from_numpy(left_features).permute(1, 2, 0).contiguous()  # (H, W, C)
        rf = torch.from_numpy(right_features).permute(1, 2, 0).contiguous()  # (H, W, C)

        # L2-normalize across channel dim
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
            # Dot across channels; negate to convert similarity -> cost
            sim = (lf[:, l0 : l0 + width, :] * rf[:, r0 : r0 + width, :]).sum(dim=2).neg_()  # (H, width)
            out[di, :, l0 : l0 + width] = sim

    return out.permute(1, 2, 0).contiguous().numpy()


def computes_cost_volume_mc_cnn_fast_opt1_notorch(
    left_features: np.ndarray,  # (C, H, W), float32, CPU
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    MC-CNN fast cost volume (cosine): normalize features once, then dot product.
    - Orientation: right-invalid for d > 0 (valid x ∈ [0, W-d))
    - Cost = -cosine_similarity (i.e., -dot of L2-normalized features)
    - Returns: np.ndarray (H, W, D) float32 with NaN in invalid columns
    """
    # Channels-last for better memory locality
    lf = np.ascontiguousarray(np.transpose(left_features, axes=(1, 2, 0)))  # (H, W, C)
    rf = np.ascontiguousarray(np.transpose(right_features, axes=(1, 2, 0)))  # (H, W, C)

    # L2-normalize across channel dim
    eps = 1e-6
    lf = lf / np.clip(np.linalg.vector_norm(lf, axis=2, keepdims=True), a_min=eps, a_max=None)
    rf = rf / np.clip(np.linalg.vector_norm(rf, axis=2, keepdims=True), a_min=eps, a_max=None)

    H, W, C = lf.shape
    D = disp_max - disp_min + 1
    out = np.full((D, H, W), np.nan, dtype=lf.dtype)

    for d in range(disp_min, disp_max + 1):
        di = d - disp_min
        l0 = max(0, -d)
        r0 = max(0, d)
        width = W - abs(d)
        if width <= 0:
            continue
        # Dot across channels; negate to convert similarity -> cost
        sim = np.negative((lf[:, l0 : l0 + width, :] * rf[:, r0 : r0 + width, :]).sum(axis=2))  # (H, width)
        out[di, :, l0 : l0 + width] = sim

    return np.ascontiguousarray(np.transpose(out, axes=(1, 2, 0)))


def computes_cost_volume_mc_cnn_fast_opt2(
    modules: Dict[str, Any],
    left_features: np.ndarray,  # (C, H, W), float32, CPU
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    MC-CNN fast cost volume (dot): assume features are L2 unit-norm (opt2 legal).
    - Orientation: right-invalid for d > 0 (valid x ∈ [0, W-d))
    - Cost = -dot(left, right) across channels (equivalent to -cosine if unit-norm)
    - Returns: np.ndarray (H, W, D) float32 with NaN in invalid columns
    """
    torch = modules["torch"]

    with torch.no_grad():
        lf = torch.from_numpy(left_features).permute(1, 2, 0).contiguous()  # (H, W, C)
        rf = torch.from_numpy(right_features).permute(1, 2, 0).contiguous()  # (H, W, C)

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
            # Dot across channels; negate to convert similarity -> cost
            sim = (lf[:, l0 : l0 + width, :] * rf[:, r0 : r0 + width, :]).sum(dim=2).neg_()  # (H, width)
            out[di, :, l0 : l0 + width] = sim

    return out.permute(1, 2, 0).contiguous().numpy()


def computes_cost_volume_mc_cnn_fast_opt2_notorch(
    left_features: np.ndarray,  # (C, H, W), float32, CPU
    right_features: np.ndarray,  # (C, H, W)
    disp_min: int,
    disp_max: int,
) -> np.ndarray:
    """
    MC-CNN fast cost volume (dot): assume features are L2 unit-norm (opt2 legal).
    - Orientation: right-invalid for d > 0 (valid x ∈ [0, W-d))
    - Cost = -dot(left, right) across channels (equivalent to -cosine if unit-norm)
    - Returns: np.ndarray (H, W, D) float32 with NaN in invalid columns
    """
    # Channels-last for better memory locality
    lf = np.ascontiguousarray(np.transpose(left_features, axes=(1, 2, 0)))  # (H, W, C)
    rf = np.ascontiguousarray(np.transpose(right_features, axes=(1, 2, 0)))  # (H, W, C)

    H, W, C = lf.shape
    D = disp_max - disp_min + 1
    out = np.full((D, H, W), np.nan, dtype=lf.dtype)

    for d in range(disp_min, disp_max + 1):
        di = d - disp_min
        l0 = max(0, -d)
        r0 = max(0, d)
        width = W - abs(d)
        if width <= 0:
            continue
        # Dot across channels; negate to convert similarity -> cost
        sim = np.negative(np.sum(lf[:, l0 : l0 + width, :] * rf[:, r0 : r0 + width, :], axis=2))  # (H, width)
        out[di, :, l0 : l0 + width] = sim

    return np.ascontiguousarray(np.transpose(out, axes=(1, 2, 0)))


def computes_cost_volume_mc_cnn_fast_cpp(
    modules: Dict[str, Any], left_features: np.ndarray, right_features: np.ndarray, disp_min: int, disp_max: int
):
    torch = modules["torch"]
    from .cv_opt2_single_loader import computes_cost_volume_mc_cnn_fast_opt2_single_cpp

    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)

    with torch.no_grad():
        out = computes_cost_volume_mc_cnn_fast_opt2_single_cpp(lf, rf, disp_min, disp_max)  # (D,H,W)
        return out.permute(1, 2, 0).contiguous().numpy()  # (H,W,D), zero-copy


def computes_cost_volume_mc_cnn_fast_cpp2(
    modules: Dict[str, Any], left_features: np.ndarray, right_features: np.ndarray, disp_min: int, disp_max: int
):

    torch = modules["torch"]
    from .cv_opt2_pixelmajor_loader import computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp

    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)

    with torch.no_grad():
        out = computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp(lf, rf, disp_min, disp_max)  # (H,W,D)
    return out.numpy()


def computes_cost_volume_mc_cnn_fast_cpp_notorch(left_features, right_features, disp_min, disp_max):
    """
    Calls native single kernel (returns D,H,W) and transposes to (H,W,D).
    Accepts torch.Tensor or np.ndarray as inputs; converts to NumPy (C,H,W),
    then performs CHW -> HWC in Python and calls the native HWC kernel.
    """
    from .cv_opt2_single_loader_notorch import (
        computes_cost_volume_mc_cnn_fast_opt2_single_cpp as computes_cost_volume_mc_cnn_fast_opt2_single_cpp_notorch,
    )

    # Validate CHW
    if left_features.ndim != 3 or right_features.ndim != 3:
        raise ValueError("left/right features must be 3D (C,H,W)")
    if left_features.shape != right_features.shape:
        raise ValueError("left/right feature shapes must match")
    if not left_features.flags.c_contiguous:
        left_features = np.ascontiguousarray(left_features)
    if not right_features.flags.c_contiguous:
        right_features = np.ascontiguousarray(right_features)

    # CHW -> HWC (fast NumPy path) with C-order copy
    lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
    rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

    # Native notorch kernel (expects HWC, returns DHW)
    out_dhw = computes_cost_volume_mc_cnn_fast_opt2_single_cpp_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return np.transpose(out_dhw, (1, 2, 0))  # (H,W,D)


def computes_cost_volume_mc_cnn_fast_cpp2_notorch(left_features, right_features, disp_min, disp_max):
    """
    Calls native pixel-major kernel (returns H,W,D) and returns as-is.
    Accepts torch.Tensor or np.ndarray as inputs; converts to NumPy (C,H,W),
    then performs CHW -> HWC in Python and calls the native HWC kernel.
    """
    from .cv_opt2_pixelmajor_loader_notorch import (
        computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp as computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp_notorch,
    )

    # Validate CHW
    if left_features.ndim != 3 or right_features.ndim != 3:
        raise ValueError("left/right features must be 3D (C,H,W)")
    if left_features.shape != right_features.shape:
        raise ValueError("left/right feature shapes must match")
    if not left_features.flags.c_contiguous:
        left_features = np.ascontiguousarray(left_features)
    if not right_features.flags.c_contiguous:
        right_features = np.ascontiguousarray(right_features)

    # CHW -> HWC (fast NumPy path) with C-order copy for downstream speed
    lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
    rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

    # Native notorch kernel (expects HWC, returns HWD)
    out_hwd = computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return out_hwd


def computes_cost_volume_mc_cnn_fast_cpp2_notorch_int32(left_features, right_features, disp_min, disp_max):
    """
    Calls native pixel-major kernel (returns H,W,D) and returns as-is.
    Accepts torch.Tensor or np.ndarray as inputs; converts to NumPy (C,H,W),
    then performs CHW -> HWC in Python and calls the native HWC kernel.
    """
    from .cv_opt2_pixelmajor_loader_notorch_int32 import (
        computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp_int32 as computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp_notorch,
    )

    # Validate CHW
    if left_features.ndim != 3 or right_features.ndim != 3:
        raise ValueError("left/right features must be 3D (C,H,W)")
    if left_features.shape != right_features.shape:
        raise ValueError("left/right feature shapes must match")
    if not left_features.flags.c_contiguous:
        left_features = np.ascontiguousarray(left_features)
    if not right_features.flags.c_contiguous:
        right_features = np.ascontiguousarray(right_features)

    # CHW -> HWC (fast NumPy path) with C-order copy for downstream speed
    lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
    rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

    # Native notorch kernel (expects HWC, returns HWD)
    out_hwd = computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return out_hwd
