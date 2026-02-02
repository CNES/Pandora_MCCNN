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
- 2 frameworks are available: PyTorch (.pt) and onnx (.onnx)
"""

import os
from typing import Tuple, Optional, Dict, Any

import numpy as np


def import_libraries(framework: str, variant: str)-> Dict[str, Any]:
    """
    Import the required libraries based on the variant and framework.

    :param framework: name of the framework
    :param variant: name of the variant

    :return: dict of imported libraries
    """
    modules = {}
    if (variant == "baseline") or (framework == "pytorch"):
        import torch
        import torch.nn
        modules["torch"] = torch
        modules["nn"] = torch.nn

    if framework == "onnx":
        import onnxruntime as ort
        modules["ort"] = ort

    return modules


def _num_threads() -> int:
    """
    Unified threading knob.
    Default to 1 for reproducibility; override by setting env MCCNN_THREADS.

    :return: int = 1
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
    window_size: Optional[int] = None,  # configured window size (Pandora); used by PyTorch
) -> np.ndarray:
    """
    Compute the cost volume for a pair of images with MC-CNN fast (CPU-only).
    Notes:
    - 2 frameworks are available for the AI part: pytorch (nominal method) and onnx (optimized method)
    - 2 variant for the cost volume loop computation: baseline (nominal method) and cpp (optimized method)

    :param img_left: left image, shape (H, W), dtype float or uint8
    :param img_right: right image, shape (H, W)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)
    :param model_path: path to the trained network weights (.pt/.onnx)
    :param framework: {"pytorch", "onnx"}
    :param variant: {"baseline", "cpp"} for CV loop
    :param window_size: odd patch size 7/11/13/15, parameter only for PyTorch, for ONNX point to the right onnx file.

    :return: cost volume as numpy array of shape (H, W, D), float32
    """
    # ---------------- Stage: Import library ----------------
    modules = import_libraries(framework, variant)
    nt = _num_threads()

    # ---------------- Stage: Model init ----------------
    if framework == "pytorch":
        if window_size is None:
            # Pandora must pass window_size; choose safe default but will likely mismatch
            window_size = 11
        layer_nb = max(1, (int(window_size) - 1) // 2)  # number of 3x3 valid conv layers

        # Cap PyTorch threads for reproducibility
        torch = modules["torch"]
        nn = modules["nn"]

        try:
            torch.set_num_threads(nt)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        device = torch.device("cpu")  # Force CPU

        class FastMcCnnDyn(nn.Module):
            """
            Dynamic MC-CNN fast with N conv layers (3x3, valid), ReLU after each conv except the last
            
            :param num_layers: number of convolutional layers, depends on the window size

                              W - 1
                num_layers = ------- or 1 num_layers < 1
                                2
            """
            def __init__(self, num_layers: int):
                super().__init__()
                layers = []
                in_ch = 1
                out_ch = 64
                for layer_i in range(num_layers):
                    layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3))
                    if layer_i < num_layers - 1:
                        layers.append(nn.ReLU())
                    in_ch = out_ch
                self.conv_blocks = nn.Sequential(*layers)

            def forward(self, sample: torch.Tensor, training: bool):
                """
                Forward function

                :param sample:
                    - if training mode :
                        - normalized patch : torch (batch_size, 3, 11, 11) with: 3 is the left patch, right positive patch,
                                             right negative patch, 11 the patch
                    - else :
                        - normalized image torch(batch_size, row, col)
                :param training: training mode, true for train false else, bool 

                :return:

                    - if training mode : left, right positive and right negative features, 
                                         (torch(batch_size, 64, 1, 1), torch(batch_size, 64, 1, 1), torch(batch_size, 64, 1, 1))
                    - else : extracted features, torch(64, row, col)
                """
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
        net = FastMcCnnDyn(num_layers=layer_nb)
        state = torch.load(model_path, map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        # strip DataParallel 'module.' if present
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd)  # strict=True by default
        net.to(device)
        net.eval()

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            """
            Inference function with PyTorch

            :param: image to infer (H, W). 
        
            :return: image features (C=64, H, W), float32
            """
            # Expect img_np shape (H, W)
            img = torch.from_numpy(img_np.astype(np.float32, copy=False)).to(device=device)
            with torch.no_grad():
                feats = net(img, training=False)  # (64, H', W')
            return feats.numpy()

    elif framework == "onnx":
        ort = modules["ort"]
        if ort is None:
            raise ImportError("onnxruntime is not installed but framework='onnx' was selected.")

        so = ort.SessionOptions()
        so.intra_op_num_threads = nt
        so.inter_op_num_threads = 1
        # Sequential mode to avoid extra thread pools
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = "CPUExecutionProvider"
        provider_options = {}

        session = ort.InferenceSession(
            model_path, sess_options=so, providers=[providers], provider_options=[provider_options]
        )

        def inference_func(img_np: np.ndarray) -> np.ndarray:
            """
            Inference function with ONNX runtime

            :param: image to infer (H, W). 
        
            :return: image features (C=64, H, W), float32
            """
            img = img_np.astype(np.float32, copy=False)
            outs = session.run(None, {"input": img})
            feats_np = outs[0]  # Expect (64, H, W)
            return feats_np

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # ---------------- Stage: Feature extraction ----------------
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Image normalization

                    img - mean
        img_norm  = ----------
                       std
    
        :param img: image to normalized (H, W)

        :return: normalized image (H, W), float32
        """
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
    if variant == "cpp":
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

    :param modules: dict with the libraries to import
    :param left_features: features from the left images encoded by convolutional network part (64, H, W)
    :param right_features: features from the right images encoded by convolutional network part (64, H, W)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)

    :return: cost volume as numpy array of shape (H, W, D), float32
    """
    torch = modules["torch"]
    nn = modules["nn"]
    # Construct the cost volume
    disparity_range = np.arange(disp_min, disp_max + 1).astype(np.int32)

    # Allocate cost volume as (D, W, H) for intermediate fill, initialized with NaN
    lf = torch.from_numpy(left_features)
    rf = torch.from_numpy(right_features)
    height = lf.shape[1]
    width = lf.shape[2]
    cv = np.empty((len(disparity_range), width, height), dtype=np.float32)
    cv.fill(np.nan)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)  # cosine over channel dimension C

    def point_interval(
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        disp: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Compute the horizontal intervals over which similarity is applied for a given disparity.
        left_features/right_features shape: (C=64, H, W)

        :param left_features: features from the left images encoded by convolutional network part (64, H, W)
        :param right_features: features from the right images encoded by convolutional network part (64, H, W)
        :param disp: disparity integer value.

        :return: the pixel range for the left and right image. (min_left, max_left), (min_right, max_right)
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


def computes_cost_volume_mc_cnn_fast_cpp2_notorch_int32(
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int
) -> np.ndarray:
    """
    Calls native pixel-major kernel (returns H, W, D) and returns as-is.
    Accepts torch.Tensor or np.ndarray as inputs; converts to NumPy (C, H, W),
    then performs CHW -> HWC in Python and calls the native HWC kernel.

    :param modules: dict with the libraries to import
    :param left_features: features from the left images encoded by convolutional network part (64, H, W)
    :param right_features: features from the right images encoded by convolutional network part (64, H, W)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)

    :return: cost volume as numpy array of shape (H, W, D), float32
    """
    from .cv_pixelmajor_loader_notorch_int32 import (
        computes_cost_volume_mc_cnn_fast_pixelmajor_cpp_int32 as computes_cost_volume_mc_cnn_fast_pixelmajor_cpp_notorch,
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
    out_hwd = computes_cost_volume_mc_cnn_fast_pixelmajor_cpp_notorch(lf_hwc, rf_hwc, disp_min, disp_max)
    return out_hwd
