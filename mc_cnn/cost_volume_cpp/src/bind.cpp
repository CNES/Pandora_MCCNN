/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA-MCCNN
 *
 *     https://github.com/CNES/Pandora_MCCNN
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cv_pixelmajor_notorch_int32"


PYBIND11_MODULE(cv_pixelmajor_notorch_int32, m) {
    m.doc() = "MC-CNN CV pixel-major (NumPy I/O, CPU, expects HWC input, returns HWD)";
    m.def("cv_pixelmajor_int32", &cv_pixelmajor_int32,
          "Compute cost volume: inputs HWC float32, output HWD float32.",
          py::arg("left_features"),
          py::arg("right_features"),
          py::arg("disp_min"),
          py::arg("disp_max"),
          py::arg("write_invalid_nan") = true);
}