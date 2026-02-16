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
Dynamic MC-CNN module
"""

from torch import nn, squeeze, Tensor, no_grad


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

    def forward(self, sample: Tensor, training: bool):
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
            left = nn.functional.normalize(left, p=2, dim=1)

            pos = self.conv_blocks(sample[:, 1:2, :, :])
            pos = nn.functional.normalize(pos, p=2, dim=1)

            neg = self.conv_blocks(sample[:, 2:3, :, :])
            neg = nn.functional.normalize(neg, p=2, dim=1)

            return left, pos, neg
        else:
            with no_grad():
                feats = self.conv_blocks(sample.unsqueeze(0).unsqueeze(0))
                return squeeze(nn.functional.normalize(feats, p=2, dim=1))
