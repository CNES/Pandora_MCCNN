<h1 align="center"> MCCNN </h1>

<h4 align="center">MCCNN neural network for stereo matching cost.</h4>

<p align="center">
  <a href="https://github.com/CNES/Pandora_MCCNN/actions"> <img src="https://github.com/CNES/Pandora_MCCNN/actions/workflows/mccnn_ci.yml/badge.svg?branch=master"></a>
  <a href="https://opensource.org/licenses/Apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#usage">Pretrained Weights for MCCNN networks</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>

## Overview

Pytorch implementation of [[MCCNN]](#1.) neural network which computes a similarity measure on pair of small image patches.

## Install from Pypi

**MCCNN** is available on Pypi and can be installed by:

```bash
pip install MCCNN
```

## Developer install

After cloning source code from repository, do a local pip install in a virtualenv through MCCNN Makefile:

```bash
make install
```

## Usage

Documentation explains how to train and use MCCNN convolutional neural network.
To generate it, please execute the following commands:

```bash
make docs
```

Let's see [pandora_plugin_mccnn](https://github.com/CNES/Pandora_plugin_mccnn) for real life example.

## Pretrained Weights for MCCNN networks

### Download weights files

Pretrained weights for mc-cnn fast and mc-cnn accurate neural networks are available in the weights directory :
-  mc_cnn_fast_mb_weights.pt and mc_cnn_accurate_mb_weights.pt are the weights of the pretrained networks on the Middlebury dataset [[Middlebury]](#Middlebury)
-  mc_cnn_fast_data_fusion_contest.pt and mc_cnn_accurate_data_fusion_contest.pt are the weights of the pretrained networks on the Data Fusion Contest dataset [[DFC]](#DFC)

To download the pretrained weights:

```bash
wget https://raw.githubusercontent.com/CNES/Pandora_MCCNN/master/mc_cnn/weights/mc_cnn_fast_mb_weights.pt
wget https://raw.githubusercontent.com/CNES/Pandora_MCCNN/master/mc_cnn/weights/mc_cnn_fast_data_fusion_contest.pt
wget https://raw.githubusercontent.com/CNES/Pandora_MCCNN/master/mc_cnn/weights/mc_cnn_accurate_mb_weights.pt
wget https://raw.githubusercontent.com/CNES/Pandora_MCCNN/master/mc_cnn/weights/mc_cnn_accurate_data_fusion_contest.pt
```

### Access weights from pip package

Pretrained weights are stored into the pip package and downloaded for any installation of mc_cnn pip package.
To access it, use the `weights` submodule :

```python
from mc_cnn.weights import get_weights
mc_cnn_fast_mb_weights_path = get_weights(arch="fast", training_dataset="middlebury")
mc_cnn_fast_data_fusion_contest_path = get_weights(arch="fast", training_dataset="dfc")
mc_cnn_accurate_mb_weights_path = get_weights(arch="accurate", training_dataset="middlebury")
mc_cnn_accurate_data_fusion_contest = get_weights(arch="accurate", training_dataset="dfc")
```

## References

Please cite the following paper when using MCCNN:

*Defonte, V., Dumas, L., Cournet, M., & Sarrazin, E. (2021, July). Evaluation of MC-CNN Based Stereo Matching Pipeline for the CO3D Earth Observation Program. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS (pp. 7670-7673). IEEE.*
   
*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*

<a id="1.">[MCCNN]</a> 
*Zbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res., 17(1), 2287-2318.*

<a id="Middlebury">[Middlebury]</a> 
*Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014, September). High-resolution stereo datasets with subpixel-accurate ground truth. In German conference on pattern recognition (pp. 31-42). Springer, Cham.*

<a id="DFC">[DFC]</a> 
*Bosch, M., Foster, K., Christie, G., Wang, S., Hager, G. D., & Brown, M. (2019, January). Semantic stereo for incidental satellite images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1524-1532). IEEE.*
