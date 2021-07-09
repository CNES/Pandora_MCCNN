<h1 align="center"> MCCNN </h1>

<h4 align="center">MCCNN neural network for stereo matching cost.</h4>

<p align="center">
  <a><img src="https://github.com/CNES/Pandora_MCCNN/actions/workflows/mccnn_ci.yml/badge.svg?branch=master"></a>
  <a href="https://opensource.org/licenses/Apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
    <a href="#usage">Usage</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>

## Overview

Pytorch implementation of [[MCCNN]](#1.) neural network which computes a similarity measure on pair of small image patches.

[Pandora](https://github.com/CNES/Pandora) stereo matching framework is designed to provide some state of the art stereo algorithms and to add others one as plugins.  
This [Pandora plugin](https://pandora.readthedocs.io/userguide/plugin.html) aims to compute the cost volume using the similarity measure produced by MC-CNN neural network [[MCCNN]](#1.), with the [MCCNN](https://github.com/CNES/Pandora_MCCNN) library.

## Install

**MCCNN** is available on Pypi and can be installed by:

```bash
pip install MCCNN
```

## Usage

Documentation explains how to train and use MCCNN convolutional neural network.
To generate it, please execute the following commands:

```bash
pip install MCCNN[doc]
python setup.py build_sphinx
```

Let's see [pandora_plugin_mccnn](https://github.com/CNES/Pandora_plugin_mccnn) for real life example.

## Related

[Pandora](https://github.com/CNES/Pandora) - A stereo matching framework  
[Plugin_mccnn](https://github.com/CNES/Pandora_plugin_mccnn) - Stereo Matching Algorithm plugin for Pandora  


## References

Please cite the following paper when using MCCNN:
   
*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*

<a id="1.">[MCCNN]</a> 
*Zbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res., 17(1), 2287-2318.*

