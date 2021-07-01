#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
This module contains the required libraries and softwares allowing to execute the software,
and setup elements to configure and identify the software.
"""

from codecs import open as copen
from setuptools import setup, find_packages


REQUIREMENTS = ['numpy',
                'numba',
                'rasterio',
                'torch',
                'torchvision',
                'h5py',
                'opencv-python',
                'scipy',]

SETUP_REQUIREMENTS = ["setuptools-scm"]

REQUIREMENTS_DEV = {
    "dev": [
        "pytest",
        "pytest-cov",
        "pylint",
        "pre-commit",
        "black",
    ],
}


def readme():
    with copen("README.md", "r", "utf-8") as file:
        return file.read()


setup(name='mc_cnn',
      use_scm_version=True,
      description='MC-CNN is a neural network for learning a similarity measure on image patches',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/mc-cnn',
      author='CNES',
      author_email='myriam.cournet@cnes.fr',
      license="Apache License 2.0",
      install_requires=REQUIREMENTS,
      setup_requires=SETUP_REQUIREMENTS,
      extras_require=REQUIREMENTS_DEV,
      packages=find_packages())
