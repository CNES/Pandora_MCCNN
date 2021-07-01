"""
This module contains the required libraries and softwares allowing to execute the software,
and setup elements to configure and identify the software.
"""

from codecs import open as copen
from setuptools import setup, find_packages


requirements = ['numpy',
                'numba',
                'rasterio',
                'torch',
                'torchvision',
                'h5py',
                'opencv-python',
                'scipy',
                'nose2']


def readme():
    with copen('README.md', 'r', 'utf-8') as file:
        return file.read()


setup(name='mc_cnn',
      version='x.y.z',
      description='MC-CNN is a neural network for learning a similarity measure on image patches',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/mc-cnn',
      author='CNES',
      author_email='myriam.cournet@cnes.fr',
      license='',
      install_requires=requirements,
      packages=find_packages())
