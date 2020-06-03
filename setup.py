from setuptools import setup, find_packages
from codecs import open
import subprocess

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
    with open('README.md', "r", "utf-8") as f:
        return f.read()


setup(name='mc_cnn',
      version_format='{sha}',
      setup_requires=['very-good-setuptools-git-version'],
      description='MC-CNN is a neural network for learning a similarity measure on image patches',
      long_description=readme(),
      url='https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/mc-cnn',
      author='CNES',
      author_email='myriam.cournet@cnes.fr',
      license='',
      install_requires=requirements,
      packages=find_packages())
