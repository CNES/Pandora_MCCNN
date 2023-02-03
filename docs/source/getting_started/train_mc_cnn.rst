Train MCCNN neural network
==========================

Dataset preparation
###################

The first step is to prepare the database.
`mc_cnn/preprocessing` folder contains files to preprocess and create hdf5 datasets for [Middlebury]_ and [DFC]_.

During the preprocessing cross-checking is computed to locate occluded pixels, and image are normalized.
At the end of the script 3 files are created:

- training_dataset.hdf5, testing_dataset.hdf5 : contains the training and testing data.
                          There is one dataset per disparity map ground truth.
                          Each dataset contains a numpy.array of size (number of valid pixels in the disparity map, 4).
                          Last dimension contains the dataset key, the row position of the valid pixel, the col position of the valid pixel and
                          ground truth disparity of the pixel row,col.

- images.hdf5 : contains testing and training normalized images. There is one group per image pair.
                Each group contains datasets that are a numpy.arrays of size (2 (left image, right image), row, col).


Example

>>> # 7 ground truth disparity maps available
>>> training_dataset.keys()
<KeysViewHDF5 ['0', '1', '2', '3', '4', '5', '6']>
>>> # The first disparity map contains 753877 valid pixels
>>> training_dataset['1']
<HDF5 dataset "1": shape (753877, 4), type "<f4">
>>> # Coordinates and disparity of the first valid pixel in the ground truth : pixel(row=11, col=72) = 59.35666 (disparity)
>>> training_dataset['1'][0]
array([ 1.     , 11.     , 72.     , 59.35666], dtype=float32)

>>> # 7 images available
>>> images.keys()
<KeysViewHDF5 ['0', '1', '2', '3', '4', '5', '6']>
>>> images['1']
<HDF5 group "/1" (7 members)>
>>> images['1'].keys()
<KeysViewHDF5 ['0', '1', '2', '3', '4', '5', '6']>
>>> images['1']['5']
<HDF5 dataset "0": shape (1, 2, 994, 1474), type "<f8">


Training
########

Once the hdf5 databases are created, the training can be launched:


.. code:: bash

    python mc_cnn/train.py -h
    usage: train.py [-h] injson outdir

    positional arguments:
      injson      Input json file
      outdir      Output directory

    optional arguments:
      -h, --help  show this help message and exit



The json file contains the training parameters, and data augmentation parameters (more information about data augmentation parameters can be found here [MCCNN]_).
Configuration examples can be found in the `mc_cnn/training_config` folder.

.. [Middlebury] Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014, September). High-resolution stereo datasets with subpixel-accurate ground truth. In German conference on pattern recognition (pp. 31-42). Springer, Cham.
.. [DFC] Bosch, M., Foster, K., Christie, G., Wang, S., Hager, G. D., & Brown, M. (2019, January). Semantic stereo for incidental satellite images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1524-1532). IEEE.
.. [MCCNN] Zbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res., 17(1), 2287-2318.