# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: Jan. 2020
"""

import argparse
import os
import numpy as np
import h5py
import tifffile
import glob


def save_dataset(img, sample, img_name, img_file, sample_file):
    """
    Save the sample in hdf5 files :
        - images are saved in the img_file file: creation of a dataset for each image pair
        - sample are saved in the sample_file file : creation of dataset containing valid pixels
    The dataset name is the ground truth file ( exemple : JAX_004_009_007_LEFT_DSP.tif )

    :param img: images
    :type img: np.array (2, 1024, 1024, 3) ( 2 = left image, right image)
    :param sample: samples of the image
    :type sample: np.array(number of valid pixels for all the images, 4).
    The last dimension is : number of the image, row, col, disparity for the pixel p(row, col)
    :param img_name: name of the current image pair ( name of the gt disparity )
    :type img_name: string
    :param img_file: image database file
    :type img_file: hdf5 file
    :param sample_file: training or testing database file
    :type sample_file: hdf5 file
    """
    sample_file.create_dataset(img_name, data=sample)
    img_file.create_dataset(img_name, data=img)

def fusion_contest(data_path, gt, output):
    """
    Preprocess and create data fusion contest hdf5 database

    :param data_path: path to the Train-Track2-RGB dataset
    :type data_path: string
    :param gt: path to the Train-Track2-Truth dataset
    :type gt: string
    :param output: output directory
    :type output: string
    """
    img_file = h5py.File(os.path.join(output, 'images_training_dataset_fusion_contest_0.hdf5'), 'w')
    training_file = h5py.File(os.path.join(output, 'training_dataset_fusion_contest_0.hdf5'), 'w')

    gt = glob.glob(gt + '/*_LEFT_DSP.tif')
    nb_img = len(gt)

    # Shuffle the file list
    indices = np.arange(nb_img)
    np.random.seed(0)
    np.random.shuffle(indices)
    gt = [gt[i] for i in indices]

    # 80 % Training, 20 % Testing
    end_training = int(nb_img * 0.8)

    num_file = 0
    num_img_per_file = 0
    for num_image in range(nb_img):
        # Get the three letter code for the current image
        file = os.path.basename(gt[num_image]).split('LEFT')[0]

        # Read images
        left = np.array(tifffile.imread(os.path.join(data_path, file + "LEFT_RGB.tif")))
        right = np.array(tifffile.imread(os.path.join(data_path, file + "RIGHT_RGB.tif")))
        dsp = np.array(tifffile.imread(gt[num_image]))
        nb_row, nb_col = dsp.shape

        # Normalization
        left = (left - left.mean()) / left.std()
        right = (right - right.mean()) / right.std()

        # Truncated the disparity map: remove 100 pixels on the top, bottom, left, right in order to have complete patch,
        # and remove 128 pixels on the left and right that correspond to the maximal and minimal disparity
        max_patch_size = 100 + 128
        dsp[0:max_patch_size, :] = -999
        dsp[nb_row-max_patch_size:, :] = -999
        dsp[:, 0:max_patch_size] = -999
        dsp[:, nb_col-max_patch_size:] = -999

        # Remove disparity that fall outside the secondary image
        mask = np.ones(dsp.shape) * np.arange(nb_col)
        # Data fusion contest disparity is : left(x,y) = right(x-d,y)
        mask -= dsp

        # dataset neg high can be 30
        dsp[np.where((mask > nb_col - 30) & (mask < 30))] = -999

        # Remove invalid disparity = -999
        y, x = np.where(dsp != -999)

        # data np.array of shape ( number of valid pixels the current image, 4 )
        # 4 = number of the image, row, col, disparity for the pixel p(row, col)
        valid_disp = np.column_stack((np.zeros_like(y) + num_image, y, x, dsp[y, x])).astype(np.float32)
        # img of shape (2, 1024, 1024, 3)
        img = np.stack((left, right), axis=0)

        if num_image == end_training:
            num_img_per_file = 0
            num_file = 0
            img_file = h5py.File(os.path.join(output, 'images_testing_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')
            testing_file = h5py.File(os.path.join(output, 'testing_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')

            save_dataset(img, valid_disp, file, img_file, testing_file)
            num_img_per_file += 1
        else:
            # Testing
            if num_image > end_training:
                # Max images per testing file
                if num_img_per_file == 200:
                    num_file += 1
                    img_file = h5py.File(os.path.join(output, 'images_testing_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')
                    testing_file = h5py.File(os.path.join(output, 'testing_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')
                    num_img_per_file = 0

                save_dataset(img, valid_disp, file, img_file, testing_file)
                num_img_per_file += 1
            # Training
            else:
                # Max images per training file
                if num_img_per_file == 200:
                    num_file += 1
                    img_file = h5py.File(os.path.join(output, 'images_training_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')
                    training_file = h5py.File(
                        os.path.join(output, 'training_dataset_fusion_contest_' + str(num_file) + '.hdf5'), 'w')
                    num_img_per_file = 0

                save_dataset(img, valid_disp, file, img_file, training_file)
                num_img_per_file += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for creating the data fusion contest database, creation files ( with a max of 200 images per files ): '
                                                 '- training_dataset_fusion_contest_.hdf5, testing_fusion_contest_.hdf5 which contains the '
                                                 'coordinates of the valid pixels and their disparity'
                                                 '- images_fusion_contest.hdf5 which contains the RGB normalized images ')
    parser.add_argument('input_data', help='Path to the input directory containing the data fusion contests Train-Track2-RGB')
    parser.add_argument('input_gt',
                        help='Path to the input directory containing the data fusion contests Train-Track2-Truth')
    parser.add_argument('output_dir', help='Path to the output directory ')
    args = parser.parse_args()

    fusion_contest(args.input_data, args.input_gt, args.output_dir)
