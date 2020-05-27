import os
import numpy as np
import h5py
from osgeo import gdal
import glob
from numba import njit
import argparse


@njit()
def compute_mask(disp, mask_ref, mask_sec, patch_size):
    """
    Masks invalid pixels

    :param disp: disparity map
    :type disp: 2D numpy array
    :param mask_ref: reference mask
    :type mask_ref: 2D numpy array
    :param mask_sec: secondary mask
    :type mask_sec: 2D numpy array
    :param patch_size: patch size
    :type patch_size: int
    :return: the disparity map with invalid pixels = -999
    :rtype: 2D numpy array
    """
    radius = int(patch_size / 2)
    row, col = disp.shape

    for r in range(row):
        for c in range(col):
            d = disp[r, c]
            patch_ref = mask_ref[(r - radius):(r + radius + 1), (c - radius):(c + radius + 1)]
            x1 = int(c + d)
            patch_sec = mask_sec[(r - radius):(r + radius + 1), (x1 - radius):(x1 + radius + 1)]

            # Patch outside epipolar image
            if np.sum(patch_ref != 0) != 0:
                disp[r, c] = -9999

            # Patch outside epipolar image
            if np.sum(patch_sec != 0) != 0:
                disp[r, c] = -9999

            # Dataset neg high can be -6
            x1 -= 6
            patch_sec = mask_sec[(r - radius):(r + radius + 1), (x1 - radius):(x1 + radius + 1)]

            # Patch outside epipolar image
            if np.sum(patch_sec != 0) != 0:
                disp[r, c] = -9999

            x1 = int(c + d)

            if disp != -999:
                # Negative patch outside image
                if ((r - radius) > 0) and ((r + radius + 1) < row) and ((x1 - radius) > 0) and (
                        (x1 + radius + 1) < col) and ((c - radius) > 0) and ((c + radius + 1) < col):
                    disp[r, c] = -9999

    return disp


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


def fusion_contest(input_dir, output):
    """
    Preprocess and create data fusion contest hdf5 database

    :param input_dir: path to the  input directory
    :type input_dir: string
    :param output: output directory
    :type output: string
    """
    img_file = h5py.File(os.path.join(output, 'images_training_dataset_fusion_contest_ours_.hdf5'), 'w')
    training_file = h5py.File(os.path.join(output, 'training_dataset_fusion_contest_ours_.hdf5'), 'w')
    img_testing_file = h5py.File(os.path.join(output, 'images_testing_dataset_fusion_contest_ours_.hdf5'), 'w')
    testing_file = h5py.File(os.path.join(output, 'testing_dataset_fusion_contest_ours_.hdf5'), 'w')

    gt = glob.glob(input_dir + '/*/left_epipolar_disp.tif')

    nb_img = len(gt)
    # Shuffle the file list
    indices = np.arange(nb_img)
    np.random.seed(0)
    np.random.shuffle(indices)
    gt = [gt[i] for i in indices]

    # 90 % Training, 10 % Testing
    end_training = int(nb_img * 0.9)

    for num_image in range(nb_img):
        name_image = gt[num_image].split(input_dir)[1].split('/')[1]
        path_image = gt[num_image].split('left_epipolar_disp.tif')[0]

        # Read images
        left = gdal.Open(os.path.join(path_image, 'left_epipolar_image.tif')).ReadAsArray()
        left_mask = gdal.Open(os.path.join(path_image,  'left_epipolar_mask.tif')).ReadAsArray()
        right = gdal.Open(os.path.join(path_image, 'right_epipolar_image.tif')).ReadAsArray()
        right_mask = gdal.Open(os.path.join(path_image, 'right_epipolar_mask.tif')).ReadAsArray()
        dsp = gdal.Open(gt[num_image]).ReadAsArray()
        mask_dsp = gdal.Open(os.path.join(path_image, 'left_epipolar_disp_mask.tif')).ReadAsArray()
        cross_checking = gdal.Open(os.path.join(path_image, 'valid_disp.tif')).ReadAsArray()

        # Mask disparities
        mask_disp = compute_mask(dsp, left_mask, right_mask, 11)
        # Remove invalid disp
        mask_disp[np.where(cross_checking == 255)] = -9999
        mask_disp[np.where(mask_dsp == 255)] = -9999

        # Change the disparity convention to ref(x,y) = sec(x-d,y)
        mask_disp *= -1
        # Remove invalid disparity
        y, x = np.where(mask_disp != 9999)

        # Red band selection
        left = np.squeeze(left[0, :, :])
        right = np.squeeze(right[0, :, :])

        # Normalization
        valid_left = np.where(left_mask == 0)
        valid_right = np.where(right_mask == 0)
        left[valid_left] = (left[valid_left] - left[valid_left].mean()) / left[valid_left].std()
        right[valid_right] = (right[valid_right] - right[valid_right].mean()) / right[valid_right].std()

        # data np.array of shape ( number of valid pixels the current image, 4 )
        # 4 = number of the image, row, col, disparity for the pixel p(row, col)
        valid_disp = np.column_stack((np.zeros_like(y) + num_image, y, x, mask_disp[y, x])).astype(np.float32)

        # img of shape (2, 2048, 2048, 3)
        img = np.stack((left, right), axis=0)
        if num_image > end_training:
            save_dataset(img, valid_disp, name_image, img_testing_file, testing_file)
        else:
            save_dataset(img, valid_disp, name_image, img_file, training_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for creating the data fusion contest database, creation files: '
                                                 '- training_dataset_fusion_contest_ours_.hdf5, testing_dataset_fusion_contest_ours_.hdf5 which contains the '
                                                 'coordinates of the valid pixels and their disparity'
                                                 '- images_training_dataset_fusion_contest_ours_.hdf5 and images_testing_dataset_fusion_contest_ours_ which contains the red band normalized images ')
    parser.add_argument('input_data', help='Path to the input directory containing the data')
    parser.add_argument('output_dir', help='Path to the output directory ')
    args = parser.parse_args()

    fusion_contest(args.input_data, args.output_dir)
