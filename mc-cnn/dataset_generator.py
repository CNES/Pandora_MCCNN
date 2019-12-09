# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""

import random
from torch.utils import data
import h5py
import numpy as np


class MiddleburyGenerator(data.Dataset):
    """
    Generate middlebury dataset
    """
    def __init__(self, file, image, cfg):
        """
        Initialization

        :param file: training or testing hdf5 file
        :param image: image hdf5 file
        :param cfg: configuration
        :type cfg: dict( dataset_neg_low, dataset_neg_high, dataset_pos )
        """
        self.data = None
        self.h5_file_image = h5py.File(image, 'r')
        self.patch_size = 11
        self.image = []
        self.id_image = []

        self.neg_low = float(cfg['dataset_neg_low'])
        self.neg_high = float(cfg['dataset_neg_high'])
        self.pos = float(cfg['dataset_pos'])

        with h5py.File(file, 'r') as h5_file:

            for dst in h5_file.keys():
                if self.data is None:
                    self.data = h5_file[dst][:]
                else:
                    self.data = np.concatenate((self.data, h5_file[dst][:]), axis=0)
                self.id_image.append(int(h5_file[dst][0, 0]))

        with h5py.File(image, 'r') as h5_file:
            for grp in self.id_image:
                image_grp = []
                for dst in h5_file[str(int(grp))].keys():
                    image_grp.append(h5_file[str(int(grp))][dst][:])
                self.image.append(image_grp)

    def __getitem__(self, index):
        """
        Generates one sample : the left patch, the right positive patch, the right negative patch

        :return: left patch, right positive patch, right negative patch
        :rtype: np.array(3, patch_size, patch_size)
        """
        # Make patch
        id_image = int(self.data[index, 0])
        y = int(self.data[index, 1])
        x = int(self.data[index, 2])
        disp = int(self.data[index, 3])
        id_data = self.id_image.index(id_image)

        nb_illuminations = len(self.image[id_data])
        light_l = random.randint(0, nb_illuminations-1)

        # nb_exposures = self.h5_file_image[str(id_image)][str(light_l)].shape[0]
        nb_exposures = self.image[id_data][light_l].shape[0]
        exp_l = random.randint(0, nb_exposures-1)

        # Right illuminations and exposures
        light_r = light_l
        exp_r = exp_l

        # train 20 % of the time, on images where either the shutter exposure or the arrangements of lights are
        # different for the left and right image.
        if np.random.uniform() < 0.2:
            light_r = random.randint(0, nb_illuminations-1)

            if exp_r > self.image[id_data][light_r].shape[0] - 1:
                exp_r = random.randint(0, self.image[id_data][light_r].shape[0] - 1)

        if np.random.uniform() < 0.2:
            nb_exposures = self.image[id_data][light_r].shape[0]
            exp_r = random.randint(0, nb_exposures-1)

        # Make the left patch
        w = int(self.patch_size / 2)
        left = self.image[id_data][light_l][exp_l, 0, y - w: y + w + 1, x - w: x + w + 1]

        x_pos = -1
        width = self.image[id_data][light_r].shape[3] - 11

        while x_pos < 0 or x_pos >= width:
            x_pos = int((x - disp) + np.random.uniform(-self.pos, self.pos))

        x_neg = -1
        while x_neg < 0 or x_neg >= width:
            x_neg = int((x - disp) + np.random.uniform(self.neg_low, self.neg_high))

        # Make the right positive patch
        right_pos = self.image[id_data][light_r][exp_r, 1, y - w: y + w + 1, x_pos - w: w + x_pos + 1]

        # Make the right negative patch
        right_neg = self.image[id_data][light_r][exp_r, 1, y - w: y + w + 1, x_neg - w: w + x_neg + 1]

        return np.stack((left, right_pos, right_neg), axis=0)

    def __len__(self):
        """
        Return the total number of samples

        """
        return self.data.shape[0]
