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
This module contains all functions to train mc-cnn fast and accurate networks
"""

import argparse
import os
import errno
import json
import copy
import torch
from torch import nn, optim
from torch.utils import data


from mc_cnn.model.mc_cnn_accurate import AccMcCnn
from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train_mc_cnn_fast(cfg, output_dir):
    """
    Train the fast mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    """
    # Create the output directory
    mkdir_p(output_dir)
    save_cfg(output_dir, cfg)

    # Create the network
    net = FastMcCnn()
    net.to(device)

    criterion = nn.MarginRankingLoss(margin=0.2, reduction="mean")

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    # lr = 0.002 if epoch < 9
    # lr = 0.0002 if 9 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, gamma=0.1)

    batch_size = 128
    params = {"batch_size": batch_size, "shuffle": True}

    # Testing configuration : deactivate data augmentation
    test_cfg = copy.deepcopy(cfg)
    test_cfg["transformation"] = False

    if cfg["dataset"] == "middlebury":
        training_loader = MiddleburyGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = MiddleburyGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)

    if cfg["dataset"] == "data_fusion_contest":
        training_loader = DataFusionContestGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = DataFusionContestGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)

    training_generator = data.DataLoader(training_loader, **params)
    testing_generator = data.DataLoader(testing_loader, **params)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print("-------- Fast epoch" + str(epoch) + " ------------")

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        net.train()

        for _, batch in enumerate(training_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            left, pos, neg = net(batch.to(device=device, dtype=torch.float), training=True)

            # Cosine  similarity
            output_positive = cos(left, pos)
            output_negative = cos(left, neg)

            target = torch.ones(batch.size(0))
            loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * batch.size(0)

        training_loss.append(train_epoch_loss / len(training_loader))
        scheduler.step(epoch)

        net.eval()
        for _, batch in enumerate(testing_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            left, pos, neg = net(batch.to(device=device, dtype=torch.float), training=True)

            # Cosine  similarity
            output_positive = cos(left, pos)
            output_negative = cos(left, neg)

            target = torch.ones(batch.size(0))
            loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))

            test_epoch_loss += loss.item() * batch.size(0)

        testing_loss.append(test_epoch_loss / len(testing_loader))

        # Save the network, optimizer, scheduler at each epoch
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_epoch_loss": train_epoch_loss / len(training_loader),
                "test_epoch_loss": test_epoch_loss / len(testing_loader),
            },
            os.path.join(output_dir, "mc_cnn_fast_epoch" + str(epoch) + ".pt"),
        )


def train_mc_cnn_acc(cfg, output_dir):
    """
    Train the accurate mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    """
    # Create the output directory
    mkdir_p(output_dir)
    save_cfg(output_dir, cfg)

    # Create the network
    net = AccMcCnn()
    net.to(device)

    criterion = nn.BCELoss(reduction="mean")

    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # lr = 0.003 if epoch < 10
    # lr = 0.0003 if 10 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    batch_size = 128
    params = {"batch_size": batch_size, "shuffle": True}

    # Testing configuration : deactivate data augmentation
    test_cfg = copy.deepcopy(cfg)
    test_cfg["transformation"] = False

    if cfg["dataset"] == "middlebury":
        training_loader = MiddleburyGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = MiddleburyGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)

    if cfg["dataset"] == "data_fusion_contest":
        training_loader = DataFusionContestGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = DataFusionContestGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)

    training_generator = data.DataLoader(training_loader, **params)
    testing_generator = data.DataLoader(testing_loader, **params)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print("-------- Accurate epoch" + str(epoch) + " ------------")

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        net.train()

        for _, batch in enumerate(training_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            pos, neg = net(batch.to(device=device, dtype=torch.float))

            sample = torch.cat((pos, neg), dim=0)
            sample = torch.squeeze(sample)

            target = torch.cat((torch.ones(batch.size(0)), torch.zeros(batch.size(0))), dim=0)

            loss = criterion.forward(sample, target.to(device=device, dtype=torch.float))
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * batch.size(0)

        training_loss.append(train_epoch_loss / len(training_loader))
        scheduler.step(epoch)

        net.eval()
        for _, batch in enumerate(testing_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            pos, neg = net(batch.to(device=device, dtype=torch.float))

            sample = torch.cat((pos, neg), dim=0)
            sample = torch.squeeze(sample)

            target = torch.cat((torch.ones(batch.size(0)), torch.zeros(batch.size(0))), dim=0)

            loss = criterion.forward(sample, target.to(device=device, dtype=torch.float))

            test_epoch_loss += loss.item() * batch.size(0)

        testing_loss.append(test_epoch_loss / len(testing_loader))

        # Save the network, optimizer, scheduler at each epoch
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_epoch_loss": train_epoch_loss / len(training_loader),
                "test_epoch_loss": test_epoch_loss / len(testing_loader),
            },
            os.path.join(output_dir, "mc_cnn_acc_epoch" + str(epoch) + ".pt"),
        )


def read_config_file(config_file):
    """
    Read a json configuration file

    :param config_file: path to a json file containing the algorithm parameters
    :type config_file: string
    :return: the configuration
    :rtype: dict
    """
    with open(config_file, "r") as file:
        user_configuration = json.load(file)
    return user_configuration


def save_cfg(output, configuration):
    """
    Save user configuration in the json file : config.json

    :param output: output directory
    :param configuration: user configuration
    """
    with open(os.path.join(output, "config.json"), "w") as file:
        json.dump(configuration, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("injson", help="Input json file")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()

    user_cfg = read_config_file(args.injson)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if user_cfg["network"] == "fast":
        train_mc_cnn_fast(user_cfg, args.outdir)
    else:
        train_mc_cnn_acc(user_cfg, args.outdir)
