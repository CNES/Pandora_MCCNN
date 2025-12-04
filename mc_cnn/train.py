#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
from tqdm import tqdm
import mlflow

from mc_cnn.model.mc_cnn_accurate import AccMcCnn
from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.model.mc_cnn_fast_dw import FastMcCnnDw
from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def load_dataset(cfg):
    """
    Load training and testing data.

    :param cfg: configuration
    :type cfg: dict
    """

    # Testing configuration : deactivate data augmentation
    test_cfg = copy.deepcopy(cfg)
    test_cfg["transformation"] = False

    if cfg["dataset"] == "middlebury":
        training_loader = MiddleburyGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = MiddleburyGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)
    elif cfg["dataset"] == "data_fusion_contest":
        training_loader = DataFusionContestGenerator(cfg["training_sample"], cfg["training_image"], cfg)
        testing_loader = DataFusionContestGenerator(cfg["testing_sample"], cfg["testing_image"], test_cfg)
    else:
        raise ValueError(
            f"dataset key {cfg['dataset']} does not correspond to one of the options in the list "
            "['middlebury', 'data_fusion_contest'] ."
        )

    return training_loader, testing_loader


def get_parameters_for_logs(cfg):
    """
    Get parameters for logs.

    :param cfg: configuration
    :type cfg: dict
    """
    params = {
        "network": cfg["network"],
        "conv": cfg["conv"],
        "dataset": cfg["dataset"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"],
        "training_sample": cfg["training_sample"],
        "training_image": cfg["training_image"],
        "testing_sample": cfg["testing_sample"],
        "testing_image": cfg["testing_image"],
        "dataset_neg_low": cfg["dataset_neg_low"],
        "dataset_neg_high": cfg["dataset_neg_high"],
        "dataset_pos": cfg["dataset_pos"],
        "data_augmentation": cfg["data_augmentation"],
    }
    if cfg["data_augmentation"]:
        for key, value in cfg["augmentation_param"].items():
            params[key] = value

    return params


def mcc_fast_training_epoch(net, training_generator, optimizer, criterion):
    """
    Run a mccnn fast training epoch.
    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Loss
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    train_epoch_loss = 0.0
    train_cur_size = 0

    net.train()

    train_progress_bar = tqdm(total=len(training_generator), desc="Training")
    for batch_idx, batch in enumerate(training_generator, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        left, pos, neg = net(batch.to(device=device, dtype=torch.float), training=True)
        # Cosine  similarity
        output_positive = cos(left, pos).squeeze()
        output_negative = cos(left, neg).squeeze()

        target = torch.ones(batch.size(0))
        loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item() * batch.size(0)
        train_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            train_progress_bar.set_postfix({"train_loss": f"{train_epoch_loss / train_cur_size :.4f}"}, refresh=False)
            train_progress_bar.update(1000)
            mlflow.log_metrics({"train_loss": train_epoch_loss / train_cur_size})

    return train_epoch_loss


def mcc_fast_testing_epoch(net, testing_generator, optimizer, criterion):
    """
    Run a mccnn fast testing epoch.

    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Loss
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    test_epoch_loss = 0.0
    test_cur_size = 0

    net.eval()

    test_progress_bar = tqdm(total=len(testing_generator), desc="Evaluation")
    for batch_idx, batch in enumerate(testing_generator, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        left, pos, neg = net(batch.to(device=device, dtype=torch.float), training=True)

        # Cosine  similarity
        output_positive = cos(left, pos).squeeze()
        output_negative = cos(left, neg).squeeze()

        target = torch.ones(batch.size(0))
        loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))

        test_epoch_loss += loss.item() * batch.size(0)
        test_cur_size += batch.size(0)

        if batch_idx % 100 == 0:
            test_progress_bar.set_postfix({"eval_loss": f"{test_epoch_loss / test_cur_size :.4f}"}, refresh=False)
            test_progress_bar.update(1000)

    return test_epoch_loss


def train_mc_cnn_fast(cfg, output_dir, dataloader_params):
    """
    Train the fast mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    :param dataloader_params: params for DataLoader
    :type dataloader_params: dict
    """
    mlflow.start_run()

    mlflow.log_params(get_parameters_for_logs(cfg))

    # Create the output directory
    mkdir_p(output_dir)
    save_cfg(output_dir, cfg)

    # Create the network
    if cfg["conv"] == "std":
        net = FastMcCnn()
    elif cfg["conv"] == "depthwise":
        net = FastMcCnnDw()
    else:
        raise ValueError(
            f"conv {cfg['network']} does not correspond to one of the options in the list " "['std', 'depthwise'] ."
        )
    net.to(device)

    # Optimizer
    if cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=cfg["learning_rate"], momentum=0.9)
    elif cfg["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    else:
        raise ValueError(
            f"optimizer {cfg['optimizer']} does not correspond to one of the options in the list " "['SGD', 'Adam'] ."
        )

    criterion = nn.MarginRankingLoss(margin=0.2, reduction="mean")

    # lr = 0.002 if epoch < 9
    # lr = 0.0002 if 9 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, gamma=0.1)

    # Load training and testing data
    training_loader, testing_loader = load_dataset(cfg)

    training_generator = data.DataLoader(training_loader, **dataloader_params)
    testing_generator = data.DataLoader(testing_loader, **dataloader_params)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print("-------- Fast epoch" + str(epoch) + " ------------")

        # Training
        train_epoch_loss = mcc_fast_training_epoch(net, training_generator, optimizer, criterion)
        training_loss.append(train_epoch_loss / len(training_loader))
        scheduler.step(epoch)

        # Evaluation
        test_epoch_loss = mcc_fast_testing_epoch(net, testing_generator, optimizer, criterion)
        testing_loss.append(test_epoch_loss / len(testing_loader))

        # Log metrics
        mlflow.log_metrics(
            {"train_loss": train_epoch_loss / len(training_loader), "eval_loss": test_epoch_loss / len(testing_loader)}
        )

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

        mlflow.pytorch.log_model(net, name=f"checkpoint_{epoch}")

    mlflow.end_run()


def mcc_acc_training_epoch(net, training_generator, optimizer, criterion):
    """
    Run a mccnn acc training epoch.
    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Loss
    """
    train_epoch_loss = 0.0
    train_cur_size = 0

    net.train()

    train_progress_bar = tqdm(total=len(training_generator))
    for batch_idx, batch in enumerate(training_generator, 0):
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
        train_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            train_progress_bar.set_postfix({"train_loss": f"{train_epoch_loss / train_cur_size :.4f}"}, refresh=False)
            train_progress_bar.update(1000)
            mlflow.log_metrics({"train_loss": train_epoch_loss / train_cur_size})

    return train_epoch_loss


def mcc_acc_testing_epoch(net, testing_generator, optimizer, criterion):
    """
    Run a mccnn acc testing epoch.
    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Loss
    """
    test_epoch_loss = 0.0
    test_cur_size = 0

    net.eval()

    test_progress_bar = tqdm(total=len(testing_generator), desc="Evaluation")
    for batch_idx, batch in enumerate(testing_generator, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        pos, neg = net(batch.to(device=device, dtype=torch.float))

        sample = torch.cat((pos, neg), dim=0)
        sample = torch.squeeze(sample)

        target = torch.cat((torch.ones(batch.size(0)), torch.zeros(batch.size(0))), dim=0)

        loss = criterion.forward(sample, target.to(device=device, dtype=torch.float))

        test_epoch_loss += loss.item() * batch.size(0)
        test_cur_size += batch.size(0)

        if batch_idx % 100 == 0:
            test_progress_bar.set_postfix({"eval_loss": f"{test_epoch_loss / test_cur_size :.4f}"}, refresh=False)
            test_progress_bar.update(1000)

    return test_epoch_loss


def train_mc_cnn_acc(cfg, output_dir, dataloader_params):
    """
    Train the accurate mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    :param dataloader_params: params for DataLoader
    :type dataloader_params: dict
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

    # Load training and testing data
    training_loader, testing_loader = load_dataset(cfg)

    training_generator = data.DataLoader(training_loader, **dataloader_params)
    testing_generator = data.DataLoader(testing_loader, **dataloader_params)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print("-------- Accurate epoch" + str(epoch) + " ------------")

        # Training
        train_epoch_loss = mcc_acc_training_epoch(net, training_generator, optimizer, criterion)
        training_loss.append(train_epoch_loss / len(training_loader))
        scheduler.step(epoch)

        # Evaluation
        test_epoch_loss = mcc_acc_testing_epoch(net, testing_generator, optimizer, criterion)
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
    with open(config_file, "r", encoding="utf-8") as file:
        user_configuration = json.load(file)
    return user_configuration


def save_cfg(output, configuration):
    """
    Save user configuration in the json file : config.json

    :param output: output directory
    :param configuration: user configuration
    """
    with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("injson", help="Input json file")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()

    user_cfg = read_config_file(args.injson)

    mlflow.set_experiment("rt_mccnn")

    # params for DataLoader
    batch_size = 128  # pylint: disable=C0103
    data_loader_params = {"batch_size": batch_size, "shuffle": True}

    if user_cfg["network"] == "fast":
        train_mc_cnn_fast(user_cfg, args.outdir, data_loader_params)
    elif user_cfg["network"] == "accurate":
        train_mc_cnn_acc(user_cfg, args.outdir, data_loader_params)
    else:
        raise ValueError(
            f"network {user_cfg['network']} does not correspond to one of the options in the list "
            "['accurate', 'acc'] ."
        )
