#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
from typing import Dict, Any, Tuple

from mc_cnn.model.mc_cnn_accurate import AccMcCnn
from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.model.mc_cnn_fast_dw import FastMcCnnDw
from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mkdir_p(path: str):
    """
    Create a directory without complaining if it already exists.

    :param path: path to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_dataset(
    cfg: Dict[str, Any]) -> Tuple[data.Dataset, data.Dataset]:
    """
    Load training and testing data.

    :param cfg: dict configuration

    :return: training and testing datasets.
    :rtype: Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
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


def get_parameters_for_mlflow_logs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get parameters for logs.

    :param cfg: dict configuration

    :return: parameters dict
    """
    params = {
        "network": cfg["network"],
        "conv": cfg["conv"],
        "num_conv_feature_maps": cfg.get("num_conv_feature_maps", 64),
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


def load_checkpoint(
    cfg: Dict[str, Any],
    net: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.LRScheduler
) -> Tuple[int, int]:
    """
    Run a mccnn fast testing epoch.

    :param cfg: configuration
    :type cfg: dict
    :param net: network
    :type net: torch.nn.Module
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param scheduler: scheduler
    :type scheduler: torch.optim.LRScheduler

    :return: start and end epoch; Tuple(int, int)
    """
    # Get run and params
    run = mlflow.active_run()
    run_data = run.to_dictionary()
    params = run_data["data"]["params"]

    # Compute start epoch
    start_epoch = int(params["epochs"])
    epoch_i = 0
    additional_epochs_key = f"additional_epochs{i}"
    while additional_epochs_key in params:
        start_epoch += int(params[additional_epochs_key])
        epoch_i += 1
        additional_epochs_key = f"additional_epochs{i}"

    # Get checkpoint path
    checkpoint_path = cfg["resume"].get("checkpoint", None)
    run_id = cfg["resume"]["run_id"]
    if not checkpoint_path:
        training_state_uri = f"runs:/{run_id}/checkpoints/mc_cnn_fast_epoch{start_epoch-1}.pt"
        checkpoint_path = mlflow.artifacts.download_artifacts(training_state_uri)

    # Load checkpoint
    print(f"Load checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    mlflow.log_params({additional_epochs_key: cfg["epochs"]})

    end_epoch = start_epoch + cfg["epochs"]

    return start_epoch, end_epoch


def mcc_fast_training_epoch(
    epoch: int,
    net: nn.Module,
    training_generator: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, int]:
    """
    Run a mccnn fast training epoch.
    :param epoch: Number of epoch
    :type epoch: int
    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Module

    :return: mean train loss per epoch and train number of accurate prediction per epoch
    :rtype: Tuple[float, int]
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    train_epoch_loss = 0.0
    train_num_correct = 0
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
        train_num_correct += (output_positive > output_negative).sum()
        train_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            train_loss = train_epoch_loss / train_cur_size
            train_acc = train_num_correct / train_cur_size
            train_progress_bar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}"}, refresh=False
            )
            train_progress_bar.update(1000)
            mlflow.log_metrics(
                {"batch_train_loss": train_loss, "batch_train_acc": train_acc},
                step=epoch * len(training_generator) + batch_idx,
            )

    return train_epoch_loss, train_num_correct


def mcc_fast_testing_epoch(
    net: nn.Module,
    testing_generator: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, int]:
    """
    Run a mccnn fast testing epoch.

    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Module

    :return: mean train loss per epoch and train number of accurate prediction per epoch
    :rtype: Tuple[float, int]
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    test_epoch_loss = 0.0
    test_num_correct = 0
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
        test_num_correct += (output_positive > output_negative).sum()
        test_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            test_loss = test_epoch_loss / test_cur_size
            test_accuracy = test_num_correct / test_cur_size
            test_progress_bar.set_postfix(
                {"test_loss": f"{test_loss:.4f}", "test_acc": f"{test_accuracy:.4f}"}, refresh=False
            )
            test_progress_bar.update(1000)

    return test_epoch_loss, test_num_correct


def train_mc_cnn_fast(
    cfg: Dict[str, Any],
    output_dir: str,
    dataloader_params: Dict[str, Any],
    experiment_id: str
):
    """
    Train the fast mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    :param dataloader_params: params for DataLoader
    :type dataloader_params: dict
    :param experiment_id: Mlflow experiment id
    :type experiment_id: string

    :raise ValueError: error is raised if
        - network option is not on the list ["std", "depthwise"]
        - optimizer is not in the list ['SGD', 'Adam']

    """
    # Create the output directory
    mkdir_p(output_dir)
    save_cfg(output_dir, cfg)

    # Create the network
    if cfg["conv"] == "std":
        net = FastMcCnn(num_conv_feature_maps=cfg.get("num_conv_feature_maps", 64))
    elif cfg["conv"] == "depthwise":
        net = FastMcCnnDw(num_conv_feature_maps=cfg.get("num_conv_feature_maps", 64))
    else:
        raise ValueError(
            f"conv {cfg['network']} does not correspond to one of the options in the list ['std', 'depthwise']."
        )
    net.to(device)

    # Optimizer
    if cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=cfg["learning_rate"], momentum=0.9)
    elif cfg["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    else:
        raise ValueError(
            f"optimizer {cfg['optimizer']} does not correspond to one of the options in the list ['SGD', 'Adam']."
        )

    criterion = nn.MarginRankingLoss(margin=0.2, reduction="mean")

    # lr = 0.002 if epoch < 9
    # lr = 0.0002 if 9 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, gamma=0.1)

    # Load training and testing data
    training_loader, testing_loader = load_dataset(cfg)

    training_generator = data.DataLoader(training_loader, **dataloader_params)
    testing_generator = data.DataLoader(testing_loader, **dataloader_params)

    # Start or resume mlflow run
    resume = cfg.get("resume", None)
    if resume:
        run_id = resume["run_id"]
        mlflow.start_run(experiment_id=experiment_id, run_id=run_id)
        start_epoch, end_epoch = load_checkpoint(cfg, net, optimizer, scheduler)
    else:
        mlflow.start_run(experiment_id=experiment_id)

        start_epoch = 0
        end_epoch = cfg["epochs"]

        mlflow.log_params(get_parameters_for_mlflow_logs(cfg))

    for epoch in range(start_epoch, end_epoch):
        print("-------- Fast epoch" + str(epoch) + " ------------")

        # Training
        train_epoch_loss, train_num_correct = mcc_fast_training_epoch(
            epoch, net, training_generator, optimizer, criterion
        )
        scheduler.step(epoch)

        # Evaluation
        test_epoch_loss, test_num_correct = mcc_fast_testing_epoch(net, testing_generator, optimizer, criterion)

        # Log metrics
        train_loss = train_epoch_loss / len(training_loader)
        test_loss = test_epoch_loss / len(testing_loader)
        train_acc = train_num_correct / len(training_loader)
        test_acc = test_num_correct / len(testing_loader)
        mlflow.log_metrics(
            {"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc}, step=epoch
        )

        checkpoint_filepath = os.path.join(output_dir, "mc_cnn_fast_epoch" + str(epoch) + ".pt")

        # Save the network, optimizer, scheduler at each epoch
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_epoch_loss": train_loss,
                "test_epoch_loss": test_loss,
                "train_epoch_acc": train_acc,
                "test_epoch_acc": test_acc,
            },
            checkpoint_filepath,
        )

        mlflow.log_artifact(checkpoint_filepath, artifact_path="checkpoints")

    mlflow.pytorch.log_model(net, name=f"model_epoch{epoch}")

    mlflow.end_run()


def mcc_acc_training_epoch(
    epoch: int,
    net: nn.Module,
    training_generator: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, int]:
    """
    Run a mccnn acc training epoch.
    
    :param epoch; number of epoch
    :type epoch: int
    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Module

    :return: mean train loss per epoch and train number of accurate prediction per epoch
    :rtype: Tuple[float, int]
    """
    train_epoch_loss = 0.0
    train_num_correct = 0
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
        train_num_correct += (pos.squeeze() > neg.squeeze()).sum()
        train_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            train_loss = train_epoch_loss / train_cur_size
            train_acc = train_num_correct / train_cur_size
            train_progress_bar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}"}, refresh=False
            )
            train_progress_bar.update(1000)
            mlflow.log_metrics(
                {"batch_train_loss": train_loss, "batch_train_acc": train_acc},
                step=epoch * len(training_generator) + batch_idx,
            )

    return train_epoch_loss, train_num_correct


def mcc_acc_testing_epoch(
    net: nn.Module,
    testing_generator: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, int]:
    """
    Run a mccnn acc testing epoch.

    :param net: network
    :type net: torch.nn.Module
    :param training_generator: training generator
    :type training_generator: torch.utils.data.Dataloader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: criterion
    :type criterion: torch.nn.Module

    :return: mean test loss per epoch and test number of accurate prediction per epoch
    :rtype: Tuple[float, int]
    """
    test_epoch_loss = 0.0
    test_num_correct = 0
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
        test_num_correct += (pos.squeeze() > neg.squeeze()).sum()
        test_cur_size += batch.size(0)

        if batch_idx % 1000 == 0:
            test_loss = test_epoch_loss / test_cur_size
            test_accuracy = test_num_correct / test_cur_size
            test_progress_bar.set_postfix(
                {"test_loss": f"{test_loss:.4f}", "test_accuracy": f"{test_accuracy:.4f}"}, refresh=False
            )
            test_progress_bar.update(1000)

    return test_epoch_loss, test_num_correct


def train_mc_cnn_acc(
    cfg: Dict[str, Any],
    output_dir: str,
    dataloader_params: Dict[str, Any],
    experiment_id: str):
    """
    Train the accurate mc_cnn network

    :param cfg: configuration
    :type cfg: dict
    :param output_dir: output directory
    :type output_dir: string
    :param dataloader_params: params for DataLoader
    :type dataloader_params: dict
    :param experiment_id: Mlflow experiment id
    :type experiment_id: string
    """
    # Create the output directory
    mkdir_p(output_dir)
    save_cfg(output_dir, cfg)

    # Create the network
    net = AccMcCnn()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    criterion = nn.BCELoss(reduction="mean")

    # lr = 0.003 if epoch < 10
    # lr = 0.0003 if 10 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    # Load training and testing data
    training_loader, testing_loader = load_dataset(cfg)

    training_generator = data.DataLoader(training_loader, **dataloader_params)
    testing_generator = data.DataLoader(testing_loader, **dataloader_params)

    # Start or resume mlflow run
    resume = cfg.get("resume", None)
    if resume:
        run_id = resume["run_id"]
        mlflow.start_run(experiment_id=experiment_id, run_id=run_id)
        start_epoch, end_epoch = load_checkpoint(cfg, net, optimizer, scheduler)
    else:
        mlflow.start_run(experiment_id=experiment_id)

        start_epoch = 0
        end_epoch = cfg["epochs"]

        mlflow.log_params(get_parameters_for_mlflow_logs(cfg))

    for epoch in range(start_epoch, end_epoch):
        print("-------- Accurate epoch" + str(epoch) + " ------------")

        # Training
        train_epoch_loss, train_num_correct = mcc_acc_training_epoch(
            epoch, net, training_generator, optimizer, criterion
        )
        scheduler.step(epoch)

        # Evaluation
        test_epoch_loss, test_num_correct = mcc_acc_testing_epoch(net, testing_generator, optimizer, criterion)

        train_loss = train_epoch_loss / len(training_loader)
        test_loss = test_epoch_loss / len(testing_loader)
        train_acc = train_num_correct / len(training_loader)
        test_acc = test_num_correct / len(testing_loader)
        # Log metrics
        mlflow.log_metrics(
            {"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc}, step=epoch
        )

        checkpoint_filepath = os.path.join(output_dir, "mc_cnn_acc_epoch" + str(epoch) + ".pt")

        # Save the network, optimizer, scheduler at each epoch
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_epoch_loss": train_loss,
                "test_epoch_loss": test_loss,
                "train_epoch_acc": train_acc,
                "test_epoch_acc": test_acc,
            },
            checkpoint_filepath,
        )

        mlflow.log_artifact(checkpoint_filepath, artifact_path="checkpoints")

    mlflow.pytorch.log_model(net, name=f"model_epoch{epoch}")

    mlflow.end_run()


def read_config_file(config_file: str) -> Dict[str, Any]:
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


def save_cfg(output: str, configuration: Dict[str, Any]):
    """
    Save user configuration in the json file : config.json

    :param output: output directory
    :type output: string
    :param configuration: user configuration
    :type configuration: dict
    """
    with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)


def setup_mlflow(cfg_mlflow: Dict[str, Any]) -> str:
    """
    Setup MLFlow

    :param cfg_mlflow: mlflow config
    :type cfg_mlflow: dict

    :return: experiment id
    :rtype: str
    """
    mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    try:
        print("Create new experiment ...")
        mlflow.create_experiment(cfg_mlflow["experiment"], artifact_location=cfg_mlflow.get("artifact_location", None))
    except mlflow.exceptions.MlflowException:
        print("Experiment already exist ...")

    experiment = mlflow.get_experiment_by_name(cfg_mlflow["experiment"])

    print(f"MLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
    print(f"MLFLOW_EXP: {experiment.name}")
    print(f"MLFLOW_ARTIFACT_LOCATION: {experiment.artifact_location}")

    return experiment.experiment_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("injson", help="Input json file")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()

    user_cfg = read_config_file(args.injson)

    exp_id = setup_mlflow(user_cfg["mlflow"])

    # params for DataLoader
    data_loader_params = {"batch_size": user_cfg["batch_size"], "shuffle": True}

    if user_cfg["network"] == "fast":
        train_mc_cnn_fast(user_cfg, args.outdir, data_loader_params, exp_id)
    elif user_cfg["network"] == "accurate":
        train_mc_cnn_acc(user_cfg, args.outdir, data_loader_params, exp_id)
    else:
        raise ValueError(
            f"network {user_cfg['network']} does not correspond to one of the options in the list "
            "['fast', 'accurate'] ."
        )
