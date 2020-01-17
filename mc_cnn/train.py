# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""

import torch
from torch import nn, optim
from torch.utils import data
import argparse
import os
import errno
import h5py
import numpy as np

from mc_cnn_fast import FastMcCnn
from mc_cnn_accurate import AccMcCnn
from dataset_generator import MiddleburyGenerator

# Global variable that control the random seed
SEED = 0

def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:   # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _init_fn(worker_id):
    """
    Set the init function in order to have multiple worker in the dataloader
    """
    np.random.seed(SEED + worker_id)


def set_seed():
    """
    Set up the random seed

    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    np.random.seed(SEED)  # Numpy module.
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_mc_cnn_fast(training, testing, image, output_dir, dataset_cfg):
    """
    Train the fast mc_cnn network

    :param training: path to the hdf5 training dataset
    :type training: string
    :param testing: path to the hdf5 testing dataset
    :type testing: string
    :param image: path to the hdf5 image dataset
    :type image: string
    :param output_dir: output directory
    :type output_dir: string
    :param dataset_cfg: data augmentation
    :type dataset_cfg: dict
    """
    # Create the output directory
    mkdir_p(output_dir)

    # Create the network
    net = FastMcCnn()
    net.to(device)

    criterion = nn.MarginRankingLoss(margin=0.2, reduction='mean')

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    # lr = 0.002 if epoch < 9
    # lr = 0.0002 if 9 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, gamma=0.1)

    batch_size = 128
    params = {'batch_size': batch_size, 'shuffle': True}

    training_loader = MiddleburyGenerator(training, image, dataset_cfg)
    training_generator = data.DataLoader(training_loader, **params)
    testing_loader = MiddleburyGenerator(testing, image, dataset_cfg)
    testing_generator = data.DataLoader(testing_loader, **params)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print('-------- Fast epoch' + str(epoch) + ' ------------')

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        net.train()
        # Change the seed at each epoch
        SEED = epoch
        for it, batch in enumerate(training_generator, 0):
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
        for it, batch in enumerate(testing_generator, 0):
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
        torch.save({'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_epoch_loss': train_epoch_loss / len(training_loader),
                    'test_epoch_loss': test_epoch_loss / len(testing_loader)},
                   os.path.join(output_dir, 'mc_cnn_fast_epoch' + str(epoch) + '.pt'))

    # Save the loss in hdf5 file
    h5_file = h5py.File(os.path.join(output_dir, 'loss.hdf5'), 'w')
    h5_file.create_dataset("training_loss", (nb_epoch,), data=training_loss)
    h5_file.create_dataset("testing_loss", (nb_epoch,), data=testing_loss)


def train_mc_cnn_acc(training, testing, image, output_dir, dataset_cfg):
    """
    Train the accurate mc_cnn network

    :param training: path to the hdf5 training dataset
    :type training: string
    :param testing: path to the hdf5 testing dataset
    :type testing: string
    :param image: path to the hdf5 image dataset
    :type image: string
    :param output_dir: output directory
    :type output_dir: string
    :param dataset_cfg: data augmentation
    :type dataset_cfg: dict
    """
    # Create the output directory
    mkdir_p(output_dir)

    # Create the network
    net = AccMcCnn()
    net.to(device)

    criterion = nn.BCELoss(reduction='mean')

    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # lr = 0.003 if epoch < 10
    # lr = 0.0003 if 10 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    batch_size = 128
    params = {'batch_size': batch_size, 'shuffle': True}

    training_loader = MiddleburyGenerator(training, image, dataset_cfg)
    training_generator = data.DataLoader(training_loader, **params)
    testing_loader = MiddleburyGenerator(testing, image, dataset_cfg)
    testing_generator = data.DataLoader(testing_loader, **params)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print('-------- Accurate epoch' + str(epoch) + ' ------------')

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        net.train()
        # Change the seed at each epoch
        SEED = epoch
        for it, batch in enumerate(training_generator, 0):
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
        for it, batch in enumerate(testing_generator, 0):
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
        torch.save({'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_epoch_loss': train_epoch_loss / len(training_loader),
                    'test_epoch_loss': test_epoch_loss / len(testing_loader)},
                   os.path.join(output_dir, 'mc_cnn_acc_epoch' + str(epoch) + '.pt'))

    # Save the loss in hdf5 file
    h5_file = h5py.File(os.path.join(output_dir, 'loss.hdf5'), 'w')
    h5_file.create_dataset("training_loss", (nb_epoch,), data=training_loss)
    h5_file.create_dataset("testing_loss", (nb_epoch,), data=testing_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', help="Type of the network : accurate or fast ", choices=['accurate', 'fast'])
    parser.add_argument('training', help='Path to a hdf5 file containing the training sample')
    parser.add_argument('testing', help='Path to a hdf5 file containing the testing sample')
    parser.add_argument('image', help='Path to a hdf5 file containing the image sample')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('-data_augmentation', help='Apply data augmentation ?', choices=['True', 'False'], default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fast_cfg = {'dataset': 'middlebury', 'transformation': args.data_augmentation, 'dataset_neg_low': 1.5,
                'dataset_neg_high': 6, 'dataset_pos': 0.5, 'scale': 0.8, 'hscale': 0.8, 'hshear': 0.1, 'trans': 0,
                'rotate': 28, 'brightness': 1.3, 'contrast': 1.1, 'd_hscale': 0.9, 'd_hshear': 0.3, 'd_vtrans': 1,
                'd_rotate': 3, 'd_brightness': 0.7, 'd_contrast': 1.1}

    acc_cfg = {'dataset': 'middlebury', 'transformation': args.data_augmentation, 'dataset_neg_low': 1.5,
               'dataset_neg_high': 18, 'dataset_pos': 0.5, 'scale': 0.8, 'hscale': 0.8, 'hshear': 0.1,
               'trans': 0, 'rotate': 28, 'brightness': 1.3, 'contrast': 1.1, 'd_hscale': 0.9, 'd_hshear': 0.3,
               'd_vtrans': 1, 'd_rotate': 3, 'd_brightness': 0.7, 'd_contrast': 1.1}

    if args.net == 'fast':
        train_mc_cnn_fast(args.training, args.testing, args.image, args.output_dir, fast_cfg)
    else:
        train_mc_cnn_acc(args.training, args.testing, args.image, args.output_dir, acc_cfg)
