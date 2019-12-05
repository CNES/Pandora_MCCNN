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

from mc_cnn_fast import FastMcCnn
from dataset_generator import MiddleburyGenerator


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


def train(training, testing, image, output_dir):
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
    training_generator = data.DataLoader(MiddleburyGenerator(training, image), **params)
    testing_generator = data.DataLoader(MiddleburyGenerator(testing, image), **params)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    nb_epoch = 14
    training_loss = []
    testing_loss = []
    for epoch in range(nb_epoch):
        print('-------- Epoch' + str(epoch) + ' ------------')

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        net.train()
        for it, batch in enumerate(training_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            left, pos, neg = net(batch.to(device=device, dtype=torch.float))

            # Cosine  similarity
            output_positive = cos(left, pos)
            output_negative = cos(left, neg)

            target = torch.ones(batch_size)
            loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * batch.size(0)

        print('Training loss ', train_epoch_loss / len(training_generator))
        training_loss.append(train_epoch_loss / len(training_generator))

        # Save the network, optimizer, scheduler at each epoch
        torch.save(net.state_dict(), os.path.join(output_dir, 'mc_cnn_fast_network_epoch'+str(epoch)+'.pt'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'mc_cnn_fast_optimizer_epoch'+str(epoch)+'.pt'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'mc_cnn_fast_scheduler_epoch'+str(epoch)+'.pt'))

        scheduler.step(epoch)

        net.eval()
        for it, batch in enumerate(testing_generator, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            left, pos, neg = net(batch.to(device=device, dtype=torch.float))

            # Cosine  similarity
            output_positive = cos(left, pos)
            output_negative = cos(left, neg)

            target = torch.ones(batch_size) * -1
            loss = criterion.forward(output_positive, output_negative, target.to(device=device, dtype=torch.float))

            test_epoch_loss += loss.item() * batch.size(0)

        print('Testing loss ', test_epoch_loss / len(testing_generator))
        testing_loss.append(test_epoch_loss / len(testing_generator))

    # Save the loss in hdf5 file
    h5_file = h5py.File(os.path.join(output_dir, 'loss.hdf5'), 'w')
    h5_file.create_dataset("training_loss", (nb_epoch,), data=training_loss)
    h5_file.create_dataset("testing_loss", (nb_epoch,), data=testing_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', help='Path to a hdf5 file containing the training sample')
    parser.add_argument('testing', help='Path to a hdf5 file containing the testing sample')
    parser.add_argument('image', help='Path to a hdf5 file containing the image sample')
    parser.add_argument('output_dir', help='Output directory')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(args.training, args.testing, args.image, args.output_dir)
