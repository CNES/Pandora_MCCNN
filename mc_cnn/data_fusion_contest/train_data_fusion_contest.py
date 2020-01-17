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
import numpy as np
import glob

from mc_cnn.data_fusion_contest.mc_cnn_accurate import AccMcCnnDataFusion
from mc_cnn.data_fusion_contest.dataset_generator_fusion_contest import DataFusionContestGenerator


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


def train_mc_cnn_acc(input_dir, output_dir, dataset_cfg):
    """
    Train the accurate mc_cnn network

    :param input_dir: path to the hdf5 input dataset
    :type input_dir: string
    :param output_dir: output directory
    :type output_dir: string
        :param dataset_cfg: data augmentation
    :type dataset_cfg: dict
    """
    # Create the output directory
    mkdir_p(output_dir)

    # Create the network
    net = AccMcCnnDataFusion()
    net.to(device)

    criterion = nn.BCELoss(reduction='mean')

    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # lr = 0.003 if epoch < 10
    # lr = 0.0003 if 10 <= epoch < 18 ...
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    # Set up the seed
    set_seed()
    batch_size = 128
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 10, 'worker_init_fn':_init_fn}

    nb_epoch = 5
    training_loss = []
    testing_loss = []
    for epoch in range(0, nb_epoch):
        print('-------- Accurate epoch' + str(epoch) + ' ------------')

        train_epoch_loss = 0.0
        test_epoch_loss = 0.0
        len_training_loader = 0
        len_testing_loader = 0
        net.train()

        # Change the seed at each epoch
        SEED = epoch

        training_files = glob.glob(input_dir + '/training_dataset_fusion_contest_*')
        testing_files = glob.glob(input_dir + '/testing_dataset_fusion_contest_*')
        exit()
        for file in training_files[0:3]:
            nb_file = os.path.basename(file).split('training_dataset_fusion_contest_')[1]
            training_loader = DataFusionContestGenerator(file, os.path.join(input_dir + 'images_training_dataset_fusion_contest_' + nb_file), dataset_cfg)
            training_generator = data.DataLoader(training_loader, **params)
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

            len_training_loader += len(training_loader)

            # Save the network, optimizer, scheduler at each training file
            torch.save({'file': file,
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'train_epoch_loss': train_epoch_loss / len_training_loader},
                       os.path.join(output_dir, 'mc_cnn_acc_epoch' + str(epoch) + '_file_training_dataset_fusion_contest_' + str(nb_file) + '.pt'))
        scheduler.step(epoch)

        net.eval()

        for file in testing_files[0:2]:
            nb_file = os.path.basename(file).split('testing_dataset_fusion_contest_')[1]
            testing_loader = DataFusionContestGenerator(file, os.path.join(input_dir + 'images_testing_dataset_fusion_contest_' + nb_file), dataset_cfg)
            testing_generator = data.DataLoader(testing_loader, **params)
            for it, batch in enumerate(testing_generator, 0):
                # zero the parameter gradients
                optimizer.zero_grad()

                pos, neg = net(batch.to(device=device, dtype=torch.float))

                sample = torch.cat((pos, neg), dim=0)
                sample = torch.squeeze(sample)

                target = torch.cat((torch.ones(batch.size(0)), torch.zeros(batch.size(0))), dim=0)

                loss = criterion.forward(sample, target.to(device=device, dtype=torch.float))

                test_epoch_loss += loss.item() * batch.size(0)

            len_testing_loader += len(testing_loader)

        testing_loss.append(test_epoch_loss / len_testing_loader)

        # Save the network, optimizer, scheduler at each epoch
        torch.save({'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_epoch_loss': train_epoch_loss / len_training_loader,
                    'test_epoch_loss': test_epoch_loss / len_testing_loader},
                   os.path.join(output_dir, 'mc_cnn_acc_epoch' + str(epoch) + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the dataset')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('-data_augmentation', help='Apply data augmentation ?', choices=['True', 'False'], default=False)
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    cfg_data_augmentation = {'transformation': args.data_augmentation, 'dataset_neg_low': 3,
               'dataset_neg_high': 18, 'dataset_pos': 2, 'vertical_disp': 2, 'scale': 0.8, 'hscale': 0.8, 'hshear': 0.1,
               'trans': 0, 'rotate': 28, 'brightness': 1.3, 'contrast': 1.1, 'd_hscale': 0.9, 'd_hshear': 0.3,
               'd_vtrans': 1, 'd_rotate': 3, 'd_brightness': 0.7, 'd_contrast': 1.1}

    train_mc_cnn_acc(args.dataset, args.output_dir, cfg_data_augmentation)