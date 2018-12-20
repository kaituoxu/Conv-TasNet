#!/usr/bin/env python

# Created on 2018/12/14
# Author: Kaituo XU

import argparse

import torch

from data import AudioDataLoader, AudioDataset
from solver import Solver
from tasnet import TasNet

parser = argparse.ArgumentParser(
    "Time-domain Audio Separation Network (TasNet) with Permutation Invariant "
    "Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
# Network architecture
parser.add_argument('--L', default=40, type=int,
                    help='Segment length (40=5ms at 8kHZ)')
parser.add_argument('--N', default=500, type=int,
                    help='The number of basis signals')
parser.add_argument('--hidden_size', default=500, type=int,
                    help='Number of LSTM hidden units')
parser.add_argument('--num_layers', default=4, type=int,
                    help='Number of LSTM layers')
parser.add_argument('--bidirectional', default=1, type=int,
                    help='Whether use bidirectional LSTM')
parser.add_argument('--nspk', default=2, type=int,
                    help='Number of speaker')
# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', '-b', default=128, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate, L=args.L)
    cv_dataset = AudioDataset(args.valid_dir, args.batch_size,
                              sample_rate=args.sample_rate, L=args.L)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = TasNet(args.L, args.N, args.hidden_size, args.num_layers,
                   bidirectional=args.bidirectional, nspk=args.nspk)
    print(model)
    model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
