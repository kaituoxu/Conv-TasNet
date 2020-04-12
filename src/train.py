#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU


import torch

from src.data import AudioDataLoader, AudioDataset
from src.solver import Solver
from src.conv_tasnet import ConvTasNet


def train(data_dir, epochs, max_hours=None):
    # General config
    # Task related
    json_dir = data_dir
    train_dir = data_dir + "tr"
    valid_dir = data_dir + "cv"
    sample_rate = 8000
    segment_len = 4
    cv_maxlen = 6

    # Network architecture
    N = 256  # Number of filters in autoencoder
    L = 20  # Length of filters in conv autoencoder
    B = 256  # number of channels in conv blocks - after bottleneck 1x1 conv
    H = 512  # number of channels in inner conv1d block
    P = 3  # length of filter in inner conv1d blocks
    X = 8  # number of conv1d blocks in (also number of dilations) in each repeat
    R = 4  # number of repeats
    C = 2  # Number of speakers

    norm_type = 'gLN'  # choices=['gLN', 'cLN', 'BN']
    causal = 0
    mask_nonlinear = 'relu'

    use_cuda = 1

    half_lr = 1  # Half the learning rate when there's a small improvement
    early_stop = 1  # Stop learning if no imporvement after 10 epochs
    max_grad_norm = 5  # gradient clipping

    shuffle = 1  # Shuffle every epoch
    batch_size = 3
    num_workers = 4
    # optimizer
    optimizer_type = "adam"
    lr = 1e-3
    momentum = 0
    l2 = 0  # Weight decay - l2 norm

    # save and visualize
    save_folder = "../egs/models"
    enable_checkpoint = 0  # enables saving checkpoints
    continue_from = save_folder + "/speech_seperation_first_try.pth"  # model to continue from
    model_path = "speech_separation_first_try_more_epochs.pth"  # TODO: Fix this
    print_freq = 20000
    visdom_enabled = 1
    visdom_epoch = 1
    visdom_id = "Conv-TasNet Training"  # TODO: Check what this does

    arg_solver = (use_cuda, epochs, half_lr, early_stop, max_grad_norm, save_folder, enable_checkpoint, continue_from,
                  model_path, print_freq, visdom_enabled, visdom_epoch, visdom_id)

    # Datasets and Dataloaders
    tr_dataset = AudioDataset(train_dir, batch_size,
                              sample_rate=sample_rate, segment=segment_len, max_hours=max_hours)
    cv_dataset = AudioDataset(valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=sample_rate,
                              segment=-1, cv_maxlen=cv_maxlen)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = ConvTasNet(N, L, B, H, P, X, R,
                       C, norm_type=norm_type, causal=causal,
                       mask_nonlinear=mask_nonlinear)
    # print(model)
    if use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=l2)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizer, arg_solver)  # TODO: Fix solver thing
    solver.train()


if __name__ == '__main__':
    print('train main')
    # args = parser.parse_args()
    # print(args)
    # train(args)
