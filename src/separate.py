#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import torch

from data import EvalDataLoader, EvalDataset
from conv_tasnet import ConvTasNet
from utils import remove_pad


parser = argparse.ArgumentParser('Separate speech using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate)
    eval_loader =  EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        librosa.output.write_wav(filename, inputs, sr)# norm=True)

    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if args.use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T]
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            # Write result
            for i, filename in enumerate(filenames):
                filename = os.path.join(args.out_dir,
                                        os.path.basename(filename).strip('.wav'))
                write(mixture[i], filename + '.wav')
                C = flat_estimate[i].shape[0]
                for c in range(C):
                    write(flat_estimate[i][c], filename + '_s{}.wav'.format(c+1))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)

