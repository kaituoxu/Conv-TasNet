#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import torch

from src.data import EvalDataLoader, EvalDataset
from src.conv_tasnet import ConvTasNet
from src.utils import remove_pad


def separate(model_path, mix_dir, mix_json, out_dir, use_cuda, sample_rate, batch_size):
    if mix_dir is None and mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    model = ConvTasNet.load_model(model_path)
    print(model)
    model.eval()
    if use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(mix_dir, mix_json,
                               batch_size=batch_size,
                               sample_rate=sample_rate)
    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs(out_dir, exist_ok=True)

    def write(inputs, filename, sr=sample_rate):
        librosa.output.write_wav(filename, inputs, sr)# norm=True)

    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T]
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            # Write result
            for i, filename in enumerate(filenames):
                filename = os.path.join(out_dir,
                                        os.path.basename(filename).strip('.wav'))
                write(mixture[i], filename + '.wav')
                C = flat_estimate[i].shape[0]
                for c in range(C):
                    write(flat_estimate[i][c], filename + '_s{}.wav'.format(c+1))


if __name__ == '__main__':
    model_path = ""  # TODO: Add this
    mix_dir = ""  # TODO: Add this, indlues dir for wav files
    mix_json = "" # TODO: Includes json file
    out_dir = ""
    use_cuda = 1
    sample_rate = 8000
    batch_size = 4

    separate(model_path, mix_dir, mix_json, out_dir, use_cuda, sample_rate, batch_size)

