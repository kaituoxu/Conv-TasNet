# Created on 2018/12/09
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x K x L (i.e. B x T x D in ASR way)
    Each targets's shape is B x C x K x L
"""

import json
import os

import numpy as np
import torch
import torch.utils.data as data

import librosa


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size,
                 sample_rate=8000, L=int(8000*0.005)):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sorted_s1_infos[start:end],
                              sorted_s2_infos[start:end],
                              sample_rate, L])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x K x L, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x K x L, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    batch = load_mixtures_and_sources(batch[0])
    mixtures, sources = batch

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x K x L x C -> N x C x K x L
    sources_pad = sources_pad.permute((0, 3, 1, 2)).contiguous()
    return mixtures_pad, ilens, sources_pad


# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size,
                 sample_rate=8000, L=int(8000*0.005)):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate, L])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x K x L, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is K x L np.ndarray
        sources: a list containing B items, each item is K x L x C np.ndarray
        K varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, L = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # Generate inputs and targets
        K = int(np.ceil(len(mix) / L))
        # padding a little. mix_len + K > pad_len >= mix_len
        pad_len = K * L
        pad_mix = np.concatenate([mix, np.zeros([pad_len - len(mix)])])
        pad_s1 = np.concatenate([s1, np.zeros([pad_len - len(s1)])])
        pad_s2 = np.concatenate([s2, np.zeros([pad_len - len(s2)])])
        # reshape
        mix = np.reshape(pad_mix, [K, L])
        s1 = np.reshape(pad_s1, [K, L])
        s2 = np.reshape(pad_s2, [K, L])
        # merge s1 and s2
        s = np.dstack((s1, s2))  # K x L x C, C = 2
        # s = np.transpose(s, (2, 0, 1))  # C x K x L

        mixtures.append(mix)
        sources.append(s)
    return mixtures, sources


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is K x L np.ndarray
        filenames: a list containing B strings
        K varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate, L = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        # Generate inputs and targets
        K = int(np.ceil(len(mix) / L))
        # padding a little. mix_len + K > pad_len >= mix_len
        pad_len = K * L
        pad_mix = np.concatenate([mix, np.zeros([pad_len - len(mix)])])
        # reshape
        mix = np.reshape(pad_mix, [K, L])
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
