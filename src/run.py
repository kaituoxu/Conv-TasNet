import numpy as np

# from src.conv_tasnet import
from src.preprocess import preprocess
from src.train import train
from src.data import AudioDataset, AudioDataLoader

# Trying to imitate the run.sh script from the original github

# I'm using the librispeech dataset, which comes in .flac type, so first thing use flac_to_wav.py
# Then use the "script Making_list_for_mixture_like_wsj-matlab-Hindi.py" to create the txt file, and then
# the matlab code to create the wsj0-2mix dataset

data_dir = "../egs/wsj0-mix/2speakers/wav8k/min/"  # TODO: Check if I should use min or max
json_dir = "../egs/wsj0-mix/2speakers/wav8k/min/"
sample_rate = 8000
# preprocess(data_dir, json_dir, sample_rate)

#  TODO: Move some of these parameters to the train.py function
train_dir = data_dir + "tr"
valid_dir = data_dir + "cv"
test_dir = data_dir + "tt"




id = 0  # TODO: What is this
epochs = 100   # TODO: Change this before run
half_lr = 1  # TODO: What is this
early_stop = 0
max_norm = 5  # TODO: What is this
# minibatch
shuffle = 1
batch_size = 3
num_workers = 4
# optimizer
optimizer = "adam"
lr = 1e-3
momentum = 0
l2 = 0
# save and visualize
checkpoint = 0
continue_from = ""
print_freq = 10
visdom = 0
visdom_epoch = 0
visdom_id = "Conv-TasNet Training"
# evaluate
ev_use_cuda = 0
cal_sdr = 1

ngpu = 1  # TODO: What is this

if __name__ == '__main__':
    batch_size = 4
    dataset = AudioDataset("../egs/wsj0-mix/2speakers/wav8k/min/tr", batch_size)
    dataloader = AudioDataLoader(dataset, batch_size=1, num_workers=num_workers)
    for data in (dataloader):
        print('hello')



