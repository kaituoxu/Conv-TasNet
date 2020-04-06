import numpy as np

# from src.conv_tasnet import
from src.preprocess import preprocess
from src.train import train
from src.data import AudioDataset, AudioDataLoader

# Trying to imitate the run.sh script from the original github

# I'm using the librispeech dataset, which comes in .flac type, so first thing use flac_to_wav.py
# Then use the "script create_txt_file_like_wsj0.py" to create the txt file, and then
# the matlab code to create the wsj0-2mix dataset

# To open visdom, run this command: "python -m visdom.server", and then open http://localhost:8097

data_dir = "../egs/wsj0-mix/2speakers/wav8k/min/"
json_dir = "../egs/wsj0-mix/2speakers/wav8k/min/"

train_dir = data_dir + "tr"
valid_dir = data_dir + "cv"
test_dir = data_dir + "tt"
sample_rate = 8000

#  TODO: Move some of these parameters to the train.py function


id = 0  # TODO: What is this
epochs = 50  # TODO: Change this before run

# save and visualize
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
    # preprocess(data_dir, json_dir, sample_rate)
    #
    batch_size = 3
    num_workers = 4
    dataset = AudioDataset(train_dir, batch_size)
    dataloader = AudioDataLoader(dataset, batch_size=1, num_workers=num_workers)
    for data in (dataloader):
        print('hello')

    # train(data_dir, epochs)
