import numpy as np
from src.preprocess import preprocess
from src.train import train
from src.data import AudioDataset, AudioDataLoader
import visdom

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

# save and visualize

continue_from = ""  # If "" then doesnt use this
print_freq = 10
visdom_enabled = 1
visdom_epoch = 1
visdom_id = "Conv-TasNet Training"
# evaluate
ev_use_cuda = 0
cal_sdr = 1

ngpu = 1  # TODO: What is this

# if __name__ == '__main__':

# preprocess(data_dir, json_dir, sample_rate)
epochs = 30

batch_size = 3
num_workers = 4
# max_hours = 30
# dataset = AudioDataset(train_dir, batch_size, max_hours=30)
# dataloader = AudioDataLoader(dataset, batch_size=1, num_workers=num_workers)
# for data in (dataloader):
#   print('hello')

train(data_dir, epochs)
# vis = visdom.Visdom()
# vis.text("Hello World2!!")