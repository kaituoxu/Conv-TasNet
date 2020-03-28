import glob
import os, random
from pathlib import Path
import shutil
from shutil import copyfile

""" Currently I don't use this file, but it basically moves a directory """


tr_data_path = glob.glob('wav_data/wav_data_val_test/*.wav')

dst_tr = 'flac_val_test_data'
for i, file in enumerate(tr_data_path):
    #print(file)
    #os.path.split(file)[1]
    #shutil.copy(file, dst_tr+os.path.split(file)[1])
    shutil.copy(file, dst_tr+file.split(sep='/')[8]+'_'+file.split(sep='/')[9]+'_'+file.split(sep='/')[10]+'_'+file.split(sep='/')[11]+'_'+os.path.split(file)[1])
 
 # Uncomment it to make for CV an tt   
'''
source = "/media/reverie-pc/speech/asr/kaldi/egs/Conv-TasNet/egs/wsj0/wsj0-hin/tr/"
dst_cv = '/media/reverie-pc/speech/asr/kaldi/egs/Conv-TasNet/egs/wsj0/wsj0-hin/cv/'
for f in range(100):
    spkr1 = random.choice(os.listdir(source))
    shutil.move(source+spkr1, dst_cv)


source = "/media/reverie-pc/speech/asr/kaldi/egs/Conv-TasNet/egs/wsj0/wsj0-hin/tr/"
dst_tt = '/media/reverie-pc/speech/asr/kaldi/egs/Conv-TasNet/egs/wsj0/wsj0-hin/tt/'
for f in range(100):
    spkr1 = random.choice(os.listdir(source))
    shutil.move(source+spkr1, dst_tt)        '''


