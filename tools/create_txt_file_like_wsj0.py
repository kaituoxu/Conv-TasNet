import os
import random
import pandas as pd
import decimal
import numpy as np


#Making list for tr as used to make mixture like wsj
# list_dir_tr = os.listdir("C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/wsj0-hin/tr/")
# n_tr = int(len(list_dir_tr) / 2)
#
# sublist1_tr = list_dir_tr[0:n_tr]
# sublist2_tr = list_dir_tr[n_tr:]
#
#
# random.shuffle(sublist1_tr)
# random.shuffle(sublist2_tr)
#
# row_list = []
# for i in range(n_tr):
#     spkr1 = sublist1_tr[i]
#     spkr2 = sublist2_tr[i]
#     snr_1 = float(decimal.Decimal(random.randrange(1, 250))/100)
#     snr_2 = -snr_1
#     data = ['wsj0-hin/tr/'+spkr1, snr_1, 'wsj0-hin/tr/'+spkr2, snr_2]
#     row_list.append(data)

# df = pd.DataFrame(row_list)
# df.to_csv('C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/create-speaker-mixtures/mixhin_2_spk_tr.txt',index=False,sep=' ',header=False)

#  ----------------------------------------------- CV AND TT ----------------------------------------

list_dir_cv_tt = os.listdir("C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/wsj0-hin/cv_tt/")
n_cv_tt = int(len(list_dir_cv_tt) / 2)

sublist1_cv_tt = list_dir_cv_tt[0:n_cv_tt]
sublist2_cv_tt = list_dir_cv_tt[n_cv_tt:]

random.shuffle(sublist1_cv_tt)
random.shuffle(sublist2_cv_tt)

#Making list for cv as used to make mixture like wsj
row_list_cv = []
row_list_tt = []
for i in range(n_cv_tt):
    spkr1 = sublist1_cv_tt[i]
    spkr2 = sublist2_cv_tt[i]
    snr_1 = float(decimal.Decimal(random.randrange(1, 250)) / 100)
    snr_2 = -snr_1
    data = ['wsj0-hin/cv_tt/' + spkr1, snr_1, 'wsj0-hin/cv_tt/' + spkr2, snr_2]

    if i < n_cv_tt / 2:
        row_list_cv.append(data)
    else:
        row_list_tt.append(data)

df_cv = pd.DataFrame(row_list_cv)
df_tt = pd.DataFrame(row_list_tt)
df_cv.to_csv('C:/Users/Ofek/PycharmProjects/Conv-TasNet/tools/mix_2_spk_cv.txt', index=False,sep=' ', header=False)
df_tt.to_csv('C:/Users/Ofek/PycharmProjects/Conv-TasNet/tools/mix_2_spk_tt.txt', index=False,sep=' ', header=False)

#
#
# #Making list for tt as used to make mixture like wsj
# row_list = []
# for x in range(100):
#     #wham_noise = random.choice(os.listdir("/media/reverie-pc/speech/asr/kaldi/egs/Neural-mask-estimation/wham_noise/tt/"))
#     spkr1 = random.choice(os.listdir("C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/wsj0-hin/tt/"))
#     spkr2 = random.choice(os.listdir("/C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/wsj0-hin/tr/"))
#     #data = [wham_noise,'wsj0/tr/'+spkr1,'wsj0/tr/'+spkr2]
#     snr_1 = float(decimal.Decimal(random.randrange(1, 250))/100)
#     snr_2 = -snr_1
#     data = ['wsj0-hin/tt/'+spkr1, snr_1 ,'wsj0-hin/tt/'+spkr2, snr_2]
#     row_list.append(data)
# df = pd.DataFrame(row_list)
# #df = pd.read_csv(pd.compat.StringIO(row_list), header=None)
# #df = pd.read_csv(row_list, header=None)
# #df.columns = ['output_filename', 's1_path', 's2_path']
# df.to_csv('C:/Users/Ofek/PycharmProjects/Conv-TasNet/egs/wsj0/create-speaker-mixtures/mixhin_2_spk_tr.txt',index=False,sep=' ',header=False)
# #df_f = df.apppend()
