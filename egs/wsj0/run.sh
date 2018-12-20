#!/bin/bash

# Created on 2018/12/10
# Author: Kaituo XU

stage=2

ngpu=1
dumpdir=data

# -- START TasNet Config
sample_rate=8000
# Network config
L=40
N=500
hidden_size=500
num_layers=4
bidirectional=1
nspk=2
# Training config
epochs=30
shuffle=0
half_lr=0
early_stop=0
max_norm=5
# minibatch
batch_size=128
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=0
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="TasNet Training"
# -- END TasNet Config

# exp tag
tag="" # tag for managing experiments.

wsj0_origin=/home/ktxu/workspace/data/CSR-I-WSJ0-LDC93S6A
wsj0_wav=/home/ktxu/workspace/data/wsj0-wav/wsj0
# Directory path of wsj0 including tr, cv and tt
data=/home/work_nfs/ktxu/data/wsj-mix/2speakers/wav8k/min/

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 0 ]; then
  echo "Stage 0: Convert sphere format to wav format and generate mixture"
  local/data_prepare.sh --data ${wsj0_origin} --wav_dir ${wsj0_wav}

  echo "NOTE: You should generate mixture by yourself now.
You can use tools/create-speaker-mixtures.zip which is download from
http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip
If you don't have Matlab and want to use Octave, I suggest to replace
all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...])
due to mkdir in Octave can not work in 'mkdir -p' way.
e.g.:
mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
->
system(['mkdir -p ' output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);"
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi


if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_L${L}_N${N}_h${hidden_size}_l${num_layers}_bi${bidirectional}_C${nspk}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}
else
  expdir=exp/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    train.py \
    --train_dir $dumpdir/tr \
    --valid_dir $dumpdir/cv \
    --sample_rate $sample_rate \
    --L $L \
    --N $N \
    --hidden_size $hidden_size \
    --num_layers $num_layers \
    --bidirectional $bidirectional \
    --nspk $nspk \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id"
fi


if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  ${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
    evaluate.py \
    --model_path ${expdir}/final.pth.tar \
    --data_dir $dumpdir/tt \
    --cal_sdr 1 \
    --use_cuda 0 \
    --sample_rate $sample_rate \
    --batch_size 10
fi


if [ $stage -le 4 ]; then
  echo "Stage 4: Separate speech using TasNet"
  separate_dir=${expdir}/separate
  ${decode_cmd} --gpu ${ngpu} ${separate_dir}/separate.log \
    separate.py \
    --model_path ${expdir}/final.pth.tar \
    --mix_json $dumpdir/tt/mix.json \
    --out_dir ${separate_dir} \
    --use_cuda 0 \
    --sample_rate $sample_rate \
    --batch_size 10
fi
