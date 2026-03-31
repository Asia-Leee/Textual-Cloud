#!/bin/bash




for i in {1..2}
do
  AV_coeff=1.0 context_coeff=1.0
  python main.py --epochs_stage1 10  --bs 64 --n_batches 300 --lr 0.0001 --mininterval 600  --lr_scheduler reduce \
  --exp_name "(${AV_coeff}AV_${context_coeff}context_10words)_$i"  --num_workers 2 \
  --dataset_name VGGSound --data_dir "/Dataset/VGGSound/" \
  --AV_coefficient $AV_coeff  --context_coefficient $context_coeff \
  --dropout 0.3 --hidden_size 512 --xiaorong_textual_cloud 'visual+context'
done


