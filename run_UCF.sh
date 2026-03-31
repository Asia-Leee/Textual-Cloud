#!/bin/bash




for i in {10..12}
do
  AV_coeff=1.0 context_coeff=1.0
  python main.py --epochs_stage1 18  --bs 64 --n_batches 200 --lr 0.00007 --mininterval 600  --lr_scheduler reduce  \
  --exp_name "(${AV_coeff}AV_${context_coeff}context_0.3dropout)_$i"  --num_workers 4  \
  --dataset_name UCF --data_dir "/Dataset/UCF/" \
  --AV_coefficient $AV_coeff  --context_coefficient $context_coeff \
  --dropout 0.1 --hidden_size -1 --weight_decay 1e-5
done






