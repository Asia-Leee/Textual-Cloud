#!/bin/bash






for i in {7..8}
do
  AV_coeff=1.0 context_coeff=1.0
  python main.py --epochs_stage1 10  --bs 64 --n_batches 300 --lr 0.0001 --mininterval 600  --lr_scheduler reduce  \
  --exp_name "(${AV_coeff}AV_${context_coeff}context)_$i"  --num_workers 4 \
  --dataset_name ActivityNet --data_dir "" \
  --AV_coefficient $AV_coeff  --context_coefficient $context_coeff \
  --dropout 0.1 --hidden_size -1
done
