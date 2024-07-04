#!/bin/bash
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-3.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-6.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-19.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-26.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-49.json &
#python hyperparameter_tuning.py --start 51 --n-iter 9 --output ./results/hyperparameter_tuning_2 &
#python hyperparameter_tuning.py --start 62 --n-iter 8 --output ./results/hyperparameter_tuning_2 &
#python hyperparameter_tuning.py --start 70 --n-iter 10 --output ./results/hyperparameter_tuning_2 &
#python hyperparameter_tuning.py --start 81 --n-iter 9 --output ./results/hyperparameter_tuning_2 &
#python hyperparameter_tuning.py --start 91 --n-iter 9 --output ./results/hyperparameter_tuning_2 &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-64.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-96.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-69.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-78.json &
#python train_eval.py --config ./config/tuning/7_hyperparameters_id-99.json &
python train_eval.py --config ./config/baseline/single_vertical.json &
python train_eval.py --config ./config/baseline/single_diagonal-25.json &
python train_eval.py --config ./config/baseline/single_diagonal-50.json &
python train_eval.py --config ./config/baseline/single_parallel.json &
wait
