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
#python train_eval.py --config ./config/baseline/single_vertical.json &
#python train_eval.py --config ./config/baseline/single_diagonal-25.json &
#python train_eval.py --config ./config/baseline/single_diagonal-50.json &
#python train_eval.py --config ./config/baseline/single_parallel.json &
#python train_eval.py --config ./config/baseline/sequential_in-order.json &
#python train_eval.py --config ./config/baseline/sequential_reverse-order.json &
#python train_eval.py --config ./config/baseline/interleaved.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical.json &
#python train_eval.py --config ./config/ppo-ewc/single_diagonal-25.json &
#python train_eval.py --config ./config/ppo-ewc/single_diagonal-50.json &
#python train_eval.py --config ./config/ppo-ewc/single_parallel.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_num_cells-128.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_num_cells-256.json &
#python train_eval.py --config ./config/ppo-ewc/sequential_in-order.json &
#python train_eval.py --config ./config/ppo-ewc/sequential_in-order_num_cells-128.json &
#python train_eval.py --config ./config/ppo-ewc/sequential_in-order_num_cells-256.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_lambda-0.25.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_lambda-1.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_lambda-0.9.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_discount-0.5.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_discount-0.999.json &
#python train_eval.py --config ./config/ppo-ewc/single_vertical_ewc_discount-1.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_discount-0.5.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_discount-0.999.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_discount-1.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_lambda-0.9.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_lambda-0.25.json &
python train_eval.py --config ./config/ppo-ewc/sequential_in-order_ewc_lambda-1.json &
#python train_eval.py --config ./config/ewc_test.json &
wait
