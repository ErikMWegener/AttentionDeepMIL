#!/bin/bash
# ============================================================
# Record of the local main_mlflow.py calls of the early experiments
# (MLflow: compare_datasets, sigmoid_v_softmax, first_results,
# gwhd_model_parameter_sweep). Commented lines were run at an earlier time.
# The MNIST calls use the beta datasets (mnist_bags.h5), which are not part of the thesis.
# Run from runs/.
# ============================================================
#MNIST runs
python3 main_mlflow.py --dataset mnist_bags_base --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name compare_datasets --log_attention_weights
python3 main_mlflow.py --dataset mnist_bags1000_beta1.0 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name compare_datasets --log_attention_weights
python3 main_mlflow.py --dataset mnist_bags1000_beta2.0 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name compare_datasets --log_attention_weights 
python3 main_mlflow.py --dataset mnist_bags1000_beta1.5 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name compare_datasets --log_attention_weights
python3 main_mlflow.py --dataset mnist_bags1000_beta2.5 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name compare_datasets --log_attention_weights

# python3 main_mlflow.py --dataset mnist_bags_base --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name sigmoid_v_softmax --sigmoid_attention
# python3 main_mlflow.py --dataset mnist_bags1000_beta1.0 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name sigmoid_v_softmax --sigmoid_attention
# python3 main_mlflow.py --dataset mnist_bags1000_beta2.0 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name sigmoid_v_softmax --sigmoid_attention
# python3 main_mlflow.py --dataset mnist_bags1000_beta1.5 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name sigmoid_v_softmax --sigmoid_attention 
# python3 main_mlflow.py --dataset mnist_bags1000_beta2.5 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --exp_name sigmoid_v_softmax --sigmoid_attention 

#python3 main_mlflow.py --dataset mnist_bags_beta2.0 --path ../data/datasets/bags/mnist_bags.h5 --naive_counting --seeds 1 10 100 --attention_activation min_max --log_attention_weights

# #GWHD run
#python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 --sigmoid_attention --log_attention_weights --exp_name sigmoid_v_softmax
#python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 --log_attention_weights --exp_name sigmoid_v_softmax
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 2048 --model_L 128 --model_pool_size 8 --model_num_maps 128 --naive_counting --seeds 1 10 100 --sigmoid_attention --log_attention_weights --exp_name sigmoid_v_softmax
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 2048 --model_L 128 --model_pool_size 8 --model_num_maps 128 --naive_counting --seeds 1 10 100 --log_attention_weights --exp_name sigmoid_v_softmax
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model GatedAttention --naive_counting --seeds 1 10 100 --log_attention_weights --exp_name sigmoid_v_softmax
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model GatedAttention --naive_counting --seeds 1 10 100 --log_attention_weights --exp_name sigmoid_v_softmax --sigmoid_attention
# python3 main_mlflow.py --dataset gwhd_bags_dense_color --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --naive_counting --seeds 1 10 100 --exp_name first_results --sigmoid_attention --rgb
# python3 main_mlflow.py --dataset gwhd_bags --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --naive_counting --seeds 1 10 100 --exp_name first_results --sigmoid_attention 
# python3 main_mlflow.py --dataset gwhd_bags --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --naive_counting --seeds 1 10 100 --exp_name first_results 

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 4 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 500 --model_L 128 --model_pool_size 8 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 4 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 5 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 7 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep

# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --sigmoid_attention --log_attention_weights --exp_name gwhd_model_parameter_sweep
# python3 main_mlflow.py --dataset gwhd_bags_dense --path ../data/datasets/bags/gwhd_bags.h5 --epochs 50 --model_M 1024 --model_L 128 --model_pool_size 8 --model_kernel_size 9 --model_num_maps 128 --naive_counting --seeds 1 10 100 1000 10000 --log_attention_weights --exp_name gwhd_model_parameter_sweep
