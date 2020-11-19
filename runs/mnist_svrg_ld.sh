#!/bin/bash
# python3 generate_sgmcmc_trajs.py \
# --n_trajs 20 \
# --n_burn 10000 \
# --n_sample 1000 \
# --lr 0.1 \
# --burn_batch_size 150 \
# --sample_batch_size 150 \
# --alpha 1 \
# --beta 1 \
# --model_config_path model_configs/log_reg.json \
# --resample_prior_every 1000000 \
# --save_freq 1 \
# --sg_mcmc_method svrg-ld \
# --seed 24 \
# --data_dir ../mcmc_vr/sgmcmc/data/mnist \
# --dataset mnist \
# --save_dir ../mcmc_vr/sgmcmc/mnist \
# --prefix_name svrg_ld_train_150_1 \
# --epoch_length 50 \
# --report_every 15000

python3 generate_sgmcmc_trajs.py \
--n_trajs 60 \
--n_burn 5000 \
--n_sample 1000 \
--lr 0.1 \
--burn_batch_size 1024 \
--sample_batch_size 1024 \
--alpha 1 \
--beta 1 \
--model_config_path model_configs/log_reg.json \
--resample_prior_every 1000000 \
--save_freq 1 \
--sg_mcmc_method svrg-ld \
--seed 25 \
--data_dir ../mcmc_vr/sgmcmc/data/mnist \
--dataset mnist \
--save_dir ../mcmc_vr/sgmcmc/mnist \
--prefix_name svrg_ld_test_1024 \
--epoch_length 100 \
--report_every 15000
