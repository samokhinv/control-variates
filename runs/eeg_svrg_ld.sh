#!/bin/bash
python3 generate_sgmcmc_trajs.py \
--n_trajs 55 \
--n_burn 10000 \
--n_sample 1000 \
--lr 0.1 \
--batch_size 15 \
--alpha 1 \
--beta 1 \
--model_config_path model_configs/log_reg.json \
--resample_prior_every 100000 \
--save_freq 1 \
--sg_mcmc_method svrg-ld \
--seed 42 \
--data_dir ../sgmcmc_data/eeg \
--dataset uci \
--save_path ../sgmcmc_trajs/eeg/svrg_ld \
--epoch_length 200 \
--report_every 3000
