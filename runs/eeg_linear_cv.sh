#!/bin/bash
python3 run_sgmcmc_vr.py \
--cv_lr 5e-6 \
--n_cv_iter 200 \
--psy_type linear \
--var_estimator evm esvm \
--n_train_traj 5 \
--max_sample_size 10000 \
--predictive_distribution \
--n_points 100 \
--data_dir ../mcmc_vr/sgmcmc/data/eeg \
--figs_dir ../mcmc_vr/sgmcmc/eeg/figs \
--metrics_dir ../mcmc_vr/sgmcmc/eeg/metrics \
--prefix_name svrg_ld \
--model_config_path ../mcmc_vr/sgmcmc/eeg/svrg_ld_train_var100_config.json \
--train_trajs_path ../mcmc_vr/sgmcmc/eeg/svrg_ld_train_var100_traj.pkl \
--test_trajs_path ../mcmc_vr/sgmcmc/eeg/svrg_ld_test_var100_traj.pkl \
--dataset uci \
--seed 42 \
--cv_dir ../mcmc_vr/sgmcmc/eeg
#--cv_path ../mcmc_vr/sgmcmc/eeg/svrg_ld_5traj_const_psy.pt