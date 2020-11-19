#!/bin/bash
python3 run_sgmcmc_vr.py \
--cv_lr 1e-4 \
--n_cv_iter 50 \
--psy_type const \
--var_estimator esvm \
--n_train_traj 20 \
--max_sample_size 1000 \
--predictive_distribution \
--n_points 100 \
--data_dir ../mcmc_vr/sgmcmc/data/mnist \
--figs_dir ../mcmc_vr/sgmcmc/mnist/figs \
--metrics_dir ../mcmc_vr/sgmcmc/mnist/metrics \
--prefix_name svrg_ld \
--model_config_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_train_150_1_config.json \
--train_trajs_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_train_150_1_traj.pkl \
--test_trajs_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_test_150_1_traj.pkl \
--cv_dir ../mcmc_vr/sgmcmc/mnist \
--dataset mnist \
--seed 42
#--cv_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_5traj_const_psy.pt