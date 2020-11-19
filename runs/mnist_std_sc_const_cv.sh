#!/bin/bash
python3 run_sgmcmc_vr.py \
--cv_lr 1e-5 \
--n_cv_iter 250 \
--psy_type const \
--var_estimator sample \
--n_train_traj 5 \
--max_sample_size 1000 \
--predictive_distribution \
--n_points 100 \
--data_dir ../mcmc_vr/sgmcmc/data/mnist \
--figs_dir ../mcmc_vr/sgmcmc/mnist/figs \
--metrics_dir ../mcmc_vr/sgmcmc/mnist/metrics \
--prefix_name svrg_ld \
--model_config_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_std_sc_train_config.json \
--train_trajs_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_std_sc_train_traj.pkl \
--test_trajs_path ../mcmc_vr/sgmcmc/mnist/svrg_ld_std_sc_test_traj.pkl \
--dataset mnist \
--standard_scale \
--seed 42