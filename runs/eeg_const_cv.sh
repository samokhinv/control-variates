#!/bin/bash
python3 run_cv.py \
--cv_lr 1e-5 \
--n_cv_iter 40 \
--psy_type const \
--var_estimator sample \
--n_train_traj 5 \
--max_sample_size 1000 \
--predictive_distribution \
--n_points 100 \
--data_dir ../sgmcmc_data/eeg \
--figs_dir ../sgmcmc_vr/figs \
--metrics_dir ../sgmcmc_vr/metrics \
--prefix_name svrg_ld \
--model_config_path ../sgmcmc_trajs/eeg/svrg_ld_config.json \
--trajs_path ../sgmcmc_trajs/eeg/svrg_ld_traj.pkl \
--dataset uci \
--seed 42