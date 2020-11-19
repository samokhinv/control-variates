from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm, trange
import dill as pickle
from pathlib import Path
import json

from .cv_utils import (
        SampleVarianceEstimator, 
        SpectralVarianceEstimator, 
    )

from .cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from .cv_utils import state_dict_to_vec
from .uncertainty_quantification import ClassificationUncertaintyMCMC


def test_cv(trajs, ncvs, batches, function_f):
    if isinstance(ncvs, SteinCV):
        ncvs = [ncvs]
    n_batches = len(batches)
    metrics = {'sample_var': 0., 'sample_var_cv': 0, 'sample_var_reduction': 0., 
                'spectral_var': 0., 'spectral_var_cv': 0., 'spectral_var_reduction': 0.}

    trajs, traj_weights, traj_grads = trajs
    mean_avg_pred = torch.zeros(len(trajs))
    mean_avg_pred_cv = torch.zeros(len(ncvs), len(trajs))

    for x in tqdm(batches):
        avg_predictions = torch.zeros(len(trajs))
        avg_predictions_cv = torch.zeros(len(ncvs), len(trajs))

        for tr_id, (traj, traj_weight, traj_grad) in tqdm(enumerate(zip(trajs, traj_weights, traj_grads)), leave=False):
            pred = function_f(traj, x)
            avg_predictions[tr_id] = pred.mean()
            for cv_id, ncv in enumerate(ncvs):
                cv_vals = ncv(traj_weight, x, potential_grad=traj_grad)
                # function
                avg_predictions_cv[cv_id, tr_id] = (pred - cv_vals).mean()

            #uncertainty_quant = ClassificationUncertaintyMCMC(models, ncv)

            # metrics['sample_var'] += sample_var.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_cv'] += sample_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_reduction'] += (sample_var / sample_var_cv).mean().item() / (n_batches * n_traj)
            # metrics['spectral_var'] += spectral_var.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_cv'] += spectral_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_reduction'] += (spectral_var / spectral_var_cv).mean().item() / (n_batches * n_traj)

        mean_avg_pred = mean_avg_pred + avg_predictions / n_batches
        mean_avg_pred_cv = mean_avg_pred_cv + avg_predictions_cv / n_batches

    return mean_avg_pred.detach().cpu().numpy(), mean_avg_pred_cv.detach().cpu().numpy(), metrics


def train_cv(trajs, ncv, batches, function_f, var_estimator, preds=None,
                cv_lr=1e-6, n_cv_iter=40, centr_reg_coef=0.0, predictive_distribution=True):
    # def centr_regularizer(ncv, traj, x, potential_grads=None):
    #     return (ncv(traj, x, potential_grads).mean(0))**2

    trajs, traj_weights, traj_grads = trajs

    for x in tqdm(batches):
        ncv_optimizer = torch.optim.Adam(ncv.psy_model.parameters(), lr=cv_lr)

        loss_history = []
        if preds is None:
            preds = []
            for traj in trajs:
                preds.append(function_f(traj, x))
            preds = torch.stack(preds, dim=0)

        chunks = [(traj_weights[i:i + 5], traj_grads[i:i + 5]) for i in range(0, len(trajs), 5)]
        for _ in trange(n_cv_iter):
            cv_vals = []
            ncv_optimizer.zero_grad()
            
            for chunk in chunks:
                traj_weights, traj_grads = chunk
                cv_vals.append(ncv(traj_weights, x, potential_grad=traj_grads))
            cv_vals = torch.cat(cv_vals, 0)
            #for tr_id, (traj, traj_weight, traj_grad) in tqdm(enumerate(zip(trajs, traj_weights, traj_grads)), leave=False)
            #cv_vals = ncv(traj_weights, x, potential_grad=traj_grads)
            processed_trajs = preds - cv_vals

            # posterior_mean
            processed_trajs = processed_trajs.mean(-1)
            
            loss = var_estimator.estimate_variance(processed_trajs).mean()

            #         if centr_reg_coef != 0:
            #             loss += centr_reg_coef * centr_regularizer(ncv, traj, x).mean()

            loss_history.append(loss.mean().item())
            #print('hi')
            loss.backward()
            ncv_optimizer.step()
        print(ncv.psy_model.state_dict())

        return loss_history, preds
