from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm, trange
import dill as pickle
from pathlib import Path
import json

# from .cv_utils import (
#         SampleVarianceEstimator, 
#         SpectralVarianceEstimator, 
#         state_dict_to_vec,
#         compute_naive_variance,
#     )

from .cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from .uncertainty_quantification import ClassificationUncertaintyMCMC


def test_cv(trajs, ncv, batches, function_f):
    print(ncv.psy_model.state_dict())
    n_traj = len(trajs)
    n_batches = len(batches)
    
    mean_avg_pred = np.zeros(n_traj)
    mean_avg_pred_cv = np.zeros(n_traj)

    metrics = {'sample_var': 0., 'sample_var_cv': 0, 'sample_var_reduction': 0., 
                'spectral_var': 0., 'spectral_var_cv': 0., 'spectral_var_reduction': 0.}

    for x in tqdm(batches):
        avg_predictions_cv = []
        avg_predictions_no_cv = []

        for (traj, potential_grads) in trajs:
            #uncertainty_quant = ClassificationUncertaintyMCMC(models, ncv)
            #sample_var_estimator = SampleVarianceEstimator(function_f, bayesian_nns)
            #spectral_var_estimator = SpectralVarianceEstimator(function_f, models)

            pred = function_f(traj, x)
            cv_vals = ncv(traj, x, potential_grad=potential_grads)

            avg_predictions_cv.append((pred-cv_vals).mean().item())
            avg_predictions_no_cv.append(pred.mean().item())

            #sample_var_cv = sample_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            #sample_var = sample_var_estimator.estimate_variance(x, use_cv=False, all_values=pred)
            
            #spectral_var_cv = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            #spectral_var = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=pred)

            # metrics['sample_var'] += sample_var.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_cv'] += sample_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_reduction'] += (sample_var / sample_var_cv).mean().item() / (n_batches * n_traj)
            # metrics['spectral_var'] += spectral_var.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_cv'] += spectral_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_reduction'] += (spectral_var / spectral_var_cv).mean().item() / (n_batches * n_traj)

            # avg_predictions_cv.append(
            #   uncertainty_quant.estimate_emperical_mean(x=x, predictions=pred, cv_values=cv_vals, use_cv=True).mean().item())
            # avg_predictions_no_cv.append(
            #   uncertainty_quant.estimate_emperical_mean(x=x, predictions=pred, use_cv=False).mean().item())

        mean_avg_pred += np.array(avg_predictions_no_cv) / n_batches
        mean_avg_pred_cv += np.array(avg_predictions_cv) / n_batches

    return mean_avg_pred, mean_avg_pred_cv, metrics


def train_cv(trajs, ncv, batches, function_f, var_estimator,
                cv_lr=1e-6, n_cv_iter=40, centr_reg_coef=0.0, predictive_distribution=True):
    def centr_regularizer(ncv, traj, x, potential_grads=None):
        return (ncv(traj, x, potential_grads).mean(0))**2

    for x in tqdm(batches):
        ncv_optimizer = torch.optim.SGD(ncv.psy_model.parameters(), lr=cv_lr, momentum=1e-3, nesterov=True)
        loss_history = []
        preds = []
        for it in range(n_cv_iter):
            ncv_optimizer.zero_grad()
            def closure():
                loss = 0
                for traj_id, (traj, potential_grads) in enumerate(trajs):
                    if it == 0:
                        preds.append(function_f(traj, x))
                    pred = preds[traj_id]

                    var_estimator.traj = traj

                    cv_vals = ncv(traj, x, potential_grad=potential_grads)

                    if predictive_distribution is True:
                        pred = pred.mean(1)
                        cv_vals = cv_vals.mean(1)

                    #var_cv = var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))

                    loss += (pred - cv_vals).std() #var_cv.mean()
                    if centr_reg_coef != 0:
                        loss += centr_reg_coef * centr_regularizer(ncv, traj, x).mean()
                return loss
            loss = closure()

            loss_history.append(loss.mean().item())
            loss.backward()
            ncv_optimizer.step(closure=closure)

        return loss_history
