from itertools import repeat

import torch
from torch import nn
from torch.nn import functional as F

from .cv import SteinCV
from .cv_utils import state_dict_to_vec
from joblib import Parallel, delayed

import warnings


@delayed
def _get_prediction(model, x):
    return F.softmax(model(x), dim=-1)[..., -1]


class ClassificationUncertaintyMCMC(object):
    '''
    SO FAR ONLY FOR ONE x
    '''

    def __init__(self, models, control_variate: SteinCV = None):
        self.models = models
        self.control_variate = control_variate

    def parallel_predictions(self, x):
        with Parallel(n_jobs=-2, backend='threading') as p:
            predictions = p(_get_prediction(*args) for args in zip(self.models, repeat(x)))
        return torch.stack(predictions, dim=0).squeeze()

    def get_predictions(self, x):
        predictions = []
        for model in self.models:
            predictions.append(F.softmax(model(x), dim=-1)[..., -1])  # p(y=1|x,\theta)
        return torch.stack(predictions, dim=0).squeeze()

    def get_cv_values(self, x):
        # weights = torch.stack([state_dict_to_vec(model.state_dict()) for model in self.models], dim=0)
        # self.cv_values = torch.stack([self.control_variate(model, x) for model in self.models], dim=0).squeeze()
        if self.control_variate is None:
            raise ValueError('Control variate is undefined')
        return self.control_variate(self.models, x)

    def _calculate_if_needed(self, x, use_cv, predictions, cv_values):
        if predictions is None:
            if x is not None:
                predictions = self.get_predictions(x)
            else:
                raise ValueError('Both predictions and x are undefined')
        if cv_values is None:
            if use_cv:
                if x is not None:
                    cv_values = self.get_cv_values(x)
                else:
                    raise ValueError('Both cv values and x are undefined')
            else:
                cv_values = torch.zeros((len(self.models), x.shape[0]))
        return predictions, cv_values

    def estimate(self, x, use_cv=True):
        predictions, cv_values = self._calculate_if_needed(x, use_cv=True, predictions=None, cv_values=None)
        return self.estimate_emperical_mean(predictions=predictions, cv_values=cv_values, check=False), \
               self.estimate_emperical_variance(predictions=predictions, cv_values=cv_values, check=False)

    def estimate_emperical_mean(self, x=None, use_cv=False, predictions=None, cv_values=None, check=True):
        if check:
            predictions, cv_values = self._calculate_if_needed(x, use_cv, predictions, cv_values)
        return (predictions - cv_values).mean(0)

    def estimate_emperical_variance(self, x=None, use_cv=False, predictions=None, cv_values=None, check=True):
        if check:
            predictions, cv_values = self._calculate_if_needed(x, use_cv, predictions, cv_values)
        return (predictions - cv_values).var(0, unbiased=True)

    def compute_variance_ratio(self, x=None, use_cv=True, predictions=None, cv_values=None, check=True):
        if not use_cv:
            warnings.warn('The ratio is 1. Did you mean use_cv=True?')
        variance_with_cv = self.estimate_emperical_variance(x, use_cv, predictions, cv_values, check)
        variance_without_cv = self.estimate_emperical_variance(x, False, predictions, cv_values, check)

        return variance_with_cv / variance_without_cv
