from itertools import repeat

import torch
from torch import nn
from torch.nn import functional as F

from control_variates.cv import SteinCV
from control_variates.cv_utils import state_dict_to_vec
from joblib import Parallel, wrap_non_picklable_objects, delayed


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
        self.predictions_storage = None
        self.cv_values = torch.zeros(len(models)) # n_classes

    def parallel_predictions(self, x):
        with Parallel(n_jobs=-2, backend='threading') as p:
            predictions = p(_get_prediction(*args) for args in zip(self.models, repeat(x)))
        return torch.stack(predictions, dim=0).squeeze()

    def get_predictions(self, x):
        predictions = []
        for model in self.models:
            predictions.append(F.softmax(model(x), dim=-1)[..., -1]) # p(y=1|x,\theta)
        predictions = torch.stack(predictions, dim=0).squeeze()
        self.predictions_storage = predictions

    def clean(self):
        self.predictions_storage = None
        self.cv_values = torch.zeros(len(self.models))

    def get_cv_values(self, x):
        #weights = torch.stack([state_dict_to_vec(model.state_dict()) for model in self.models], dim=0)
        #self.cv_values = torch.stack([self.control_variate(model, x) for model in self.models], dim=0).squeeze()
        self.cv_values = self.control_variate(self.models, x)  #  вроде мы к такому вызову стремимся, и он сейчас работает
        return self.cv_values

    def estimate_emperical_mean(self, x, use_cv=True):
        if self.predictions_storage is None:
            self.get_predictions(x)
        if use_cv:
            if self.control_variate is not None:
                self.get_cv_values(x)
            emperical_mean = self.predictions_storage.mean(0) - self.cv_values.mean(0)
        else:
            emperical_mean = self.predictions_storage.mean(0)
        return emperical_mean

    def estimate_emperical_variance(self, x, use_cv=True):
        emperical_mean = self.estimate_emperical_mean(x, use_cv)
        if use_cv:
            emperical_variance = ((self.predictions_storage - self.cv_values - emperical_mean) ** 2).sum(0)
        else:
            emperical_variance = ((self.predictions_storage - emperical_mean) ** 2).sum(0)
        return emperical_variance / (len(self.models) - 1)

    def compute_variance_ratio(self, batch_x):
        if self.control_variate is None:
            raise Exception
        variance_with_cv = self.estimate_emperical_variance(batch_x, True)
        variance_without_cv = self.estimate_emperical_variance(batch_x, False)

        return (variance_with_cv / variance_without_cv)


