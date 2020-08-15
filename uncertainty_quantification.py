import torch
from torch import nn
from torch.nn import functional as F
from control_variates.cv_utils import state_dict_to_vec


class ClassificationUncertaintyMCMC(object):
    '''
    SO FAR ONLY FOR ONE x
    '''
    def __init__(self, models, control_variate=None):
        self.models = models
        self.control_variate = control_variate
        self.predictions_storage = None
        self.cv_values = torch.zeros(len(models)) # n_classes

    def get_predictions(self, x):
        predictions = []
        for model in self.models:
            predictions.append(F.softmax(model(x)[:, 1], dim=-1)) # p(y=1|x,\theta)
        predictions = torch.stack(predictions, dim=0).squeeze()
        self.predictions_storage = predictions

    def get_cv_values(self, x):
        #weights = torch.stack([state_dict_to_vec(model.state_dict()) for model in self.models], dim=0)
        self.cv_values = torch.stack([self.control_variate(model, x) for model in self.models], dim=0).squeeze()
        return self.cv_values

    def estimate_emperical_mean(self, x):
        if self.predictions_storage is None:
            self.get_predictions(x)
        if self.control_variate is not None:
            self.get_cv_values(x)
        emperical_mean = self.predictions_storage.mean(0) - self.cv_values.mean(0)
        return emperical_mean

    def estimate_emperical_variance(self, x):
        emperical_mean = self.estimate_emperical_mean(x)
        emperical_variance = ((self.predictions_storage - self.cv_values - emperical_mean) ** 2).sum(0)
        return emperical_variance / (len(self.models) - 1)

