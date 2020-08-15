import torch
from torch import nn
from torch.nn import functional as F
from control_variates.cv_utils import state_dict_to_vec


class ClassificationUncertaintyMCMC(object):
    def __init__(self, models, control_variate):
        self.models = models
        self.control_variate = control_variate
        self.predictions_storage = None
        self.cv_values = None

    def get_predictions(self, x):
        predictions = []
        for model in self.models:
            predictions.append(F.softmax(model(x), dim=-1))
        predictions = torch.stack(predictions, dim=0)
        self.predictions_storage = predictions

    def get_cv_values(self, x):
        self.cv_values = []
        weights = torch.stack([state_dict_to_vec(model.state_dict() for model in models], dim=0)
        self.cv_values = self.control_variate(wights, x)

    def estimate_emperical_mean(self, x):
        if self.predictions_storage is None:
            self.get_predictions(x)
        emperical_mean = self.predictions_storage.mean(0)
        return emperical_mean

    def estimate_emperical_variance(self, x):
        emperical_mean = self.estimate_emperical_mean(x)
        emperical_variance = ((self.predictions_storage[model_id] - emperical_mean) ** 2).sum(0)
        return emperical_variance / (len(models) - 1)

