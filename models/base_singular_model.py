from helper.utils import get_model_params
import torch
import torch.nn as nn

class BaseModel:
    params = None
    model = None

    def __init__(self, model_name, ensemble):
        self.params = get_model_params(ensemble, model_name)

    def get_model(self):
        return self.model


class BaseNNModel(nn.Module):
    params = None
    model = None

    def __init__(self, model_name, ensemble):
        super(BaseNNModel, self).__init__()
        self.params = get_model_params(ensemble, model_name)
