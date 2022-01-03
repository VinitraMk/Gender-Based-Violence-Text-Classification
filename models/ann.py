from models.base_singular_model import BaseNNModel
from constants.model_enums import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ANN(BaseNNModel):

    def __init__(self, ensemble = False):
        BaseNNModel.__init__(self, Model.CTB, ensemble)
        self.fc_layer = nn.Linear(in_features = 319, out_features = 162)
        self.output_layer = nn.Linear(in_features = 162, out_features = 5)

    def forward(self, x):
        x = F.relu(self.fc_layer(x))
        x = self.output_layer(x)
        return x

    def get_model(self, model):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), self.params['lr'], momentum = 0.9)
        self.model = model_object = {
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer,
            'criterion': self.criterion
        }
        return self.model

