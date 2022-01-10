from models.base_singular_model import BaseNNModel
from constants.model_enums import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ANN(BaseNNModel):

    def __init__(self, ensemble = False):
        BaseNNModel.__init__(self, Model.CTB, ensemble)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features = 320, out_features = 162),
            nn.Sigmoid(),
            nn.Linear(in_features = 162, out_features = 81),
            nn.Sigmoid(),
            nn.Linear(in_features = 81, out_features = 5)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = F.sigmoid(x)
        return x

    def apply_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def get_model(self, model):
        model.apply(self.apply_weights)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr = self.params['lr'])
        self.model = {
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer,
            'criterion': self.criterion
        }
        return self.model

