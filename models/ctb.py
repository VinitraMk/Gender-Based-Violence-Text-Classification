from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from helper.utils import get_model_params
from constants.model_enums import Model

class CTB:
    model = None

    def __init__(self, ensemble = False):
        model_params = get_model_params(ensemble, Model.CTB)
        self.model = OneVsOneClassifier(CatBoostClassifier(
            iterations = model_params['iterations'],
            learning_rate = model_params['alpha'],
            depth = model_params['depth'],
            l2_leaf_reg = model_params['l2_leaf_reg'],
            verbose = False
        ))

    def get_model(self):
        return self.model