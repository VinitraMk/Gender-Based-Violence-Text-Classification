from catboost import CatBoostClassifier
from models.base_singular_model import BaseModel
from sklearn.multiclass import OneVsOneClassifier
from constants.model_enums import Model

class CTB(BaseModel):

    def __init__(self, ensemble = False):
        BaseModel.__init__(self, Model.CTB, ensemble)
        self.model = OneVsOneClassifier(CatBoostClassifier(
            iterations = self.params['iterations'],
            learning_rate = self.params['alpha'],
            depth = self.params['depth'],
            l2_leaf_reg = self.params['l2_leaf_reg'],
            verbose = False
        ))
