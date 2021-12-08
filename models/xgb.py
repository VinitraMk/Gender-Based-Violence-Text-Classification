from xgboost import XGBClassifier
from constants.model_enums import Model
from models.base_singular_model import BaseModel

class XGB(BaseModel):

    def __init__(self, ensemble = False):
        BaseModel.__init__(self, Model.XGB, ensemble)
        self.model = XGBClassifier(
            random_state = 42,
            seed = 2,
            objective = 'multi:softmax',
            eval_metric = 'merror',
            use_label_encoder = False,
            learning_rate = self.params['alpha'],
            n_jobs = -1,
            num_classes = 5,
            colsample_bytree=1.0,
            n_estimators = self.params['n_estimators'],
            min_split_loss = self.params['min_split_loss'],
            tree_method = self.params['tree_method'],
            grow_policy = self.params['grow_policy'],
            single_precision_histogram = self.params['single_precision_histogram'],
            nthread = -1,
            max_depth = self.params['max_depth']
        )
