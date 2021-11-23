from constants.model_enums import Model
from sklearn.ensemble import RandomForestClassifier
from models.base_singular_model import BaseModel

class RFA:

    def __init__(self, ensemble = False):
        BaseModel.__init__(self, Model.RFA, ensemble)
        self.model = RandomForestClassifier(
            random_state = 42,
            n_estimators = self.params['n_estimators'],
            max_depth = self.params['max_depth'],
            criterion = self.params['criterion'],
            bootstrap = self.params['bootstrap_samples'],
            min_samples_split = self.params['min_samples_split'],
            min_samples_leaf = self.params['min_samples_leaf'],
            n_jobs = -1,
            warm_start = self.params['warm_start'],
            class_weight = self.params['class_weight'],
            ccp_alpha = self.params['ccp_alpha']
        )
        