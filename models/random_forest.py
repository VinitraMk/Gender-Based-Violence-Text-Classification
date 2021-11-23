from helper.utils import get_model_params
from constants.model_enums import Model
from sklearn.ensemble import RandomForestClassifier
class RFA:
    model = None

    def __init__(self, ensemble = False):
        model_params = get_model_params(ensemble, Model.RFA)
        self.model = RandomForestClassifier(
            random_state = 42,
            n_estimators = model_params['n_estimators'],
            max_depth = model_params['max_depth'],
            criterion = model_params['criterion'],
            bootstrap = model_params['bootstrap_samples'],
            min_samples_split = model_params['min_samples_split'],
            min_samples_leaf = model_params['min_samples_leaf'],
            n_jobs = -1,
            warm_start = model_params['warm_start'],
            class_weight = model_params['class_weight'],
            ccp_alpha = model_params['ccp_alpha']
        )

    def get_model(self):
        return self.model


        