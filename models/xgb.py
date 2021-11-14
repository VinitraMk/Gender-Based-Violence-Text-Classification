from helper.utils import get_model_params
from xgboost import XGBClassifier

class XGB:
    model = None

    def __init__(self):
        model_params = get_model_params()
        self.model = XGBClassifier(
            random_state = 42,
            seed = 2,
            objective = 'multi:softmax',
            eval_metric = 'merror',
            use_label_encoder = False,
            learning_rate = model_params['alpha'],
            n_jobs = -1,
            num_classes = 5,
            n_estimators = model_params['n_estimators'],
            min_split_loss = model_params['min_split_loss'],
            tree_method = model_params['tree_method'],
            grow_policy = model_params['grow_policy'],
            single_precision_histogram = model_params['single_precision_histogram'],
            nthread = model_params['nthread']
        )


    def get_model(self):
        return self.model()