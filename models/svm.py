from sklearn.svm import SVC
from helper.utils import get_model_params

class SVM:
    model = None

    def __init__(self):
        model_params = get_model_params()
        self.model = SVC(
            kernel=model_params['kernel'],
            gamma=model_params['gamma'],
            tol=model_params['tol'],
            C=model_params['C'],
            decision_function_shape=model_params['multi_class'],
            class_weight = model_params['class_weight'],
            random_state = 42,
            max_iter = model_params['max_iter'],
            break_ties = model_params['break_ties']
        )

    def get_model(self):
        return self.model