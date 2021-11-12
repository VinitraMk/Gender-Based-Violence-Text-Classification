from sklearn.svm import LinearSVC
from helper.utils import get_model_params

class SVM:
    model = None

    def __init__(self):
        model_params = get_model_params()
        self.model = LinearSVC(
            penalty=model_params['penalty'],
            loss=model_params['loss'],
            tol=model_params['tol'],
            C=model_params['C'],
            multi_class=model_params['multi_class'],
            fit_intercept = model_params['fit_intercept'],
            class_weight = model_params['class_weight'],
            random_state = 42,
            max_iter = model_params['max_iter']
        )

    def get_model(self):
        return self.model