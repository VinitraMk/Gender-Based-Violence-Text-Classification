from constants.model_enums import Model
from sklearn.svm import SVC
from models.base_singular_model import BaseModel

class SVM:

    def __init__(self, ensemble = False):
        BaseModel.__init__(self, Model.SVM, ensemble)
        self.model = SVC(
            kernel=self.params['kernel'],
            gamma=self.params['gamma'],
            tol=self.params['tol'],
            C=self.params['C'],
            decision_function_shape=self.params['multi_class'],
            class_weight = self.params['class_weight'],
            random_state = 42,
            max_iter = self.params['max_iter'],
            break_ties = self.params['break_ties']
        )
