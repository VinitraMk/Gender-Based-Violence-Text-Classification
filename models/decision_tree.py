from constants.model_enums import Model
from sklearn.tree import DecisionTreeClassifier
from helper.utils import get_model_params

class DecisionTree:
    model = None

    def __init__(self, ensemble = False):
        model_params = get_model_params(ensemble, Model.DECISION_TREE)
        self.model = DecisionTreeClassifier(
            random_state = 42,
            criterion = model_params['criterion'],
            max_depth = model_params['max_depth'],
            class_weight = model_params['class_weight']
        )

    def get_model(self):
        return self.model