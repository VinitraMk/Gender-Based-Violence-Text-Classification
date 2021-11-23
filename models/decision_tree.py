from constants.model_enums import Model
from sklearn.tree import DecisionTreeClassifier
from models.base_singular_model import BaseModel

class DecisionTree:

    def __init__(self, ensemble = False):
        BaseModel.__init__(self, Model.CTB, ensemble)
        self.model = DecisionTreeClassifier(
            random_state = 42,
            criterion = self.params['criterion'],
            max_depth = self.params['max_depth'],
            class_weight = self.params['class_weight']
        )
