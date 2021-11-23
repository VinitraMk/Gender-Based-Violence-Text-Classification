from helper.utils import get_model_params

class BaseModel:
    params = None
    model = None

    def __init__(self, model_name, ensemble):
        self.params = get_model_params(ensemble, model_name)

    def get_model(self):
        return self.model