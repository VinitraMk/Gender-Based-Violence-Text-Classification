from helper.utils import get_model_params
from sklearn.ensemble import VotingClassifier

class Ensembler:
    ensembler = None

    def __init__(self, models):
        params = get_model_params()
        self.ensembler = VotingClassifier(
            estimators = models,
            voting = params['voting_type'],
            n_jobs = -1
        )

    def get_model(self):
        return self.ensembler