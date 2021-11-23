from constants.types.ensemble_method import EnsembleMethod
from helper.utils import get_model_params
from sklearn.ensemble import VotingClassifier, StackingClassifier

class Ensembler:
    ensembler = None

    def __init__(self, models, final_estimator = None):
        params = get_model_params()
        ensembler_type = params['ensembler_type']
        if ensembler_type == EnsembleMethod.VOTING:
            self.__build_voting_classifier(params, models)
        elif ensembler_type == EnsembleMethod.STACKING:
            self.__build_stacking_classifier(params, models, final_estimator)
        else:
            raise ValueError(f'{ensembler_type} is an invalid ensembler type')

    def __build_voting_classifier(self, params, models):
        self.ensembler = VotingClassifier(
            estimators = models,
            voting = params['voting_type'],
            n_jobs = -1
        )

    def __build_stacking_classifier(self, params, models, final_estimator):
        self.ensembler = StackingClassifier(
            estimators = models,
            final_estimator = final_estimator,
            n_jobs = -1,
            passthrough = params['passthrough'],
            verbose = 0,
            stack_method = params['stack_method'],
            cv = params['cv']
        )
    

    def get_model(self):
        return self.ensembler