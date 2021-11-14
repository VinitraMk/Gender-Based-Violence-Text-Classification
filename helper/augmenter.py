from constants.types.sampling_method import SamplingMethod
from helper.utils import get_preproc_params
from imblearn.over_sampling import SMOTE

class Augmenter:
    preproc_args = None
    train_X = None
    y = None
    augmenter = None

    def __init__(self, train_X, y):
        self.preproc_args = get_preproc_params()
        self.train_X = train_X
        self.y = y

    def apply_data_augmentation(self):
        if self.preproc_args['sampling_method'] == SamplingMethod.SMOTE_SAMPLING:
            return self.__apply_smote_sampling()

    def __apply_smote_sampling(self):
        self.augmenter = SMOTE(
            random_state = 42,
            k_neighbors = self.preproc_args['sampling_k'],
            sampling_strategy = self.preproc_args['sampling_strategy']
        )
        self.train_X, self.y = self.augmenter.fit_resample(self.train_X, self.y)
        return self.train_X, self.y
