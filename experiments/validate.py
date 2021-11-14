import pandas as pd
from helper.utils import get_validation_params, get_config
from sklearn.model_selection import train_test_split
import pandas as pd

class Validate:
    data = None
    train_X = None
    train_y = None
    test_X = None
    val_X = None
    val_y = None
    test_ids = None
    validation_params = None
    config_params = None
    train_ids = None
    valid_ids = None

    def __init__(self, data, test_X, test_ids):
        self.data = data
        self.test_ids = test_ids
        self.test_X = test_X
        self.validation_params = get_validation_params()
        self.config_params = get_config()

    def prepare_validation_dataset(self):
        if (self.validation_params['validation_type'] == 'normal_split'):
            self.split_train_validation_set()
        else:
            pass

    def split_train_validation_set(self):
        y = self.data['type']
        data = self.data.drop(columns=['type'])
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(data, y, test_size = self.validation_params['validation_split_share'], random_state = 42)
        self.train_ids = self.train_X['Tweet_ID']
        self.valid_ids = self.val_X['Tweet_ID']
        self.train_X = self.train_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.val_X = self.val_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.test_X = self.test_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.train_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_X.csv', index = False, mode='w+')
        self.train_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_y.csv', index = False, mode='w+')
        self.val_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_X.csv', index = False, mode='w+')
        self.val_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_y.csv', index = False, mode='w+')
        self.test_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_X.csv', index = False, mode='w+')
        self.test_ids.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_ids.csv', index = False, mode='w+')

