from PIL.Image import new
import pandas as pd
from helper.utils import get_validation_params, get_config
from sklearn.model_selection import train_test_split
import pandas as pd
from helper.vectorizer import Vectorizer

class Validate:
    data = None
    new_data = None
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
    class_dict = None
    test_X_copy = None
    test_ids_copy = None

    def __init__(self, data, test_X, test_ids, class_dict, new_data):
        self.data = data
        self.test_ids = test_ids
        self.test_X = test_X
        self.test_X_copy = test_X
        self.test_ids_copy = test_ids
        self.class_dict = class_dict
        self.new_data = new_data
        self.validation_params = get_validation_params()
        self.config_params = get_config()

    def prepare_validation_dataset(self):
        if (self.validation_params['validation_type'] == 'normal_split'):
            self.__split_train_validation_set()
        else:
            pass

    def prepare_validation_data_in_runs(self, run_index):
        if run_index == 0:
            self.__prepare_firstrun_test_data()
        else:
            self.__prepare_secondrun_test_data()
    
    def __save_test_data(self):
        self.test_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_X.csv', index = False, mode='w+')
        self.test_ids.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_ids.csv', index = False, mode='w+')

    def __save_train_data(self):
        self.train_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_X.csv', index = False, mode='w+')
        self.train_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_y.csv', index = False, mode='w+')
        self.val_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_X.csv', index = False, mode='w+')
        self.val_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_y.csv', index = False, mode='w+')

    def __prepare_firstrun_test_data(self):
        y = self.data['type']
        data = self.data.drop(columns=['type'])
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(data, y, test_size = self.validation_params['validation_split_share'], random_state = 42)
        self.train_ids = self.train_X['Tweet_ID']
        self.valid_ids = self.val_X['Tweet_ID']
        self.train_X = self.train_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.val_X = self.val_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.test_X = self.new_data
        self.test_ids = self.new_data['Tweet_ID']
        self.test_X = self.test_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.__save_train_data()
        self.__save_test_data()
        
    def __prepare_secondrun_test_data(self):
        data_preds = pd.read_csv(f'{self.config_params["internal_output_path"]}\\new_data_preds.csv')
        self.new_data = self.new_data.merge(data_preds, on = 'Tweet_ID')
        self.train_y = pd.concat([self.train_y, self.new_data['type']])
        train_X = self.new_data.drop(columns = ['type', 'tweet', 'Tweet_ID'])
        self.train_X = pd.concat([self.train_X, train_X])
        self.test_X = self.test_X_copy
        self.test_ids = self.test_ids_copy
        self.test_X = self.test_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.__save_train_data()
        self.__save_test_data()

    def __split_train_validation_set(self):
        y = self.data['type']
        data = self.data.drop(columns=['type'])
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(data, y, test_size = self.validation_params['validation_split_share'], random_state = 42)
        self.train_ids = self.train_X['Tweet_ID']
        self.valid_ids = self.val_X['Tweet_ID']
        self.train_X = self.train_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.val_X = self.val_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.test_X = self.test_X.drop(columns = ['Tweet_ID', 'tweet'])
        self.__save_train_data()
        self.__save_test_data()

