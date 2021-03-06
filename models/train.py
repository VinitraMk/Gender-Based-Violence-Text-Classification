import json
import os
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import datetime


def restore_label(label_dict, y_preds):
    for key in label_dict.keys():
        val = label_dict[key]
        y_preds = [key if x == val else x for x in y_preds]
    return y_preds

def get_model_path(model_path, model_name):
    return f'{model_path}/{model_name}_model.sav'

def create_output_directories(model_output_path, csv_output_path, log_output_path):
    if os.path.isdir(model_output_path) == False:
        os.mkdir(model_output_path)
    if os.path.isdir(csv_output_path) == False:
        os.mkdir(csv_output_path)
    if os.path.isdir(log_output_path) == False:
        os.mkdir(log_output_path)

def get_data(model_path, model_name, input_data_path, save_preds):
    model = joblib.load(get_model_path(model_path, model_name))
    X = pd.read_csv(f'{input_data_path}/train_X.csv')
    y = pd.read_csv(f'{input_data_path}/train_y.csv')
    valid_X = pd.read_csv(f'{input_data_path}/valid_X.csv')
    valid_y = pd.read_csv(f'{input_data_path}/valid_y.csv')
    test_X = None
    test_ids = None
    if save_preds:
        test_X = pd.read_csv(f'{input_data_path}/test_X.csv')
        test_ids = pd.read_csv(f'{input_data_path}/test_ids.csv')
    return model, X,y, valid_X, valid_y, test_X, test_ids

def train_model_and_predict(model, save_preds, model_output_path, model_name):
    model = model.fit(X, y)
    joblib.dump(model, get_model_path(model_output_path, model_name))
    if save_preds:
        ypreds = model.predict(test_X)
        ypreds = restore_label(label_dict, ypreds)
    else:
        ypreds = model.predict(valid_X)
    return ypreds

if __name__ == "__main__":
    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    save_preds = sys.argv[4] == 'True'
    epoch_index = int(sys.argv[5])
    k = int(sys.argv[6])
    model_filename = sys.argv[7]
    model_params = json.loads(sys.argv[8])
    preproc_params = json.loads(sys.argv[9]) if sys.argv[9] != '' else ''
    selected_features = sys.argv[10]
    label_dict = eval(sys.argv[11]) if sys.argv[11] != '' else dict()
    input_data_path = f'{mounted_input_path}/input'
    model_path = f'{mounted_input_path}/models'
    model_output_path = f'{mounted_output_path}/models'
    csv_output_path = f'{mounted_output_path}/results'
    log_output_path = f'{mounted_output_path}/experiment_logs'

    create_output_directories()

    '''
    model = joblib.load(f'{model_path}/{model_name}_model.sav')
    X = pd.read_csv(f'{input_data_path}/train_X.csv')
    y = pd.read_csv(f'{input_data_path}/train_y.csv')
    valid_X = pd.read_csv(f'{input_data_path}/valid_X.csv')
    valid_y = pd.read_csv(f'{input_data_path}/valid_y.csv')
    test_X = None
    test_ids = None
    '''
    model, X, y, valid_X, valid_y, test_X, test_ids = get_data(model_path, model_name, input_data_path, save_preds)
    y_preds = None
    avg_score = 0.0

    '''
    if save_preds:
        test_X = pd.read_csv(f'{input_data_path}/test_X.csv')
        test_ids = pd.read_csv(f'{input_data_path}/test_ids.csv')
    '''
    ypreds = train_model_and_predict(model, save_preds, model_output_path, model_name)
    '''
    model = model.fit(X, y)
    joblib.dump(model, f'{model_output_path}/{model_name}_model.sav')
    if save_preds:
        ypreds = model.predict(test_X)
        ypreds = restore_label(label_dict, ypreds)
    else:
        ypreds = model.predict(valid_X)
    '''

    log_path = f'{mounted_output_path}/{model_name}_log.txt'
    log_file_contents = dict()
    if not(save_preds):
        avg_score = accuracy_score(valid_y.values, ypreds)
    if os.path.isfile(log_path):
        with open(log_path) as log_file:
            log_file_contents = json.load(log_file)
        if not(save_preds):
            avg_score = avg_score + log_file_contents['model_output']['current_run_accuracy']
        else:
            avg_score = log_file_contents['model_output']['current_run_accuracy']
        if epoch_index == k-1 and k!=0:
            avg_score = avg_score / k
    log_file_contents['model_output'] = { 'current_run_accuracy': avg_score }
    if save_preds == True:
        log_path = f'{log_output_path}/{model_filename}.txt'
        log_file_contents['model_params'] = model_params
        log_file_contents['preproc_params'] = preproc_params
        log_file_contents['selected_features'] = selected_features
        preds_df = test_ids
        preds_df['type'] = ypreds
        preds_df.to_csv(f'{csv_output_path}/{model_filename}.csv', index = False)
    with open(log_path, 'w') as out_file:
        json.dump(log_file_contents, out_file)

