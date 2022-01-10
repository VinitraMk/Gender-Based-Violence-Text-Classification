import sys
import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import joblib
import datetime
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

#### Model Definition ####

class ANN(nn.Module):

    def __init__(self):
        super(ANN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features = 320, out_features = 162),
            nn.Sigmoid(),
            nn.Linear(in_features = 162, out_features = 81),
            nn.Sigmoid(),
            nn.Linear(in_features = 81, out_features = 5)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = F.sigmoid(x)
        return x

def get_model_path(model_path, model_name):
    return f'{model_path}/{model_name}_model.pt'

def restore_label(label_dict, y_preds):
    for key in label_dict.keys():
        val = label_dict[key]
        y_preds = [key if x == val else x for x in y_preds]
    return y_preds

def copy_model_file(model_path):
    cd = os.getcwd()
    os.system(f'cp {model_path} {cd}')

def create_output_directories(model_output_path, csv_output_path, log_output_path):
    if os.path.isdir(model_output_path) == False:
        os.mkdir(model_output_path)
    if os.path.isdir(csv_output_path) == False:
        os.mkdir(csv_output_path)
    if os.path.isdir(log_output_path) == False:
        os.mkdir(log_output_path)

def fine_tune_model(model_path, model_name, lr):
    print('model path', get_model_path(model_path, model_name))
    print('is file exists', os.path.isfile(get_model_path(model_path, model_name)))
    #copy_model_file(get_model_path(model_path, model_name))
    #cd = os.getcwd()
    #new_path = f'{cd}/{model_name}_model.pt'
    #print('new path', new_path)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print('This is  not a cuda enabled machine')
        print('Aborting training')
        exit()
    model_object = torch.load(get_model_path(model_path, model_name))
    model = ANN()
    #model_state = model.state_dict()
    #model_state.update(model_object['model_state_dict'])
    model.load_state_dict(model_object['model_state_dict'])
    model.cuda()
    print('model params after', model)
    criterion = model_object['criterion']
    print('criterion object', criterion)
    #optimzer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    optimizer = model_object['optimizer']
    print('optimizer object', optimizer)
    return model, criterion, optimizer

def get_data(input_data_path, save_preds, pseudo_labeling_run):
    X = pd.read_csv(f'{input_data_path}/train_X.csv')
    y = pd.read_csv(f'{input_data_path}/train_y.csv')
    valid_X = pd.read_csv(f'{input_data_path}/valid_X.csv')
    valid_y = pd.read_csv(f'{input_data_path}/valid_y.csv')
    test_X = None
    test_ids = None
    if save_preds or (not(save_preds) and pseudo_labeling_run == 0):
        test_X = pd.read_csv(f'{input_data_path}/test_X.csv')
        test_ids = pd.read_csv(f'{input_data_path}/test_ids.csv')
    return X, y, valid_X, valid_y, test_X, test_ids

def get_target_labels(y):
    convert_to_class_arr = lambda x: np.array([1 if x==i else 0 for i in range(5)])
    y['type_converted'] = y['type'].apply(convert_to_class_arr)
    return y

def convert_input_to_tensor(X, y = None, start_index = -1, end_index = -1):
    if start_index != -1 and end_index != -1:
        X_values = torch.tensor(X.values[start_index:end_index])
        X_values = X_values.type(torch.FloatTensor).to(torch.device("cuda:0"))
    elif start_index != -1 and end_index == -1:
        X_values = torch.tensor(X.values[start_index:])
        X_values = X_values.type(torch.FloatTensor).to(torch.device("cuda:0"))
    else:
        X_values = torch.tensor(X.values)
        X_values = X_values.type(torch.FloatTensor).to(torch.device("cuda:0"))
    
    if y is not(None):
        if start_index != -1 and end_index != -1:
            y_values = torch.tensor(y.iloc[start_index:end_index].values).to(torch.device('cuda:0'))
        elif start_index != -1 and end_index == -1:
            y_values = torch.tensor(y.iloc[start_index:].values).to(torch.device("cuda:0"))
        else:
            y_values = torch.tensor(y.values).to(torch.device('cuda:0'))
        return X_values, y_values
    return X_values

def train_model(model, X, y, criterion, optimizer, num_epochs, batch_size):
    outputs = None
    print(X.shape)
    model.train()
    no_of_batches = int(X.shape[0] / batch_size) + 1
    running_loss = 0.0
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}...\n')
        loss_list = []
        for i, batch in enumerate(range(no_of_batches)):
            start_index = (i * batch_size)
            if ((start_index + batch_size) < X.shape[0]):
                end_index = start_index + batch_size
                X_values, y_values = convert_input_to_tensor(X, y, start_index, end_index)
            else:
                start_index = ((i-1) * batch_size) + batch_size
                X_values, y_values = convert_input_to_tensor(X, y, start_index, -1)
            optimizer.zero_grad()
            outputs = model(X_values)
            loss = criterion(outputs, y_values)
            loss.backward()
            optimizer.step()
            if not(torch.isnan(loss)):
                loss_list.append(loss.item())
            else:
                print(X_values)
                print(y_values)
                print(outputs)
                print(loss, torch.isnan(X_values))
                print(f'loss nan in batch {i}')
                exit()
        print(f'Running loss after epoch #{epoch}: {sum(loss_list) / no_of_batches}')
    return model, running_loss

def predict(model, save_preds, pseudo_labeling_run, valid_X, test_X, label_dict):
    ypreds = None
    model.eval()
    if save_preds:
        X_values = convert_input_to_tensor(test_X)
        _, ypreds = torch.max(model(X_values), 1)
        ypreds = ypreds.cpu().detach().numpy()
        ypreds = restore_label(label_dict, ypreds)
    elif pseudo_labeling_run == 0:
        X_values = convert_input_to_tensor(test_X)
        _, ypreds = torch.max(model(X_values), 1)
        ypreds = ypreds.cpu().detach().numpy()
    else:
        X_values = convert_input_to_tensor(valid_X)
        _, ypreds = torch.max(model(X_values), 1)
        ypreds = ypreds.cpu().detach().numpy()
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
    pseudo_labeling_run = int(sys.argv[12])

    lr = int(model_params['lr'])
    num_epochs = int(model_params['num_epochs'])
    batch_size = int(model_params['batch_size'])

    input_data_path = f'{mounted_input_path}/input'
    model_path = f'{mounted_input_path}/models'
    model_output_path = f'{mounted_output_path}/models'
    csv_output_path = f'{mounted_output_path}/results'
    log_output_path = f'{mounted_output_path}/experiment_logs'

    create_output_directories(model_output_path, csv_output_path, log_output_path)
    model, criterion, optimizer = fine_tune_model(model_path, model_name, lr)
    print('Starting script...')
    X, y, valid_X, valid_y, test_X, test_ids = get_data(input_data_path, save_preds, pseudo_labeling_run)
    avg_score = 0.0
    val_preds_confusion_matrix = None
    print('Starting training....\n')
    if not(save_preds):
        #converted_y = get_target_labels(y)
        model, running_loss = train_model(model, X, y['type'], criterion, optimizer, num_epochs, batch_size)
    #joblib.dump(model, get_model_path(model_output_path, model_name))
    torch.save(model.state_dict(), get_model_path(model_output_path, model_name))
    ypreds = predict(model, save_preds, pseudo_labeling_run, valid_X, test_X, label_dict)

    print('Finished training and predict')
    print(save_preds, pseudo_labeling_run)
    
    log_path = f'{mounted_output_path}/{model_name}_log.txt'
    log_file_contents = dict()
    if not(save_preds) and pseudo_labeling_run != 0:
        avg_score = accuracy_score(valid_y.values, ypreds)
        #print('not save preds', avg_score)
        val_preds_confusion_matrix = str(confusion_matrix(valid_y.values, ypreds))
    if os.path.isfile(log_path):
        with open(log_path) as log_file:
            log_file_contents = json.load(log_file)
        if not(save_preds):
            avg_score = avg_score + log_file_contents['model_output']['current_run_accuracy']
            #print('log path file, save_preds = false', avg_score)
        else:
            avg_score = log_file_contents['model_output']['current_run_accuracy']
            val_preds_confusion_matrix = log_file_contents['model_output']['confusion_matrix_final']
            #print('save_preds = True', avg_score)

        if epoch_index == k-1 and k!=0:
            #print('avg score before mean', avg_score)
            avg_score = avg_score / k
            #print('avg score after mean', avg_score)
    log_file_contents['model_output'] = { 'current_run_accuracy': avg_score, 'confusion_matrix_final': val_preds_confusion_matrix }
    preds_df = test_ids
    if save_preds == True:
        log_path = f'{log_output_path}/{model_filename}.txt'
        log_file_contents['model_params'] = model_params
        log_file_contents['preproc_params'] = preproc_params
        log_file_contents['selected_features'] = selected_features
        preds_df['type'] = ypreds
        preds_df.to_csv(f'{csv_output_path}/{model_filename}.csv', index = False, mode = 'w+')
    elif pseudo_labeling_run == 0:
        preds_df['type'] = ypreds
        preds_df.to_csv(f'{csv_output_path}/new_data_preds.csv', index = False, mode = 'w+')
    with open(log_path, 'w') as out_file:
        print('Saving log file...')
        json.dump(log_file_contents, out_file)

