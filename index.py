#python imports
import json
import os
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from azure.storage.blob import BlobServiceClient
from datetime import date
import pandas as pd

#custom imports
from helper.utils import get_config, get_filename, get_model_params, get_preproc_params, get_validation_params, save_model 
from helper.preprocessor import Preprocessor
from models.svm import SVM
from models.xgb import XGB
from models.decision_tree import DecisionTree
from models.ctb import CTB
from models.random_forest import RFA
from models.ensembler import Ensembler
from experiments.validate import Validate
from constants.model_enums import Model
#from experiments.experiment import Experiment


def get_models_for_ensembler(model_args):
    all_models = model_args['ensembler_args'].keys()
    models = []
    for model in all_models:
        model_arg = model_args['ensembler_args'][model]
        model_clf = make_model(model_arg, True)
        models.append((model, model_clf))
    return models

def make_model(args, ensemble = False):
    if args['model'] == Model.SVM:
        svc_classifier= SVM(ensemble)
        return svc_classifier.get_model()
    elif args['model'] == Model.XGB:
        xgb_classifier = XGB(ensemble)
        return xgb_classifier.get_model()
    elif args['model'] == Model.DECISION_TREE:
        decision_tree_clf = DecisionTree(ensemble)
        return decision_tree_clf.get_model()
    elif args['model'] == Model.CTB:
        catboost_classifier = CTB(ensemble)
        return catboost_classifier.get_model()
    elif args['model'] == Model.RFA:
        rfa = RFA(ensemble)
        return rfa.get_model()
    elif args['model'] == Model.ENSEMBLER:
        models = get_models_for_ensembler(args)
        ensembler_classifer = Ensembler(models)
        return ensembler_classifer.get_model()
    else:
        print('Invalid model name :-( \n')
        exit()

def make_azure_res():
    print('\nConfiguring Azure Resources...')
    # Configuring workspace
    print('\tConfiguring Workspace...')
    today = date.today()
    todaystring = today.strftime("%d-%m-%Y")
    ws = Workspace.from_config()

    print('\tConfiguring Environment...\n')
    user_env = Environment.get(workspace=ws, name="vinazureml-env")
    experiment = AzExperiment(workspace=ws, name=f'{todaystring}-experiments')
    
    return experiment, ws, user_env
    

def train_model_in_azure(azexp, azws, azuserenv, model_name, epoch_index, validation_k, model_args_string, preproc_string = '', save_preds = False, filename = '', features = '', label_dict = '', pseudo_labeling_run = -1):
    def_blob_store = azws.get_default_datastore()
    def_blob_store.upload(src_dir='./processed_io', target_path='input/', overwrite=True)
    input_data = Dataset.File.from_files(path=(def_blob_store,'/input'))
    input_data = input_data.as_named_input('input').as_mount()
    output = OutputFileDatasetConfig(destination=(def_blob_store, '/output'))

    config = ScriptRunConfig(
        source_directory='./models',
        script='train.py',
        arguments=[input_data, output, model_name, save_preds, epoch_index, validation_k, filename, model_args_string, preproc_string, features, label_dict, pseudo_labeling_run],
        compute_target='mikasa',
        environment=azuserenv)
    run = azexp.submit(config)
    run.wait_for_completion(show_output=True)
    aml_url = run.get_portal_url()
    print(aml_url)

def preprocess_data():
    preprocessor = Preprocessor()
    return preprocessor.start_preprocessing()

def download_blob(local_filename, blob_client_instance):
    with open(local_filename, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)

def download_output(filename, model_name, pseudo_labeling_run = -1):
    config = get_config()
    STORAGEACCOUNTURL= 'https://mlintro1651836008.blob.core.windows.net/'
    LOCALCSVPATH = f'{config["output_path"]}\\{filename}.csv' if pseudo_labeling_run == -1 else f'{config["internal_output_path"]}\\new_data_preds.csv'
    LOCALLOGPATH = f'{config["experimental_output_path"]}\\{filename}.txt'
    MAINCONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
    CSVCONTAINER = f'{MAINCONTAINER}/results'
    LOGCONTAINER = f'{MAINCONTAINER}/experiment_logs'
    CSVBLOB = f'{filename}.csv' if pseudo_labeling_run == -1 else 'new_data_preds.csv'
    LOGBLOB = f'{filename}.txt'
    EXPLOGBLOB = f'{model_name}_log.txt'
    blob_service_client = BlobServiceClient(account_url= STORAGEACCOUNTURL, credential = os.environ['AZURE_STORAGE_CONNECTIONSTRING'])
    blob_client_csv = blob_service_client.get_blob_client(CSVCONTAINER, CSVBLOB, snapshot = None)
    if pseudo_labeling_run == -1:
        blob_client_log = blob_service_client.get_blob_client(LOGCONTAINER, LOGBLOB, snapshot = None)
        blob_client_explog = blob_service_client.get_blob_client(MAINCONTAINER, EXPLOGBLOB, snapshot = None)
        download_blob(LOCALLOGPATH, blob_client_log)
        blob_client_explog.delete_blob()
    download_blob(LOCALCSVPATH, blob_client_csv)
    

def test_model(model):
    config_params = get_config()
    X = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\train_X.csv')
    y = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\train_y.csv')
    test_X = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\test_X.csv')
    test_ids = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\test_ids.csv')
    print(X.shape, y.shape)
    print(X.head())
    model = model.fit(X, y)
    print('Test X shape' ,test_X.shape)
    print(test_X.head())
    ypreds = model.predict(test_X)
    print(ypreds[:5])
    ypreds = pd.DataFrame(ypreds)
    preds_df = test_ids.join(ypreds)
    print(preds_df.head())

def start_validation(data, test_ids, test_X, label_dict, new_data = None, features = [], model = None):
    args = get_model_params()
    validation_args = get_validation_params()
    preproc_args = get_preproc_params()
    azexp, azws, azuserenv = make_azure_res()
    print('Starting experiment...')
    validate = Validate(data, test_X, test_ids, label_dict, new_data)
    model_args_string = json.dumps(args)
    preproc_args_string = json.dumps(get_preproc_params())
    if validation_args['validation_type'] == 'normal_split':
        print('\n\n******* Validation Run ***********')
        filename = get_filename(args['model'])
        if not(preproc_args['apply_pseudo_labeling']):
            validate.prepare_validation_dataset()
            #test_model(model)
            train_model_in_azure(azexp, azws, azuserenv, args['model'], 0, 1, model_args_string, preproc_args_string, False, filename, '')
            train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , 0, model_args_string, preproc_args_string, True, filename, '', str(label_dict))
            download_output(filename, args['model'])
        else:
            validate.prepare_validation_data_in_runs(0)
            test_model(model)
            #train_model_in_azure(azexp, azws, azuserenv, args['model'], -1, 0, model_args_string, preproc_args_string, False, filename, '', '', 0)
            #download_output(filename, args['model'], 1)
            #validate.prepare_validation_data_in_runs(1)
            #train_model_in_azure(azexp, azws, azuserenv, args['model'], 0, 1, model_args_string, preproc_args_string, False, filename, '')
            #train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , 0, model_args_string, preproc_args_string, True, filename, '', str(label_dict))
            #download_output(filename, args['model'])
    else:
        for i in range(validation_args['k']):
            print('\n*************** Run', i,'****************')
            validate.prepare_validation_dataset()
            train_model_in_azure(azexp, azws, azuserenv, args['model'], i , validation_args['k'], model_args_string)
        print('\n\n*************** Final Run ****************')
        filename = get_filename(args['model'])
        validate.prepare_full_dataset()
        train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , validation_args['k'], model_args_string, True, filename)
        download_output(filename, args['model'])

def read_args():
    args = get_model_params()
    config = get_config()
    preproc_args = get_preproc_params()
    if not(preproc_args['apply_pseudo_labeling']):
        train, test_X, test_ids, features, label_dict = preprocess_data()
    else:
        train, test_X, test_ids, features, label_dict, new_data = preprocess_data()
    model = make_model(args)
    model_path = f'{config["processed_io_path"]}/models'
    save_model(model, model_path, args['model'])
    start_validation(train, test_ids, test_X, label_dict, new_data, features, model)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()