#python library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
import string
from sklearn import preprocessing

#custom imports
from helper.utils import get_config, get_preproc_params, save_fig

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    preproc_args = None
    config = None
    y = None

    def __init__(self):
        self.config = get_config()
        self.train = pd.read_csv(f'{self.config["input_path"]}/Train.csv')
        self.train_ids = self.train['Tweet_ID']
        self.y = self.train['type']
        self.test = pd.read_csv(f'{self.config["input_path"]}/Test.csv')
        self.test_ids = self.test['Tweet_ID']
        self.test.drop(columns=['Tweet_ID'])
        self.data = pd.read_csv(f'{self.config["input_path"]}/data.csv')
        self.data.drop(columns=['type'])
        self.preproc_args = get_preproc_params()

    def start_preprocessing(self):
        print('\nStarting preprocessing of data...')
        self.visualize_target_distribution()
        self.remove_stop_words()
        self.remove_punctuations()
        self.test_train_split()
        if self.preproc_args['encoding_type'] == 'label_encoding':
            return self.apply_label_encoding()
        else:
            return self.apply_onehot_encoding()
        

    def visualize_target_distribution(self):
        print('Drawing target variable visualization')
        plt.figure(figsize=(12, 10))
        plt.hist(self.train['type'])
        save_fig('Target_plot', plt)
        plt.clf()

    def remove_stop_words(self):
        print('Removing stop words')
        en_stopwords = set(stopwords.words('english'))
        clean_stopwords = lambda text: " ".join([word for word in str(text).split() if word not in en_stopwords])
        self.data['tweet'] = self.data['tweet'].apply(clean_stopwords)

    def remove_punctuations(self):
        print('Removing punctuations')
        clean_punctuation = lambda text: text.translate(str.maketrans('', '', string.punctuation))
        self.data['tweet'] = self.data['tweet'].apply(clean_punctuation)

    def test_train_split(self):
        print('Restore train and test split')
        self.train = self.data[self.data['Tweet_ID'].isin(self.train_ids)]
        self.test = self.data[self.data['Tweet_ID'].isin(self.test_ids)]

    def apply_label_encoding(self):
        print('Apply label encoding\n')
        le = preprocessing.LabelEncoder()
        self.train['type'] = le.fit_transform(self.y)
        return self.train, self.test, self.test_ids

    def apply_onehot_encoding(self):
        pass
    