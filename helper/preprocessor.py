#python library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
import string
from sklearn import preprocessing
import re
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

#custom imports
from helper.utils import get_config, get_preproc_params, get_validation_params, save_fig
from helper.vectorizer import Vectorizer
from helper.augmenter import Augmenter, generate_new_data
from constants.types.label_encoding import LabelEncoding

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    preproc_args = None
    config = None
    class_dict = dict()
    train_len = 0
    features = None
    new_data = None
    new_data_ids = None
    best_words = None

    def __init__(self):
        self.config = get_config()
        self.train = pd.read_csv(f'{self.config["input_path"]}/Train.csv', encoding="ISO-8859-1")
        self.train_ids = self.train['Tweet_ID']
        self.train_len = self.train.shape[0]
        self.test = pd.read_csv(f'{self.config["input_path"]}/Test.csv', encoding="ISO-8859-1")
        self.test_ids = self.test['Tweet_ID']
        self.data = pd.read_csv(f'{self.config["input_path"]}/data.csv', encoding="ISO-8859-1")
        self.preproc_args = get_preproc_params()

    def start_preprocessing(self):
        print('\nStarting preprocessing of data...')
        self.__visualize_target_distribution()
        if self.preproc_args['apply_pseudo_labeling']:
            self.__generate_new_data()
        if self.preproc_args['encoding_type'] == LabelEncoding.LABEL_ENCODING:
            self.__apply_label_encoding()
        else:
            self.__apply_onehot_encoding()
        self.__cleanup_data()
        self.__test_train_split()
        self.__apply_vectorization()
        self.__make_new_features()
        self.__apply_scaling()
        return self.__apply_augmentation()

    def __cleanup_data(self):
        self.__apply_consistent_case()
        self.__remove_stop_words()
        self.__remove_punctuations()
        self.__remove_repeated_characters()
        self.__remove_urls()
        self.__remove_numbers()
        self.__remove_emoticons()
        self.__getting_tokens_of_text()
        if self.preproc_args['apply_stemming']:
            self.__apply_stemming()
        else:
            self.__apply_lematizer()
        #self.__plot_word_clouds()
        

    def __visualize_target_distribution(self, target_col = 'type'):
        print('\tDrawing target variable visualization')
        plt.figure(figsize=(12, 10))
        plt.hist(self.train[target_col])
        save_fig(f'{target_col}_plot', plt)
        plt.clf()

    def __apply_consistent_case(self):
        print('\tApplying consistent case to all text')
        apply_lowercase = lambda text: str(text).lower()
        self.data['tweet'] = self.data['tweet'].apply(apply_lowercase)

    def __remove_stop_words(self):
        print('\tRemoving stop words')
        en_stopwords = set(stopwords.words('english'))
        clean_stopwords = lambda text: " ".join([word for word in str(text).split() if word not in en_stopwords])
        self.data['tweet'] = self.data['tweet'].apply(clean_stopwords)

    def __remove_punctuations(self):
        print('\tRemoving punctuations')
        clean_punctuation = lambda text: text.translate(str.maketrans('', '', string.punctuation))
        self.data['tweet'] = self.data['tweet'].apply(clean_punctuation)

    def __remove_repeated_characters(self):
        print('\tRemoving repeated characters')
        remove_repetitions = lambda text: re.sub(r'(.)1+',r'1', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_repetitions)

    def __remove_urls(self):
        print('\tRemoving urls')
        remove_urls = lambda text: re.sub('((www.[^s]+)|(https?://[^s]+)|(http?//[^s]+))', ' ', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_urls)

    def __remove_numbers(self):
        print('\tRemoving numbers')
        remove_numbers = lambda text: re.sub('([0-9]+)', '', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_numbers)

    def __remove_emoticons(self):
        print('\tRemoving emoticons')
        emoticon_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags = re.UNICODE
        )
        remove_emoticons = lambda text: emoticon_pattern.sub(r'', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_emoticons)

    def __getting_tokens_of_text(self):
        print('\tTokenization of tweet text')
        tokenizer = RegexpTokenizer('\w+')
        self.data['tweet'] = self.data['tweet'].apply(tokenizer.tokenize)

    def __apply_stemming(self):
        print('\tApplying Porter Stemmer')
        stemmer = PorterStemmer()
        stemming = lambda text: " ".join([stemmer.stem(word) for word in text])
        self.data['tweet'] = self.data['tweet'].apply(stemming)

    def __apply_lematizer(self):
        print('\tApplying Wordnet Lemmatizer')
        lematizer = WordNetLemmatizer()
        lematizing = lambda text: " ".join([lematizer.lemmatize(word) for word in text])
        self.data['tweet'] = self.data['tweet'].apply(lematizing)

    def __plot_word_clouds(self):
        print('\tPlotting Word Clouds')
        for col_key in self.class_dict.keys():
            print(f'\t\tPlotting word cloud for {col_key}')
            col = self.class_dict[col_key]
            plt.figure(figsize=(20, 20))
            data = self.train
            data = data[data['type'] == col_key]['tweet'][:]
            wc = WordCloud(max_words = 1000, width = 1600, height = 800, collocations = False).generate(" ".join(data))
            plt.imshow(wc)
            save_fig(f'wordcloud_{col_key}', plt)
            plt.clf()

    def __test_train_split(self):
        print('\tRestore train and test split')
        self.train = self.data[self.data['Tweet_ID'].isin(self.train_ids)]
        self.test = self.data[self.data['Tweet_ID'].isin(self.test_ids)]
        if self.preproc_args['apply_pseudo_labeling']:
            self.new_data = self.data[self.data['Tweet_ID'].isin(self.new_data_ids)]
        self.test = self.test.drop(columns = ['type'])

    def __generate_new_data(self):
        config_params = get_config()
        generate_new_data(self.train['tweet'])
        self.new_data = pd.read_csv(f'{config_params["input_path"]}\\new_data.csv')
        self.new_data_ids = self.new_data['Tweet_ID']
        self.data = pd.concat([self.data, self.new_data])

    def __make_new_features(self):
        self.train['num_exclamation_marks'] = self.train['tweet'].apply(lambda x: x.count('!'))
        self.train['num_question_marks'] = self.train['tweet'].apply(lambda x: x.count('?'))
        self.train['num_capitals'] = self.train['tweet'].apply(lambda x: sum([1 for c in x if c.isupper()]))
        self.test['num_exclamation_marks'] = self.test['tweet'].apply(lambda x: x.count('!'))
        self.test['num_question_marks'] = self.test['tweet'].apply(lambda x: x.count('?'))
        self.test['num_capitals'] = self.test['tweet'].apply(lambda x: sum([1 for c in x if c.isupper()]))

        if self.preproc_args['apply_pseudo_labeling']:
            self.new_data['num_exclamation_marks'] = self.new_data['tweet'].apply(lambda x: x.count('!'))
            self.new_data['num_question_marks'] = self.new_data['tweet'].apply(lambda x: x.count('?'))
            self.new_data['num_capitals'] = self.new_data['tweet'].apply(lambda x: sum([1 for c in x if c.isupper()]))
        for word in self.best_words:
            col_name = f'is_{word}_present'
            self.train[col_name] = self.train['tweet'].apply(lambda x: 1 if x.count(word) > 0 else 0)
            self.test[col_name] = self.test['tweet'].apply(lambda x: 1 if x.count(word) > 0 else 0)
            if self.preproc_args['apply_pseudo_labeling']:
                self.new_data[col_name] = self.new_data['tweet'].apply(lambda x: 1 if x.count(word) > 0 else 0)
        print('\t\tNo of features: ', len(list(self.train.columns)) - 2)

    def __apply_scaling(self):
        train_df = self.train.drop(columns = ['tweet', 'Tweet_ID', 'type'])
        test_df = self.test.drop(columns = ['tweet', 'Tweet_ID'])
        train_features = train_df.values.tolist()
        test_features = test_df.values.tolist()
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        if self.preproc_args['apply_pseudo_labeling']:
            new_data_df = self.new_data.drop(columns=['tweet', 'Tweet_ID', 'type'])
            new_data_features = new_data_df.values.tolist()
            new_data_features = scaler.transform(new_data_features)
            self.__convert_vectors_dataframe(train_features, test_features, new_data_features, False)
        else:
            self.__convert_vectors_dataframe(train_features, test_features, None, False)

    def __convert_vectors_dataframe(self, train_features, test_features, new_data_features = None, merge = True):
        train_df = pd.DataFrame(train_features)
        test_df = pd.DataFrame(test_features)
        train_df['Tweet_ID'] = self.train_ids
        test_df['Tweet_ID'] = self.test_ids
        if merge:
            self.train = self.train.merge(train_df, on='Tweet_ID')
            self.test = self.test.merge(test_df, on='Tweet_ID')
        else:
            train_df['Tweet_ID'] = self.train['Tweet_ID']
            train_df['tweet'] = self.train['tweet']
            train_df['type']  = self.train['type']
            self.train = train_df
            test_df['Tweet_ID'] = self.test['Tweet_ID']
            test_df['tweet'] = self.test['tweet']
            self.test = test_df
        if self.preproc_args['apply_pseudo_labeling']:
            new_data_df = pd.DataFrame(new_data_features)
            new_data_df['Tweet_ID'] = self.new_data_ids
            if merge:
                self.new_data = self.new_data.merge(new_data_df, on = 'Tweet_ID')
            else:
                new_data_df['tweet'] = self.new_data['tweet']
                self.new_data = new_data_df

    def __apply_vectorization(self):
        print('\tApplying vectorization')
        if self.preproc_args['apply_pseudo_labeling']:
            vectorizer = Vectorizer(self.train['tweet'], self.train['type'], self.class_dict, self.test['tweet'], self.new_data['tweet'])
        else:
            vectorizer = Vectorizer(self.train['tweet'], self.train['type'], self.class_dict, self.test['tweet'])
        if self.preproc_args['apply_pseudo_labeling']:
            train_vectors, test_vectors, new_data_vectors, self.best_words = vectorizer.apply_vectorizer(self.preproc_args['apply_pseudo_labeling'])
            self.__convert_vectors_dataframe(train_vectors, test_vectors, new_data_vectors)
        else:
            train_vectors, test_vectors, self.best_words = vectorizer.apply_vectorizer()
            self.__convert_vectors_dataframe(train_vectors, test_vectors)
        
        self.features = vectorizer.get_features()
        print('\t\tBest words: ', self.best_words)

    def __apply_augmentation(self):
        print('\tApplying data augmentation')
        y_tmp = self.train['type']
        train_texts = self.train['tweet']
        train_tmp = self.train.drop(columns = ['Tweet_ID', 'tweet', 'type'])
        augmenter = Augmenter(train_tmp, y_tmp)
        train_tmp, y_tmp = augmenter.apply_data_augmentation()
        train_df = pd.DataFrame(train_tmp)
        train_y_df = pd.DataFrame(y_tmp, columns=['type'])
        train_df = pd.concat([train_df, train_y_df], axis = 1)
        train_df = pd.concat([train_df, self.__set_string_labels(train_df['type'])], axis = 1)
        self.train = pd.concat([train_df, self.train_ids, train_texts], axis = 1)
        self.__visualize_target_distribution('stype')
        self.train = self.train.drop(columns = ['stype'])
        if not(self.preproc_args['apply_pseudo_labeling']):
            return self.train, self.test, self.test_ids, self.features, self.class_dict
        return self.train, self.test, self.test_ids, self.features, self.class_dict, self.new_data

    def __apply_label_encoding(self):
        print('\tApply label encoding')
        le = preprocessing.LabelEncoder()
        self.data['type'] = le.fit_transform(self.data['type'])
        self.class_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        self.class_dict.pop(np.nan)
        print('\t\t',self.class_dict)

    def __set_string_labels(self, y):
        for key in self.class_dict.keys():
            val = self.class_dict[key]
            y = [key if x == val else x for x in y]
        return pd.DataFrame(y, columns = ['stype'])

    def __apply_onehot_encoding(self):
        pass
    