from constants.types.vectorizer_types import VectorizerTypes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from helper.utils import get_preproc_params 
from constants.types.vectorizer_types import VectorizerTypes
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.preprocessing import StandardScaler

class Vectorizer:
    vectorizer = None
    preproc_args = None
    vectorizer_method = ''
    train_X = None
    test_X = None
    all_labels = None
    train_y = None
    
    def __init__(self, train_X, train_y, test_X, all_labels):
        self.preproc_args = get_preproc_params()
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.all_labels = all_labels

    def apply_vectorizer(self):
        if self.preproc_args['vectorization_method'] == VectorizerTypes.TFIDF:
            return self.__apply_tfidf_vectorization()
        else:
            return self.__apply_count_vectorization()

    def get_features(self):
        return self.vectorizer.get_feature_names()

    def __invoke_vectorizer(self, vocabulary = None):
        if self.preproc_args['vectorization_method'] == VectorizerTypes.TFIDF:
            self.vectorizer = TfidfVectorizer(
                sublinear_tf = self.preproc_args['sublinear_tf'],
                norm = self.preproc_args['norm'],
                vocabulary = vocabulary,
                ngram_range = (self.preproc_args['ngram_range_min'], self.preproc_args['ngram_range_max'])
            )
        elif self.preproc_args['vectorization_method'] == VectorizerTypes.WC:
            self.vectorizer = CountVectorizer(
                ngram_range = (self.preproc_args['ngram_range_min'], self.preproc_args['ngram_range_max']),
                vocabulary = vocabulary,
                max_features = self.preproc_args['vectorizer_max_features']
            )

    def __apply_scaling(self, train_features, test_features):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.fit_transform(test_features)
        return train_features, test_features

    def __get_best_features(self, all_features):
        reduced_vocabulary = []
        for label, label_id in sorted(self.all_labels.items()):
            train_features_chi2 = chi2(all_features, self.train_y == label_id)
            indices = np.argsort(train_features_chi2[0])
            feature_names = np.array(self.get_features())[indices]

            n_grams = []
            for i in range(self.preproc_args['ngram_range_min'], self.preproc_args['ngram_range_max'] + 1):
                n_gram = [v for v in feature_names if len(v.split(' ')) == i]
                n_grams.append(n_gram)

            for n_gram in n_grams:
                reduced_vocabulary = reduced_vocabulary + n_gram[-self.preproc_args['best_k_features']:]

        reduced_vocabulary = list(set(reduced_vocabulary))
        self.__invoke_vectorizer(reduced_vocabulary)
        train_features = self.vectorizer.fit_transform(self.train_X).toarray()
        test_features = self.vectorizer.transform(self.test_X).toarray()
        train_features, test_features = self.__apply_scaling(train_features, test_features)
        return train_features, test_features

    def __apply_tfidf_vectorization(self):
        self.__invoke_vectorizer()
        train_features = self.vectorizer.fit_transform(self.train_X).toarray()
        train_features, test_features = self.__get_best_features(train_features)
        return train_features, test_features

    def __apply_count_vectorization(self):
        self.__invoke_vectorizer()
        train_features = self.vectorizer.fit_transform(self.train_X).toarray()
        train_features, test_feautures = self.__get_best_features(train_features)
        return train_features, test_feautures
