''' Module realise supervised learning model
    for prediction national cuisine by recipe ingredients.
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import re

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

class SourceOrganizer(object):
    ''' Class prepare input train and test data for analisys. '''
    def __init__(self):
        # folder name with input files
        self.folder = 'app/static/data/input/'
        # trainig file for models
        self.train_data_file = 'train.json'
        # test file for models
        self.test_data_file = 'test.json'
        # vocabulary for vectorizer
        self.vocabulary = None

    def get(self, source='train'):
        ''' Take data from source file and
            create object instanse with data
            from file.

        Parameters
        ----------
        source : string
            can be 'train' or 'test' and show what file
            take for pretaring.

        Results
        -------
        train_data : Pandas DataFrame, [m rows x n columns]
            example for train mode:
            -----------------------
            index           cuisine     id                 ingredients
                0             greek  10259  [romaine lettuce, black...
                1       southern_us  25693  [plain flour, ground pe...
                2          filipino  20130  [eggs, pepper, salt, ma...
        test_data : Pandas DataFrame, [m rows x n columns]    
            example for test mode:
            ----------------------
            index      id                                  ingredients
                0   10259  [romaine lettuce, black olives, grape to...
                1   25693  [plain flour, ground pepper, salt, tomat...
                2   20130  [eggs, pepper, salt, mayonaise, cooking ...
        '''
        data_file_name = self.folder + source + '.json'
        if source == 'train':
            self.train_data = pd.read_json(data_file_name)
            # self._vocabulary()
        self.test_data = pd.read_json(data_file_name)

    def lemmatize_ingredients(self, source='train'):
        ''' Lemmatize ingredients in data object string using
            WordNet's built-in morphy function.
            Input string word will be unchanged 
            if it cannot be found in WordNet.
        Parameters
        ----------
        data : Pandas DataFrame, [m rows x n columns]
            train or test data
            example in 'get' method description.

        source : string
            can be 'train' or 'test' and show what file
            take for pretaring.

        Returns
        -------
        self.data : Pandas DataFrame, [m rows x n columns]
            Update data object with column 'ingredients_string'
            with lemmitized strings.
        '''
        ing = 'ingredients'
        ing_ls = 'ingredients_string'
        # transform ingredients words
        # with lemmatize function and add
        # ingredients_string column to data object
        wnl = WordNetLemmatizer()
        # clean word from spaces and choose only chars
        clean = lambda x: re.sub('[^A-Za-z]', ' ', x)
        # lemmatize each word
        lemma = lambda x: wnl.lemmatize(x)
        # add limmatized ingredients column to data object
        # with name 'ingredients_string'
        if source == 'train':
            self.train_data[ing_ls] = [
                ' '.join([lemma(clean(word)) for word in line]).strip()
                                        for line in self.train_data[ing]
                ]
            # self._vocabulary()
        self.test_data[ing_ls] = [
            ' '.join([lemma(clean(word)) for word in line]).strip()
                                    for line in self.test_data[ing]
            ]


    def vectorize_ingredients(self, data_train, data_test=None):
        ''' Use TfidfVectorizer to learn the vocabulary
            dictionary(in our case ingredients) and
            return term-document matrix.

        Parameters
        ----------
        data : Pandas DataFrame, [m rows x n columns]
            Train or test data.
            Example in 'get' method description.

        Returns
        -------
        matrix : array, [n_samples, n_features]
            Document-term matrix.
        '''
        corpus_train = data_train['ingredients_string']
        try:
            corpus_test = data_test['ingredients_string']
        except:
            pass
        vectorizer = TfidfVectorizer(
                                stop_words='english',
                                ngram_range=(1, 1),
                                analyzer='word',
                                max_df=0.57,
                                binary=False,
                                token_pattern=r'\w+',
                                sublinear_tf=False,
                                norm='l2')#,
                                # vocabulary=self.vocabulary)
        
        matrix_train = vectorizer.fit_transform(corpus_train)
        try:
            matrix_test = vectorizer.transform(corpus_test)
        except:
            return matrix_train

        return matrix_train, matrix_test

    def _vocabulary(self):
        ''' Take ingredients data from 'self.train_data'
            and create vocabulary from lemmitized words.

        Results
        -------
        self.vocabulary: set
            Update object attribute 'self.vocabulary
        '''
        vocabulary = []
        wnl = WordNetLemmatizer()
        vocabulary = [
            vocabulary.append(wnl.lemmatize(ingredient.lower())) 
                for ingredients in self.train_data['ingredients']
                    for ingredient in ingredients
                        if ingredient not in vocabulary
                    ]
        self.vocabulary = set(vocabulary)


class Classifiers(object):
    ''' Class for using few classifiers to
        train and make predictions.
    '''
    def __init__(self):
        # list with classifiers
        self.clf = []
        # list with predictions
        self.pred = []
        # models names
        self.models = [
            'RandomForestClassifier',
            'LinearSVC',
            'SGDClassifier',
            'LogisticRegression',
            'LogisticRegressionCV'
            ]

    def set_classifiers(self):
        ''' Define classifier object and add them
            to the list of classifiers.

        Results
        -------
        self.clf : list with object
            Update classifier object in self.clf list
        '''
        classifier1 = RandomForestClassifier(
            random_state=1, criterion = 'gini', n_estimators=75
            )
        self.classifier2 = LinearSVC(
            random_state=1, C=0.4, penalty='l2', dual=False
            )
        classifier2 = LinearSVC(
            random_state=1, C=0.4, penalty='l2', dual=False
            )

        self.classifier3 = SGDClassifier(
            alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=0, warm_start=False
            )
        classifier3 = SGDClassifier(
            alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=0, warm_start=False
            )
        classifier4 = LogisticRegression(
            random_state=1, C=7
            )
        classifier5 = LogisticRegressionCV(
            Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2',
            scoring=None, solver='lbfgs', tol=0.0001, max_iter=100,
            class_weight=None, n_jobs=1, verbose=0, refit=True,
            intercept_scaling=1.0, multi_class='ovr', random_state=None
            )
        # add number to classifier
        add = lambda x: 'classifier%s' % str(x)
        # add classifier to list
        [self.clf.append(eval(add(i))) for i in range(1,6)]

    def fit_classifiers(self, features_matrix, samples):
        ''' Fit the model according to the given training data.
        
        Parameters
        ----------
        features_matrix : {array-like, sparse matrix}, 
                         shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        
        samples : array-like, shape (n_samples,)
            Target vector relative to X.
        
        Results
        -------
        self.clf : list with object
            Update classifier object in self.clf list
        '''
        for classifier in self.clf:
            classifier = classifier.fit(features_matrix, samples)

    def predictions(self, features_matrix):
        ''' Make predictions according to the given test data.
        
        Parameters
        ----------
        features_matrix : {array-like, sparse matrix}, 
                         shape (n_samples, n_features)
            Test vector, where n_samples in the number of samples and
            n_features is the number of features.
        
        Results
        -------
        self.pred : array-like, shape (n_samples,)
            Add predictions made by classifier to self.pred list.
            Prediction is target vector relative to X.        
        '''
        for classifier in self.clf:
            prediction = classifier.predict(features_matrix)
            self.pred.append(prediction)

def main_train():
    ''' Generate main train data object.
        Train models on this data.

    Returns
    -------
    data : Pandas DataFrame, [m rows x n columns]
        Train or test data.
        Example in 'get' method description.

    clf : list
        List of classifiers objects in 'clf'
    '''
    data = SourceOrganizer()
    data.get(source='train')
    data.lemmatize_ingredients(source='train')
    matrix_train = data.vectorize_ingredients(data.train_data)
    targets_train = data.train_data['cuisine']
    clf = Classifiers()
    clf.set_classifiers()
    clf.fit_classifiers(matrix_train, targets_train)
    return data, clf

def make_json(string):
    ''' Make json-like string for input in vectorizer method.
        
    Parameters
    ----------
    string : string
        String with words separated by 
        comma and space ', '.

    Returns
    -------
    result : string
        example:
        [
          {
            "id": 10259,
            "ingredients": [
              "romaine lettuce",
              "black olives",
              "feta cheese crumbles"
            ]
          }
        ]
    '''
    string = string.split(', ')
    start = '[\n  {\n    "id": '
    id_n = str(random.randint(1, 2000)) + ',\n'
    ingredient = '    "ingredients": [\n'
    end = '    ]\n  }\n]'
    ingredients = ''
    for i, x in enumerate(string):
        if i == (len(string) - 1):
            ingredients += ('      "%s"' % x) + '\n'
        else:
            ingredients += ('      "%s"' % x) + ',\n'
    result = start + id_n + ingredient + ingredients + end
    return result

if __name__ == '__main__':
    # bar = 'green chile, jalapeno chilies, onions, ground black pepper, \
    #   salt, chopped cilantro fresh, green bell pepper, \
    #   garlic, white sugar, roma tomatoes, celery, dried oregano'
    # bar1 = 'baking powder, eggs, all-purpose flour, raisins, milk, white sugar'
    data, clf = main_train()
    while True:
        bar1 = raw_input('>')
        foo = make_json(bar1)
        # clf = Classifiers()
        # clf.set_classifiers()
        # data = SourceOrganizer()
        # data.get(source='train')
        # data.get(source='test2')
        data.test_data = pd.read_json(foo)
        print data.test_data
    ##    data.lemmatize_ingredients(source='train')

        # data.train_data['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip()
        #                 for lists in data.train_data['ingredients']]

        # vectorizer = TfidfVectorizer(
        #                         stop_words='english',
        #                         ngram_range=(1, 1),
        #                         analyzer='word',
        #                         max_df=0.57,
        #                         binary=False,
        #                         token_pattern=r'\w+',
        #                         sublinear_tf=False,
        #                         norm='l2')#,
        #                         # vocabulary=self.vocabulary)
        # matrix_train = vectorizer.fit_transform(data.train_data['ingredients_string'])

    ##    matrix_train = data.vectorize_ingredients(data.train_data)
    ##    targets_train = data.train_data['cuisine']
        data.lemmatize_ingredients(source='test')
        # data.test_data['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip()
        #                 for lists in data.test_data['ingredients']]


        matrix_test = data.vectorize_ingredients(data.train_data, data.test_data)[1]

        # matrix_test = vectorizer.transform(data.test_data['ingredients_string'])

        # classifier2 = LinearSVC(
        #     random_state=1, C=0.4, penalty='l2', dual=False
        #     )
        # classifier3 = SGDClassifier(
        #     alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
        #     eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        #     learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
        #     penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        #     verbose=0, warm_start=False
        #     )

        # classifier2 = classifier2.fit(matrix_train, targets_train)
        # classifier3 = classifier3.fit(matrix_train, targets_train)

    ##    clf.fit_classifiers(matrix_train, targets_train)
        # print clf.clf
        # print clf.clf
        clf.predictions(matrix_test)
        # pred2 = classifier2.predict(matrix_test)
        # pred3 = classifier3.predict(matrix_test)
        # print pred2
        # print pred3
        for x in clf.pred:
            print x
        # print clf.pred
        print 'Done'