#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import nltk
import numpy
import re
import sklearn.metrics
import time

from nltk.stem import WordNetLemmatizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# import matplotlib.pylab
# import nltk
# from itertools import chain

def train():
    start = time.time()
    traindf = pandas.read_json('train.json')
    traindf['ingredients_clean_string'] = [' '.join(z).strip() for z in traindf['ingredients']]  
    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       
    traindf[['id' , 'cuisine' ]].to_csv('test_submission.csv', encoding='utf-8')

    testdf = pandas.read_json('test1.json')
    testdf['ingredients_clean_string'] = [' '.join(z).strip() for z in testdf['ingredients']]
    testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

    corpustr = traindf['ingredients_string']
    vectorizertr = TfidfVectorizer(
                                stop_words='english',
                                ngram_range=(1, 1),
                                analyzer='word',
                                max_df=0.57,
                                binary=False,
                                token_pattern=r'\w+',
                                sublinear_tf=False,
                                norm='l2')
    tfidftr=vectorizertr.fit_transform(corpustr)

    corpusts = testdf['ingredients_string']
    # vectorizerts = TfidfVectorizer(stop_words='english')
    tfidfts=vectorizertr.transform(corpusts)

    predictors_tr = tfidftr

    targets_tr = traindf['cuisine']

    predictors_ts = tfidfts

    # targets_ts = testdf['cuisine']

    # pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.2)

    # pred_train.shape
    # pred_test.shape
    # tar_train.shape
    # tar_test.shape
    # classifier=RandomForestClassifier(random_state=1, criterion = 'gini', n_estimators=75)
    classifier = LinearSVC(random_state=1, C=0.4, penalty='l2', dual=False)
    # classifier = LogisticRegression(random_state=1, C=7)
    classifier = classifier.fit(predictors_tr, targets_tr)
    print time.time() - start

    predictions = classifier.predict(predictors_ts)
    testdf['cuisine'] = predictions
    testdf = testdf.sort_values('id', ascending=True)
    # print testdf
    print time.time() - start

    testdf[['id' , 'cuisine' ]].to_csv('submission11.csv', encoding='utf-8', index=False)

    # resultdf = pandas.read_csv('sample_submission.csv')
    # resultdf = resultdf.sort_values('id', ascending=True)

    # sklearn.metrics.confusion_matrix(targets_ts,predictions)
    # print sklearn.metrics.accuracy_score(testdf['cuisine'], resultdf['cuisine'])
    scores = cross_validation.cross_val_score(classifier, predictors_tr,targets_tr, cv=3, scoring='accuracy')
    print scores
    print "Accuracy: %0.4f (+/- %0.5f)" % (scores.mean(), scores.std())
    print time.time() - start

    # sklearn.metrics.classification_report(tar_test, predictions)
train()