#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import nltk
import numpy
import re
import sklearn.metrics
import time
import statsmodels.api as sm
import matplotlib
import matplotlib.pylab as plt
import pylab
import scipy
import sys

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from nltk.stem import WordNetLemmatizer
from logit import LogisticRegressionCA
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

# import matplotlib.pylab
# import nltk
# from itertools import chain

def train():
    start = time.time()
    traindf = pandas.read_json('train.json')
    # print traindf, type(traindf)
    # exit()
    print 'open train.json: ', time.time() - start
    # make clean ingredients string
    traindf['ingredients_clean_string'] = [' '.join(z).strip() 
                                            for z in traindf['ingredients']]  
    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip()
                    for lists in traindf['ingredients']]       
    traindf[['id' , 'cuisine' ]].to_csv(
                                    'test_submission.csv',
                                    encoding='utf-8'
                                    )
    print 'clean train ingredients: ', time.time() - start

    testdf = pandas.read_json('test.json')
    testdf['ingredients_clean_string'] = [' '.join(z).strip() 
                                            for z in testdf['ingredients']]
    testdf['ingredients_string'] = [
        ' '.join([WordNetLemmatizer().lemmatize(re.sub(
            '[^A-Za-z]',
            ' ',
            line)) 
                for line in lists]).strip() 
                    for lists in testdf['ingredients']]       
    print 'clean test ingredients: ', time.time() - start

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
    print 'fit train vectorizer: ', time.time() - start

    corpusts = testdf['ingredients_string']
    # vectorizerts = TfidfVectorizer(stop_words='english')
    tfidfts=vectorizertr.transform(corpusts)

    predictors_tr = tfidftr

    targets_tr = traindf['cuisine']

    predictors_ts = tfidfts
    print 'fir test vectorizer: ', time.time() - start
    print predictors_ts, predictors_ts.shape

    # print tfidftr
    # print tfidftr[2, 357], type(tfidftr[2])
    # print tfidftr.shape[1]
    
###########################################################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # x, y, z = axes3d.get_test_data(0.05)
    # x_len = tfidftr.shape[0]
    # y_len = 500#tfidftr.shape[1]
    # x = numpy.zeros((y_len, y_len))
    # for i in range(y_len):
    #     x[i, :] = numpy.arange(y_len)
    # y = numpy.zeros((y_len, y_len))
    # for i in range(y_len):
    #     y[:, i] = numpy.arange(y_len)
    # z = numpy.array(tfidftr[:y_len, :y_len].todense())
 
    # print x, x.shape, type(x)
    # print y, y.shape, type(y)
    # print z, z.shape, type(z)
    # # ax.plot_wireframe(x, y, z)
    # a = []
    # b = []
    # zc = []
    # for i in range(y_len):
    #     for j in range(y_len):
    #         if z[i, j] > 0:
    #             a.append(i)
    #             b.append(j)
    #             zc.append(z[i, j])

    # # ax.scatter(x, y, z, depthshade=False)
    # # ax.scatter(a, b, c, depthshade=False)
    # colors = [
    #     'bo-', 'co-', 'go-', 'mo-', 'ko-', 'ro-', 'yo-', 'wo-',
    #     'b^-', 'c^-', 'g^-', 'm^-', 'k^-', 'r^-', 'y^-', 'w^-',
    #     'bs-', 'cs-', 'gs-', 'ms-', 'ks-', 'rs-', 'ys-', 'ws-'
    #     ]
    # aa = []
    # bb = []
    # zzc = []
    # for i in range(y_len):
    #     cuisine_unic = list(traindf['cuisine'].unique())
    #     cuisine = traindf.cuisine[i]
    #     # print cuisine_unic, cuisine
    #     index = cuisine_unic.index(cuisine)
    #     if index == 6:
    #         # ax.scatter(a[i], b[i], zc[i], c=colors[index][0], marker=colors[index][1])
    #         aa.append(a[i])
    #         bb.append(b[i])
    #         zzc.append(zc[i])
    # ax.plot_wireframe(aa, bb, zzc)
    # # plt.show()

########################################################
    
    # targets_ts = testdf['cuisine']

    # pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.2)

    # pred_train.shape
    # pred_test.shape
    # tar_train.shape
    # tar_test.shape
    # classifier=RandomForestClassifier(random_state=1, criterion = 'gini', n_estimators=75)
    # classifier = LinearSVC(
                        # random_state=1,
                        # C=0.4, penalty='l2',
                        # dual=False)

    unic_cuisine = list(traindf['cuisine'].unique())
    traindf['cuisine_n'] = [unic_cuisine.index(x) for x in traindf['cuisine']]
    targets_tr = traindf['cuisine_n']
    classifier = LogisticRegressionCA(x_train=predictors_tr[:25, :].todense(), 
                                      y_train=targets_tr[:25],
                                      x_test=predictors_ts[:25, :].todense(),
                                      alpha=0.01)
    classifier.train()

    print classifier.test_predictions()
    print classifier.training_reconstruction()
    print classifier.betas
    print targets_tr[:25]
    exit()
    # classifier = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
    #     eta0=0.0, fit_intercept=True, l1_ratio=0.15,
    #     learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
    #     penalty='l2', power_t=0.5, random_state=None, shuffle=True,
    #     verbose=0, warm_start=False)
    # classifier = LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None)
    # classifier = LogisticRegression(random_state=1, C=7)
    classifier = classifier.fit(predictors_tr, targets_tr)
    print 'train model: ', time.time() - start

    predictions = classifier.predict(predictors_ts)
    testdf['cuisine'] = predictions
    testdf = testdf.sort_values('id', ascending=True)
    # print testdf
    print 'make predictions: ', time.time() - start

    testdf[['id' , 'cuisine' ]].to_csv('submission11.csv',
                                        encoding='utf-8',
                                        index=False)

    
    # unic_cuisine = numpy.unique(traindf['cuisine'])
    # unic_cuisine = traindf['cuisine']
    # print unic_cuisine
    # unic_cuisine = list(unic_cuisine)
    # traindf['cuisine_n'] = [unic_cuisine.index(x) for x in traindf['cuisine']]
    # unic_cuisine = numpy.array(traindf['cuisine_n'])
    # print type(unic_cuisine), type(predictors_tr)
    # logit = sm.Logit(unic_cuisine, predictors_tr)
    # result = logit.fit()
    # print result.summary()

    # resultdf = pandas.read_csv('sample_submission.csv')
    # resultdf = resultdf.sort_values('id', ascending=True)

    # sklearn.metrics.confusion_matrix(targets_ts,predictions)
    # print sklearn.metrics.accuracy_score(testdf['cuisine'], resultdf['cuisine'])

    scores = cross_validation.cross_val_score(classifier,
                                              predictors_tr,
                                              targets_tr,
                                              cv=3,
                                              scoring='accuracy')
    print scores
    print "Accuracy: %0.4f (+/- %0.5f)" % (scores.mean(), scores.std())
    print time.time() - start

    # sklearn.metrics.classification_report(tar_test, predictions)
train()