import numpy as np
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from scipy.optimize.optimize import fmin_bfgs
from sklearn.feature_extraction.text import TfidfVectorizer

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LogisticRegressionCA():
    ''' Logistic regression model with L2 regularization (zero-mean
        Gaussian priors on parameters). '''

    def __init__(self, x_train=None, y_train=None,
                       x_test=None, y_test=None,
                       alpha=.1):

        # Set L2 regularization strength
        self.alpha = alpha
         
        # Set the data.
        self.x_train = x_train.A
        self.y_train = y_train
        self.x_test = x_test.A
        self.y_test = y_test
        self.n = y_train.shape[0]
         
        # Initialize parameters to zero, for lack of a better choice.
        self.betas = np.zeros(self.x_train.shape[1])
    
    
    def negative_lik(self, betas):
        return -1 * self.lik(betas)
    
    
    def lik(self, betas):
        ''' Likelihood of the data 
            under the current settings of parameters. '''
        
        # Data likelihood
        l = 0
        for i in range(self.n):
            l += np.log(sigmoid(self.y_train[i] * \
                     np.dot(betas, self.x_train[i,:])))
        
        # Prior likelihood
        for k in range(1, self.x_train.shape[1]):
            l -= (self.alpha / 2.0) * self.betas[k]**2
        
        return l
    
    
    def train(self):
        ''' Define the gradient and hand it off to a scipy gradient-based
            optimizer. '''
        
        # Define the derivative of the likelihood with respect to beta_k.
        
        # Need to multiply by -1 because we will be minimizing.
        # dB_k = lambda B, k : (k > 0) * self.alpha * B[k] - np.sum([ \
        #     self.y_train[i] * self.x_train[i, k] * \
        #     sigmoid(-self.y_train[i] *\
        #     np.dot(B, self.x_train[i,:])) \
        #     for i in range(self.n)])
        def dB_k(B, k):
            # fit = lambda x: np.array(x)[0]
            # print B, B.shape, self.x_train[1][0], self.x_train[1][0].shape, foo[0], foo[0].shape
            # print self.x_train, self.x_train.shape, type(self.x_train)
            # np.dot(B, self.x_train[1,:], 1)
            foo = (k > 0) * self.alpha * B[k] - \
                np.sum([self.y_train[i] * self.x_train[i, k] * \
                sigmoid(-self.y_train[i] * np.dot(B, self.x_train[i,:])) 
                for i in range(self.n)])
            return foo

        # The full gradient is just an array of componentwise derivatives
        dB = lambda B : np.array([dB_k(B, k) \
            for k in range(self.x_train.shape[1])])
        
        # Optimize
        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)
    
    # def training_reconstruction(self):
    #     p_y1 = np.zeros(self.n)
    #     for i in range(self.n):
    #         p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:]))
        
    #     return p_y1
    
    def test_predictions(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:]))
        
        return p_y1    

if __name__ == "__main__":
    # Prepare train data
    traindf = pd.read_json('train.json')
    # make clean ingredients string
    traindf['ingredients_string'] = [
        ' '.join([
            WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) 
                for line in lists
                ]).strip() for lists in traindf['ingredients']]       
    # Prepare test data
    testdf = pd.read_json('test.json')
    # make clean ingredients string
    testdf['ingredients_string'] = [
        ' '.join([
            WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) 
                for line in lists
                ]).strip() for lists in testdf['ingredients']]       
    # Vertorize train ingredients
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
    # Vertorize test ingredients
    corpusts = testdf['ingredients_string']
    tfidfts=vectorizertr.transform(corpusts)
    # Train data
    predictors_tr = tfidftr
    # Add numbers column to traindf object
    unic_cuisine = list(traindf['cuisine'].unique())
    traindf['cuisine_n'] = [unic_cuisine.index(x) for x in traindf['cuisine']]
    targets_tr = traindf['cuisine_n']
    # Test data
    predictors_ts = tfidfts


    # Run for a variety of regularization strengths
    alphas = [0, .001, .01, .1, 2]
    for j, a in enumerate(alphas):
        
        # Create a new learner, but use the same data for each run
        # Take only first 25 rows
        lr = LogisticRegressionCA(
            x_train=predictors_tr[:25, :].todense(), 
            y_train=targets_tr[:25],
            x_test=predictors_ts[:25, :].todense(),
            alpha=0.01)
     
        print "Initial likelihood:"
        print lr.lik(lr.betas)
         
        # Train the model
        lr.train()
         
        # Display execution info
        print 'Final betas:'
        print lr.betas
        print 'Final lik:'
        print lr.lik(lr.betas)
        print 'Prediction:'
        print lr.test_predictions()
