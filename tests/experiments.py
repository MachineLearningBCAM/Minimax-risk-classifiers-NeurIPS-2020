import numpy as np
from scripts.MRC import MRC
from scripts.phiThreshold import PhiThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

import time

#import the datasets
from datasets import *
# dont show the warnings
import warnings
warnings.filterwarnings("ignore")

#data sets
loaders = [load_glass, load_mammographic, load_haberman, load_indian_liver, load_diabetes, load_credit]
dataName= ["glass", "mammographic", "haberman", "indian_liver", "diabetes", "credit"]

loadersBounds = [load_adult, load_magic]
dataNameBounds= ["adult", "magic"]


def ICML20_error(s=0.25, mumVars= 400, random_seed= 1):
    '''
    Experimentation: ICML 2020
    1st experiment: error estimation with bounds
    '''
    res_mean = np.zeros(len(dataName))
    res_std = np.zeros(len(dataName))
    np.random.seed(random_seed)

    for j, load in enumerate(loaders):
        X, origY = load(return_X_y=True)
        n, d= X.shape

        #Map the values of Y from 0 to r-1
        domY= np.unique(origY)
        r= len(domY)
        Y= np.zeros(X.shape[0], dtype= np.int)
        for i,y in enumerate(domY):
            Y[origY==y]= i

        print(" ############## \n" + dataName[j] + " n= " + str(n) + " , d= " + str(d) + ", cardY= "+ str(r))


        clf = MRC(r=r, phi=PhiThreshold(r=r, m=int(mumVars/r), k=int(mumVars/(r*d))), s=s)

        # Preprocess
        trans = SimpleImputer(strategy='median')
        X = trans.fit_transform(X, Y)

        # Generate the partitions of the stratified cross-validation
        cv = StratifiedKFold(n_splits=10, random_state=random_seed)

        np.random.seed(random_seed)
        cvError= list()
        numMu= 0
        numthres= 0
        upper= 0
        lower= 0
        # Paired and stratified cross-validation
        for train_index, test_index in cv.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            clf.fit(X_train, y_train)
            numMu += np.sum(clf.mu != 0)
            numthres += clf.phi.m
            upper += clf.upper
            lower += clf.getLowerBound()

            y_pred= clf.predict(X_test)

            cvError.append(np.average(y_pred != y_test))

        res_mean[j] = np.average(cvError)
        res_std[j] = np.std(cvError)

        print(" error= " + ":\t" + str(res_mean[j]) + "\t+/-\t" + str(res_std[j]) + "\n" +
              " upper= " + str(upper/10)+"\t lower= " + str(lower/10) + "\n ############## \n")


def ICML20_bounds(s=0.25, mumVars= 400, seed= 1):
    '''
    Experimentation: ICML 2020
    2nd experiment: evoution of the upper-lower bounds with respect to the size of the training set
    '''

    np.random.seed(seed)

    m = np.inf
    k = None
    minTrain = 50
    for j, load in enumerate(loadersBounds):
        X, origY = load(return_X_y=True)
        (n, d) = X.shape

        num_n = np.min((int(np.ceil(np.log2(n) - np.log2(minTrain * 2))),
                        int(np.ceil(np.log2(14000) - np.log2(minTrain * 2))) + 1))
        n_test = n - minTrain * 2 ** (num_n - 1)

        # Map the values of Y from 0 to r-1
        domY = np.unique(origY)
        r = len(domY)
        Y = np.zeros(X.shape[0], dtype=np.int)
        for i, y in enumerate(domY):
            Y[origY == y] = i

        print(" ############## \n" + dataNameBounds[j] + " n= " + str(n) + " , d= " + str(d) + ", cardY= " + str(
            r) + "\n ############## \n")

        clf = MRC(r=r, phi=PhiThreshold(r=r, m=int(mumVars / r), k=np.max((3,int(mumVars/(r*d))))), s=s)

        # Preprocess
        trans = SimpleImputer(strategy='median')
        X = trans.fit_transform(X, Y)

        error = list()
        trainTime = list()
        testTime = list()
        numMu = list()
        numthres = list()
        upper = list()
        lower = list()
        randomize = np.random.permutation(n)
        for i in np.arange(num_n):
            n_train = minTrain * 2 ** i
            Xtrain = X[randomize[:n_train]]
            Ytrain = Y[randomize[:n_train]]
            Xtest = X[randomize[-n_test:]]
            Ytest = Y[randomize[-n_test:]]

            auxTime = time.time()
            np.random.seed(seed)
            clf.fit(Xtrain, Ytrain)
            auxTime = time.time() - auxTime
            trainTime.append(auxTime)

            np.random.seed(seed)
            lower.append(clf.getLowerBound())

            numMu.append(np.sum(clf.mu != 0))
            numthres.append(clf.phi.m)
            upper.append(clf.upper)

            auxTime = time.time()
            y_pred = clf.predict(Xtest)
            auxTime = time.time() - auxTime
            testTime.append(auxTime)

            error.append(np.average(y_pred != Ytest))

            print("\n n_train= " + str(n_train) + "\n" +
                  " upper= " + str(upper[-1]) + "\n" +
                  " error= " + str(error[-1]) + "\n" +
                  " lower= " + str(lower[-1]) + "\n")

        print(" ############## \n" + dataNameBounds[j] +
              " n= " + str(n) + " , d= " + str(d) + ", cardY= " + str(r) + "\n ############## \n" +
              " n_train= " + str([minTrain * 2 ** i for i in np.arange(num_n)]) + "\n" +
              " upper= " + str(upper) + "\n" +
              " error= " + str(error) + "\n" +
              " lower= " + str(lower) + "\n")

if __name__ == '__main__':
    ICML20_error()
    ICML20_bounds()
