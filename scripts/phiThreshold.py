import numpy as np
import itertools as it
from scripts.phi import Phi
import scipy.special as scs
import csv as csv
from sklearn.tree import DecisionTreeClassifier
import time
class PhiThreshold(Phi):
    '''
    Phi function composed by products of (univariate) threshold features.
    A threshold feature is a funtion, f(x;t,d)=1 when x_d<t and 0 otherwise.
    A produc of threshold features is an indicator of a region and its expectancy is closely related to cumulative
    distributions.
    '''

    def __init__(self, r, m= 10, k= None):
        # number of classes
        self.r= r
        # number of product thresholds
        self.m= m
        # maximum number of univariate thresholds for each dimension
        self.k= k
        # length of phi(x,y)
        self.len= m*r
        # the type of constraints used: linear or non-linear
        if self.r > 4:
            self.linConstr= False
        else:
            self.linConstr = True

        self.is_fitted_ = False
        return

    def fit(self, X, y):
        '''
        Learn the set of product thresholds
        :param X: unlabeled training instances
        :param y: training class labels
        :return:
        '''

        # number of univariate thresholds for each dimension
        n, d= X.shape

        #Learn the product thresholds
        #self.thrsDim, self.thrsVal= self._predefCredit()
        self.thrsDim, self.thrsVal = self._decTreeSplit(X, y, self.k)


        # Update: the learning process can change the number of thresholds m, and thus the length of phi
        self.m = len(self.thrsDim)
        self.len= self.m* self.r

        # Learn the configurations of phi using the training
        self.learnF(X)

        self.is_fitted_ = True
        return

    def _decTreeSplit(self, X, Y, k=None):
        '''
        Learn the univariate thresholds by using the split points of decision trees for each dimension of data
        :param X: unlabeled training samples
        :param y: labels of the training samples
        :param k: maximum number of leaves of the tree
        :return: the univariate thresholds
        '''

        (n, d) = X.shape

        # Zero order threshold (always 1)
        thrsDim = [0]
        thrsVal = [np.inf]

        # update the list of thresholds with different configs,
        prodThrsVal = [thrsVal]
        prodThrsDim = [thrsDim]

        # One order thresholds: all the univariate thresholds
        for dim in range(d):
            if k== None:
                dt = DecisionTreeClassifier()
            else:
                dt= DecisionTreeClassifier(max_leaf_nodes=k+1)

            dt.fit(np.reshape(X[:,dim],(n,1)),Y)

            dimThrsVal= np.sort(dt.tree_.threshold[dt.tree_.threshold!= -2])
            for t in dimThrsVal:
                prodThrsVal.append([t])
                prodThrsDim.append([dim])

        return prodThrsDim, prodThrsVal

    def learnF(self, X):
        '''
        Stores all the unique configurations of x in X for every value of y
        :param X: Unlabeled data, np.array(float) n_instances X n_dimensions
        :return: None
        '''

        n= X.shape[0]
        aux= time.time()
        phi= self.eval(X)
        # Disctinct configurations for phi_x,y for x in X and y=1,...,r.
        # Used in the definition of the constraints of the MRC
        # F is a tuple of floats with dimension n_intances X n_classes X m
        self.F= np.vstack({tuple(phi_xy) for phi_xy in phi.reshape((n,self.r*self.len))})
        self.F.shape = (self.F.shape[0], self.r, int(self.F.shape[1] / self.r))

        return

    def eval(self,X):
        '''
        The optimized evaluation of the instances X, phi(x,y) for all x in X and y=0,...,r-1

        :param X: unlabeled instances, np.array(float) (n_instances X n_features)
        :return: evaluation of the set of instances for all class labels.
            np.array(float), (n_instances X n_classes X phi.len)
        '''

        n= X.shape[0]

        # product threshold values
        # [[p11,...,p1m],...,[p11,...,p1m],...,[pn1,...pnm],...,[pn1,...pnm]] where pij is the j-th prod theshold
        # for i-th unlabeled instance, r*n_samples X phi.len
        phi = np.zeros((n, self.r, self.len), dtype=int)
        for thrsInd in range(self.m):
            phi[:, np.arange(self.r), np.arange(self.r)*self.m+thrsInd] = \
                np.tile(np.all(X[:, self.thrsDim[thrsInd]] <= self.thrsVal[thrsInd],
                               axis=1).astype(np.int),(self.r, 1)).transpose()

        return phi

    def evaluate(self, X, Y):
        '''
        Evaluation of a labeled set of instances (X,Y), phi(x,y) for (x,y) in (X,Y)

        Used in the learning stage for estimating the expected value of phi, tau

        :param X: the set of unlabeled instances, np.array(numInstances,numFeatures)
        :param Y: np.array(numInstances)
        :return: The evaluation of phi the the set of instances (X,Y),
            np.array(int) with dimension n_instances X (n_classes * n_prod_thresholds)
        '''

        n = X.shape[0]

        # product threshold values
        # [[p11,...,p1m],...,[p11,...,p1m],...,[pn1,...pnm],...,[pn1,...pnm]] where pij is the j-th prod theshold
        # for i-th unlabeled instance, r*n_samples X phi.len
        phi = np.zeros((n, self.len), dtype=np.float)
        for thrsInd in range(self.m):
            phi[np.arange(n), thrsInd + Y * self.m] = \
                np.all(X[:, self.thrsDim[thrsInd]] <= self.thrsVal[thrsInd], axis=1).astype(np.int)

        return phi

    def estExp(self,X,Y):
        '''
        Average value of phi in the supervised dataset (X,Y)
        Used in the learning stage as an estimate of the expected value of phi, tau

        :param X: the set of unlabeled instances, np.array(numInstances,numFeatures)
        :param Y: np.array(numInstances)
        :return: Average value of phi, np.array(float) phi.len.
        '''

        return np.average(self.evaluate(X, Y), axis= 0)

    def numConfig(self):
        '''
        return the (upper bound of) number of configurations ot Phi

        one-hot encoding (y & threshold(x,ti,di)) for i=0,...,n_prod_thrs-1 and y=0,...,r-1

        :return: the maximum number of configurations fo phi (assuming that product thresholds are incomparable)
        '''
        return self.r * (2 ** self.m)

    def getLearnConstr(self, linConstr):
        '''
        Get the constraints required for determining the uncertainty set using phi with liner probabilistic
        classifiers, MRC.
        :return: The index of the variables that have to be added for creating the constraints of for learning
        the MRC. Two type of constraints: 1.exponential and 2:linear

        FORMAT:
        1.-Exponential: For each x with different phi_x average, value of F_x over every subset of the class values.
        The last row corresponds to the number of class values selected for averaging F_x. Returns a
        np.matrix(float), (n_instances * 2^r-1) X (num_classes * num_prod_feats + 1)
        '''

        n= self.F.shape[0]
        if linConstr:#self.r<4:
            #Linear constraints. Exponential number in r
            avgF= np.vstack((np.sum(self.F[:, S, ], axis=1)
                             for numVals in range(1, self.r+1)
                             for S in it.combinations(np.arange(self.r), numVals)))
            cardS= np.arange(1, self.r+1).repeat([n*scs.comb(self.r, numVals)
                                                 for numVals in np.arange(1, self.r+1)])[:, np.newaxis]

            constr= np.hstack((avgF, cardS))
        else:
            #Non-linear constraints (defined used the positive part). The number is independent from r
            constr= self.F

        return constr

    def getLowerConstr(self):
        '''
        Get the constraints required for determining the uncertainty set using phi with liner probabilistic
        classifiers, MRC.
        :return: The index of the variables that have to be added for creating the constraints of for learning
        the MRC. Two type of constraints: 1.exponential and 2:linear

        FORMAT:
        1.-Exponential: For each x with different phi_x average, value of F_x over every subset of the class values.
        The last row corresponds to the number of class values selected for averaging F_x. Returns a
        np.matrix(float), (n_instances * 2^r-1) X (num_classes * num_prod_feats + 1)
        '''

        constr= self.F

        return constr
