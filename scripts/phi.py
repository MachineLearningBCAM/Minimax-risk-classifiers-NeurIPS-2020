import numpy as np

import os

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.base import is_classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


class Phi(object):

    def __init__(self, r):
        self.r = r
        return

    def eval(self, X):
        '''
        The optimized evaluation of the instances X, phi(x,y) for all x in X and y=1,...,r

        :param X: unlabeled instances, np.array(float) (n_instances X n_features)
        :return: evaluation of the set of instances for all class labels.
            np.array(float), (n_instances X n_classes X phi.len)
        '''
        return np.zeros(1)

    def evaluate(self, x, y):
        '''
        Evaluation of single labeled instance (x,y)
        Non optimized
        :param x: an instance, np.array(float), n_features
        :param y: a class label, int
        :return: Evaluation of a single instance for the given class label. float
        '''
        return np.zeros(1)

    def estExp(self, X, Y):
        '''
        Average value of phi in the supervised dataset (X,Y)
        Used in the learning stage as an estimate of the expected value of phi, tau

        :param X: the set of unlabeled instances, np.array(numInstances,numFeatures)
        :param Y: np.array(numInstances)
        :return: Average value of phi, np.array(float) phi.len.
        '''
        return np.zeros(1)

    def len(self):
        '''
        return the length of Phi
        :return: length of phi, int
        '''
        return 0

    def numConfig(self):
        '''
        return the (upper bound of) number of configurations ot Phi
        :return:
        '''
        return 1

    def getConstr(self):
        '''
        Get the constraints required for determining the uncertainty set using phi with liner probabilistic
        classifiers, MRC.
        :return:
        '''

    def validate_input(self, X, y):
        '''Validate X and Y

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)

        y : array-like, shape (n_samples, )
            True labels.

        '''
        X, y = check_X_y(X, y, dtype=np.float64)

        # check that y is of non-regression type
        check_classification_targets(y)

        # labels can be encoded as float, int, or string literals
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        # The original class labels
        self.Y = self.label_encoder_.classes_

        # The encoded class labels
        self._encoded_classes = self.label_encoder_.transform(self.Y)

        # Encoded_labels = label_encoder.transform(label_encoder.classes_)
        y = self.label_encoder_.transform(y)

        self.r = self.Y.size

        return X, y

