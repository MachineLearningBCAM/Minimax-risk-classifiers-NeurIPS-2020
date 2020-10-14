
import numpy as np
import cvxpy as cvx

import time

from sklearn.base import BaseEstimator, ClassifierMixin


class MRC(BaseEstimator, ClassifierMixin):
    '''
    Minimax risk classifier with using univariate threshold-based feature mappings
    Submitted to ICML 2020
    '''

    def __init__(self, r, phi, equality=False, s=0.25, deterministic=False, seed=0):
        '''

        :param r: the number of values of class variable
        :param phi: Features of the MRC
        :param equality: the type of Learning. If true the MRC is asymptotically calibrated, if false the MRC is
        approximately calibrated.
        :param deterministic: if deterministic is false the MRC decision function is arg_c rand p(c|x) and if it is true
        the decision function is arg_c max p(c|x)
        :param seed: random seed
        '''
        self.r= r
        self.phi = phi
        self.equality = equality
        self.s = s
        self.deterministic = deterministic
        self.seed= seed
        if self.r> 4:
            self.linConstr= False
        else:
            self.linConstr= True



    def fit(self, X, Y):
        """
        Fit learning using....

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array-like, shape (n_samples)

        Returns
        -------
        self : returns an instance of self.

        """
        self.phi.linConstr= self.linConstr
        self.phi.fit(X, Y)
        self._minimaxRisk(X,Y)


    def _minimaxRisk(self, X, Y):
        '''
        Solves the minimax risk problem
        :param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        :param Y: the labels, numpy.array(int) with shape (numInstances,)
        :return: the upper bound of the MRC given directly by the solution to the minimax risk problem
        '''

        # Constants
        n= X.shape[0]
        m= self.phi.len
        aux= time.time()
        self.tau= self.phi.estExp(X,Y)

        self.a= np.clip(self.tau- self.s/np.sqrt(n),0,1)
        self.b= np.clip(self.tau+ self.s/np.sqrt(n),0,1)

        # Variables
        mu_a = cvx.Variable(m)
        mu_b = cvx.Variable(m)
        nu = cvx.Variable()

        # Cost function
        cost = self.b.T@mu_b - self.a.T@mu_a + nu

        # Objective function
        objective = cvx.Minimize(cost)

        # Constraints
        if self.linConstr:
            #Exponential number in num_class of linear constraints
            M = self.phi.getLearnConstr(self.linConstr)
            F = M[:, :m]
            cardS= M[:, -1]
            numConstr= M.shape[0]
            constraints= [mu_a >= 0, mu_b >= 0]
            constraints.extend([F[i, :]@(mu_b-mu_a) + cardS[i]*nu >= cardS[i]-1 for i in range(numConstr)])
        else:
            #Constant number in num_class of non-linear constraints
            F = self.phi.getLearnConstr(self.linConstr)
            numConstr = F.shape[0]
            constraints = [mu_a >= 0, mu_b >= 0]
            constraints.extend([cvx.sum(cvx.pos(np.ones(self.r) - (F[i, :, :]@ (mu_b - mu_a) + np.ones(self.r) * nu))) <= 1 for i in range(numConstr)])

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(verbose=False)

        # Optimal values
        self.mu_a= np.round(mu_a.value, 5)
        self.mu_b= np.round(mu_b.value, 5)
        self.mu= np.round(mu_b.value - mu_a.value, 5)
        self.nu= np.round(nu.value, 5)

        # Upper bound
        self.upper= self.b.T@mu_b.value - self.a.T@mu_a.value + self.nu

    def getLowerBound(self):
        '''
        Obtains the lower bound of the fitted classifier: unbounded...

        :param X:
        :param Y:
        :return:
        '''

        # Variables
        m= self.phi.len
        low_mu_a = cvx.Variable(m)
        low_mu_b = cvx.Variable(m)
        low_nu = cvx.Variable()

        # Cost function
        cost = self.b.T@low_mu_b - self.a.T@low_mu_a + low_nu

        # Objective function
        objective = cvx.Minimize(cost)

        # Constraints
        constraints= [low_mu_a >= 0, low_mu_b >= 0]

        Phi = self.phi.getLowerConstr()
        numConstr= Phi.shape[0]

        # epsilon
        eps = np.clip(1 - (Phi@self.mu + self.nu), 0., None)
        c= np.sum(eps, axis=1)
        zeros= np.isclose(c, 0)
        c[zeros]= 1
        eps[zeros, :]= 1/self.r
        c= np.tile(c, (self.r, 1)).transpose()
        eps/= c

        constraints.extend(
            [Phi[i, y, :]@(low_mu_b - low_mu_a) + low_nu >= eps[i, y]-1
             for i in range(numConstr) for y in range(self.r)])


        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(verbose=False)

        # Upper bound
        self.mu_a_l= low_mu_a.value
        self.mu_a_l[np.isclose(self.mu_a_l,0)]= 0
        self.mu_b_l= low_mu_b.value
        self.mu_b_l[np.isclose(self.mu_b_l,0)]= 0
        self.nu_l= low_nu.value
        self.nu_l[np.isclose(self.nu_l,0)]= 0
        self.lower= -self.b.T@self.mu_b_l + self.a.T@self.mu_a_l - self.nu_l


        return self.lower

    def predict_proba(self, X):
        '''
        Return the class conditional probabilities for each unlabeled instance
        :param X: the unlabeled instances, np.array(double) n_instances X dimensions
        :return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        # Unnormalized conditional probabilityes
        hy_x = np.clip(1 - (np.dot(Phi, self.mu) + self.nu), 0., None)


        # normalization constraint
        c = np.sum(hy_x, axis=1)
        # check when the sum is zero
        zeros = np.isclose(c, 0)
        c[zeros] = 1
        hy_x[zeros, :] = 1 / self.r
        c = np.tile(c, (self.r, 1)).transpose()


        return hy_x / c

    def predict(self, X):
        '''Returns the predicted classes for X samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        returns
        -------
        y_pred : array-like, shape (n_samples, )
            y_pred is of the same type as self.classes_.

        '''

        if not self.deterministic:
            np.random.seed(self.seed)

        proba = self.predict_proba(X)

        if self.deterministic:
            ind = np.argmax(proba, axis=1)
        else:
            ind = [np.random.choice(self.r, size= 1, p=pc)[0] for pc in proba]

        return ind
