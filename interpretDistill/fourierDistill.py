import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import l0learn
import l0bnb
from sklearn.linear_model import LinearRegression, Ridge 
from celer import Lasso, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures

from imodels.util.arguments import check_fit_arguments

import sys
import os
import time

sys.path.append("../interpretDistill")

#from FTutils import powerset_pruned, compute_subset_product, compute_subset_product_naive
#from binaryTransformer import BinaryTransformer

class FTDistill:
    def __init__(self, distill_model = None, selection = 'L1', lam1 = 1.0, lam2 = 1.0, size_interactions = 4, max_features=None):
        self.distill_model = distill_model
        self.selection = selection
        self.lam1 = lam1
        self.lam2 = lam2
        self.size_interactions = size_interactions
        self.max_features = max_features
        
        if (selection not in ['L1', 'L0']) and lam2 is None:
            raise 'Interaction selection model chosen requires `lam2` argument'
        
        #TODO: Implement L0, L0L2
        elif self.selection == 'OLS':
            self.regression_model = LinearRegression(fit_intercept = False)
        elif self.selection == 'L1':
            self.regression_model = ElasticNet(alpha = self.lam1, l1_ratio = 1, fit_intercept = False)
        elif self.selection == 'L0':
            assert self.max_features is not None, "L0 based models require `max_features` argument"
            raise NotImplementedError("L0 interaction selection not implemented")
        elif self.selection == 'L1L2':
            self.regression_model = ElasticNet(alpha = self.lam1+self.lam2, l1_ratio = self.lam1/(self.lam1+self.lam2), fit_intercept = False)
        elif self.selection == 'L0L2':
            assert self.max_features is not None, "L0 based models require `max_features` argument"
            raise NotImplementedError("L0L2 interaction selection not implemented")
        else:
            self.regression_model = ElasticNet(alpha = self.lam1+self.lam2, l1_ratio = self.lam1/(self.lam1+self.lam2), fit_intercept = False)

    def fit(self, X, y = None, no_interaction=[]):
        """
        Train the model using the training data.

        Parameters:
        X_train : array-like, shape (n_samples, n_features)
            Training data.
        y_train : array-like, shape (n_samples,)
            Target values.

        Returns:
        self : object
            Returns the instance itself.
        """
        #TODO: check X is compatible with self.distill_model & everything is binarized
        if self.distill_model is None and y is None:
            raise "No `distill_model` was passed during initialization and no `y` passed in fit."
            
        if self.distill_model is not None:
            y_distill = self.distill_model(X)
        else:
            y_distill = y
            
        #X = self.bt.fit_and_transform(X, y)
        self.no_interaction = no_interaction

        self.poly = PolynomialFeatures(degree = self.size_interactions, interaction_only = True)
        self.poly.fit(X)
        print('poly fitted')
        poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        
        feats_allowed = [all([len(pot_s.intersection(s)) < 2 for s in self.no_interaction]) for pot_s in poly_features]
        
        Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features))).loc[:, feats_allowed]
        
        self.features = Chi.columns.to_list()

        print('regression model fitting')
        print(Chi.shape)
        self.regression_model.fit(Chi, y_distill)
        
        return self
    
    def predict(self, X):
        """
        Predict using the trained model.

        Parameters:
        X_test : array-like, shape (n_samples, n_features)
            Test data.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted target values.
        """
        #TODO: check X is compatible with self.distill_model & everything is binarized
        
        #X = self.bt.transform(X)
        
        poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        feats_allowed = [all([len(pot_s.intersection(s)) < 2 for s in self.no_interaction]) for pot_s in poly_features]
        
        Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features))).loc[:, feats_allowed]
        
        return self.regression_model.predict(Chi)

class FTDistillCV(FTDistill):
    def __init__(self, distill_model=None, selection='L1', lam1_range=np.array([1.0]), lam2_range=np.array([1.0]), k_cv=5, size_interactions=3):
        super().__init__(distill_model, selection, lam1_range[0], lam2_range[0], size_interactions)
        self.k_cv = k_cv
        self.lam1_range = lam1_range
        self.lam2_range = lam2_range
        self.grid_search = None

    def fit(self, X, y=None, no_interaction=[]):
        if self.distill_model is None and y is None:
            raise ValueError("No `distill_model` was passed during initialization and no `y` passed in fit.")
            
        if self.distill_model is not None:
            y_distill = self.distill_model(X)
        else:
            y_distill = y
        
        #X = self.bt.fit_and_transform(X, y)
        self.no_interaction = no_interaction

        self.poly = PolynomialFeatures(degree = self.size_interactions, interaction_only = True)
        self.poly.fit(X)
        
        poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        
        feats_allowed = [all([len(pot_s.intersection(s)) < 2 for s in self.no_interaction]) for pot_s in poly_features]
        
        Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features))).loc[:, feats_allowed]
        
        self.features = Chi.columns.to_list()
        
        # Grid search over lam1 and lam2 values
        self.regression_model = ElasticNetCV(l1_ratio = 0.5, cv=self.k_cv, fit_intercept=False, max_epochs = 5000, n_alphas = 5, tol = 0.01)
        self.regression_model.fit(Chi, y)
#         param_grid = {'alpha': self.lam1_range + self.lam2_range, 'l1_ratio': self.lam1_range/(self.lam1_range + self.lam2_range)}
#         self.grid_search = GridSearchCV(self.regression_model, param_grid, cv=self.k_cv)
#         self.grid_search.fit(Chi, y_distill)
        
#         self.lam1 = self.grid_search.best_params_['alpha'] * self.grid_search.best_params_['l1_ratio']
#         self.lam2 = self.grid_search.best_params_['alpha'] - self.lam1
#         self.regression_model = self.grid_search.best_estimator_
        
        return self

    
class FTDistillClassifierCV(FTDistill):
    def __init__(self, distill_model=None, selection='L1', lam1_range=np.array([0.01, 0.1, 1.0]), k_cv=3, size_interactions=3):
        super().__init__(distill_model, selection, lam1_range[0], size_interactions)
        self.k_cv = k_cv
        self.lam1_range = lam1_range
        self.grid_search = None
        self.regression_model = LogisticRegression(C = 1/lam1_range[0], penalty = 'l1', max_epochs = 5000, max_iter=100)

    def fit(self, X, y=None, no_interaction=[]):
        if self.distill_model is None and y is None:
            raise ValueError("No `distill_model` was passed during initialization and no `y` passed in fit.")
            
        if self.distill_model is not None:
            y_distill = self.distill_model(X)
        else:
            y_distill = y
        
        #X = self.bt.fit_and_transform(X, y)
        self.no_interaction = no_interaction

        self.poly = PolynomialFeatures(degree = self.size_interactions, interaction_only = True)
        print('poly fitting')
        start = time.time()
        self.poly.fit(X)
        end = time.time()
        print(f'poly fit time {end-start}')
        print('poly features')
        start = time.time()
        #poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        end = time.time()
        print(f'poly features time {end-start}')
        
        if no_interaction != []:
        
            feats_allowed = [all([len(pot_s.intersection(s)) < 2 for s in self.no_interaction]) for pot_s in poly_features]

            Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features))).loc[:, feats_allowed]
            
        else:
            print('empty no_interaction')
            Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features)))
        
        self.features = Chi.columns.to_list()
        
        self.scores_ = [[] for _ in self.lam1_range]
        #scorer = kwargs.get("scoring", log_loss)
        kf = KFold(n_splits=self.k_cv)
        for i, (train_index, test_index) in enumerate(kf.split(Chi)):
            print(i)
            Chi_out, y_out = Chi.iloc[test_index, :], y.iloc[test_index]
            Chi_in, y_in = Chi.iloc[train_index, :], y.iloc[train_index]
            for i, reg_param in enumerate(self.lam1_range):
                base_est = LogisticRegression(C = 1/reg_param, penalty = 'l1', max_epochs = 50000, max_iter=50)
                base_est.fit(Chi_in, y_in)
                self.scores_[i].append(np.mean(base_est.predict(Chi_out) == y_out))
        self.scores_ = [np.mean(s) for s in self.scores_]

        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        self.regression_model = LogisticRegression(C = 1/self.reg_param, penalty = 'l1', max_epochs = 5000, max_iter=100)
        self.regression_model.fit(Chi, y)
        
        return self