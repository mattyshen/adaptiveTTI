import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import l0learn
import l0bnb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from imodels.util.arguments import check_fit_arguments

import sys
import os
import time

sys.path.append("../interpretDistill")

#from FTutils import powerset_pruned, compute_subset_product, compute_subset_product_naive
from binaryTransformer import BinaryTransformer

class FTDistill:
    def __init__(self, distill_model = None, selection = 'L1', lam1 = 1.0, lam2 = None, size_interactions = 4, depth=1):
        self.distill_model = distill_model
        self.selection = selection
        self.lam1 = lam1
        self.lam2 = lam2
        self.size_interactions = size_interactions
        self.bt = BinaryTransformer(depth=depth)
        
        if (selection not in ['L1', 'L0']) and lam2 is None:
            raise 'Interaction selection model chosen requires `lam2` argument'
        
        #TODO: Implement L0, L0L2
        if self.selection == 'L1':
            self.regression_model = ElasticNet(alpha = self.lam1, l1_ratio = 1, fit_intercept = False)
        elif self.selection == 'L0':
            raise NotImplementedError("L0 interaction selection not implemented")
        elif self.selection == 'L1L2':
            self.regression_model = ElasticNet(alpha = self.lam1+self.lam2, l1_ratio = self.lam1/(self.lam1+self.lam2), fit_intercept = False)
        elif self.selection == 'L0L2':
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
        
        poly_features = list(map(lambda s: set(s.split()), self.poly.get_feature_names_out(X.columns)))
        
        feats_allowed = [all([len(pot_s.intersection(s)) < 2 for s in self.no_interaction]) for pot_s in poly_features]
        
        Chi = pd.DataFrame(self.poly.transform(X), columns = list(map(lambda f: tuple(f), poly_features))).loc[:, feats_allowed]
        
        self.features = Chi.columns.to_list()

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
    def __init__(self, distill_model=None, selection='L1', lam1_range=np.array([1.0]), lam2_range=np.array([1.0]), k_cv=5, size_interactions=4, depth=1):
        super().__init__(distill_model, selection, lam1_range[0], lam2_range[0], size_interactions, depth)
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
        param_grid = {'alpha': self.lam1_range + self.lam2_range, 'l1_ratio': self.lam1_range/(self.lam1_range + self.lam2_range)}
        self.grid_search = GridSearchCV(self.regression_model, param_grid, cv=self.k_cv)
        self.grid_search.fit(Chi, y_distill)
        
        self.lam1 = self.grid_search.best_params_['alpha'] * self.grid_search.best_params_['l1_ratio']
        self.lam2 = self.grid_search.best_params_['alpha'] - self.lam1
        self.regression_model = self.grid_search.best_estimator_
        
        return self
