import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import l0learn
import sklearn

class L0L2Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_support_size = 0.025, n_alphas=5, gamma_min=0.001, gamma_max=10):
        self.penalty = 'L0L2'
        self.max_support_size = max_support_size
        self.n_alphas = n_alphas
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def fit(self, X, y):
        if self.max_support_size < 1:
            self.max_support_size = min(int(self.max_support_size*min(X.shape[0],X.shape[1])),X.shape[1])
            
        self.estimator = l0learn.fit(X.copy().values.astype(np.float64), y.copy().to_numpy(), penalty=self.penalty, max_support_size=self.max_support_size, num_gamma=self.n_alphas, gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        
        df = self.estimator.characteristics()
        stats = df[df['support_size'] <= self.max_support_size].sort_values('support_size', ascending = False).iloc[0, :]
        
        self.best_lambda = stats['l0']
        self.best_alpha = stats['l2']
        self.intercept_, self.coef_ = (lambda arr: (arr[0], arr[1:]))(self.estimator.coeff(lambda_0=self.best_lambda,gamma=self.best_alpha).toarray().reshape(-1, ))

    def predict(self, X):
        return self.estimator.predict(x = X, lambda_0=self.best_lambda, gamma=self.best_alpha).reshape(-1,)
    
class L0L2RegressorCV(BaseEstimator, RegressorMixin):
    def __init__(self, max_support_size = 0.025, n_alphas=5, gamma_min=0.001, gamma_max=10, cv=3, seed=0):
        self.penalty = 'L0L2'
        self.max_support_size = max_support_size
        self.n_alphas = n_alphas
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.cv = cv
        self.seed = seed

    def fit(self, X, y):
        if self.max_support_size < 1:
            self.max_support_size = min(int(self.max_support_size*min(X.shape[0],X.shape[1])),X.shape[1])
            
        print(self.max_support_size)
            
#         self.estimator = l0learn.cvfit(X.copy().values.astype(np.float64), y.copy().to_numpy(), num_folds=self.cv, seed=self.seed, penalty=self.penalty, max_support_size=self.max_support_size, num_gamma=self.n_alphas, gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        
#         gamma_mins = [(i, np.argmin(cv_mean), np.min(cv_mean)) for i, cv_mean in enumerate(self.estimator.cv_means)]
#         optimal_gamma_index, optimal_lambda_index, min_error = min(gamma_mins, key = lambda t: t[2])
        
#         #TODO: get self.best_lambda, self.best_alpha
#         self.best_lambda = self.estimator.lambda_0[optimal_gamma_index][optimal_lambda_index]
#         self.best_alpha = self.estimator.gamma[optimal_gamma_index]
#         self.intercept_, self.coef_ = (lambda arr: (arr[0], arr[1:]))(self.estimator.coeff(lambda_0=self.best_lambda,gamma=self.best_alpha).toarray().reshape(-1, ))
        train_idx, val_idx = sklearn.model_selection.train_test_split(np.arange(len(y)), train_size=0.8, random_state = self.seed)
    
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        self.estimator = l0learn.fit(X_train.copy().values.astype(np.float64), y_train.copy().to_numpy(), penalty=self.penalty, max_support_size=self.max_support_size, num_gamma=self.n_alphas, gamma_min=self.gamma_min, gamma_max=self.gamma_max)
    
        df = self.estimator.characteristics()
        stats = df[df['support_size'] <= self.max_support_size].sort_values('support_size', ascending = False)
    
        self.scores_ = [[] for _ in range(len(stats))]
        for i, reg_param in enumerate(range(len(stats))):
            hyper_params = stats.iloc[i,:]
            val_mse = np.mean((self.estimator.predict(x=X_val, lambda_0=hyper_params['l0'], gamma=hyper_params['l2']).reshape(-1, ) - y_val)**2)
            self.scores_[i].append(val_mse)
        self.scores_ = [np.mean(s) for s in self.scores_]

        best_params = stats.iloc[np.argmin(self.scores_),:]
            
        self.best_lambda = best_params['l0']
        self.best_alpha = best_params['l2']
        self.l1_ratio_ = self.best_lambda / (self.best_lambda+self.best_alpha)
        self.alpha_ = (self.best_lambda+self.best_alpha)
        self.intercept_, self.coef_ = (lambda arr: (arr[0], arr[1:]))(self.estimator.coeff(lambda_0=self.best_lambda,gamma=self.best_alpha).toarray().reshape(-1, ))

    def predict(self, X):
        return self.estimator.predict(x = X, lambda_0=self.best_lambda, gamma=self.best_alpha).reshape(-1,)


class L0Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_support_size = 0.025):
        self.penalty = 'L0'
        self.max_support_size = max_support_size

    def fit(self, X, y):
        if self.max_support_size < 1:
            self.max_support_size = min(int(self.max_support_size*min(X.shape[0],X.shape[1])),X.shape[1])
            
        self.estimator = l0learn.fit(X.copy().values.astype(np.float64), y.copy().to_numpy(), penalty=self.penalty, max_support_size=self.max_support_size)
        
        df = self.estimator.characteristics()
        stats = df[df['support_size'] <= self.max_support_size].sort_values('support_size', ascending = False).iloc[0, :]
        
        self.best_lambda = stats['l0']
        self.best_alpha = 0
        self.intercept_, self.coef_ = (lambda arr: (arr[0], arr[1:]))(self.estimator.coeff(lambda_0=self.best_lambda,gamma=self.best_alpha).toarray().reshape(-1, ))

    def predict(self, X):
        return self.estimator.predict(x = X, lambda_0=self.best_lambda, gamma=self.best_alpha).reshape(-1,)

#TODO: code up L0RegressorCV
