import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

class BinaryTransformer:
    def __init__(self, depth=2):
        self.depth = depth
        self.dt_models = {}
        self.encoders = {}
        self.maps = {}
        self.feature_types = {}
    
    def fit(self, X, y):
        for feature_name in X.columns:
            feature = X[feature_name]
            if np.issubdtype(feature.dtype, np.number):
                unique_vals = feature.unique()
                if len(unique_vals) == 2:
                    self.maps[feature_name] = {unique_vals[0]: -1, unique_vals[1]: 1}
                    self.feature_types[feature_name] = 'binary'
                else:
                    dt = DecisionTreeRegressor(max_depth=self.depth, random_state=42)
                    dt.fit(feature.values.reshape(-1, 1), y)
                    self.dt_models[feature_name] = dt
                    self.feature_types[feature_name] = 'continuous'
            else:
                encoder = OneHotEncoder(sparse=False)
                encoded_feature = encoder.fit_transform(feature.values.reshape(-1, 1))
                self.encoders[feature_name] = encoder
                self.feature_types[feature_name] = 'categorical'
    
    def transform(self, X):
        assert set(self.feature_types.keys()) == set(X.columns), "X not compatible with the X BinaryTransformer was fitted on"
        
        transformed_X = pd.DataFrame()
        for feature_name in X.columns:
            feature = X[feature_name]
            if self.feature_types[feature_name] == 'binary':
                #assuming binary feature is 0, 1
                transformed_X[feature_name] = feature.map(self.maps[feature_name])
            elif self.feature_types[feature_name] == 'continuous':
                dt_model = self.dt_models[feature_name]
                leaf_indices = dt_model.apply(feature.values.reshape(-1, 1))
                ohe = OneHotEncoder(sparse=False)
                encoded = ohe.fit_transform(leaf_indices.reshape(-1, 1))
                transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=ohe.get_feature_names_out([feature_name]))], axis = 1)
            else:
                ohe = self.encoders[feature_name]
                encoded = ohe.transform(feature.values.reshape(-1, 1))
                transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=ohe.get_feature_names_out([feature_name]))], axis = 1)
        
        return transformed_X.replace({0:-1}).astype(int)
    
    def fit_and_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
