import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.base import clone


import os

from featurizer_utils import binary_map, bit_repr, get_leaf_node_indices

class RegFeaturizer:
    def __init__(self, depth=2, bit=True, seed=0):
        self.depth = depth
        self.dt = DecisionTreeRegressor(max_depth=self.depth, random_state=seed)
        self.dt_models = {}
        self.encoders = {}
        self.maps = {}
        self.feature_types = {}
        self.no_interaction = []
        self.bit = bit
        self.sizes = {}
        self.seed=seed
    
    def fit(self, X, y):
        for feature_name in X.columns:
            feature = X[feature_name]
            if np.issubdtype(feature.dtype, np.number):
                unique_vals = feature.unique()
                if len(unique_vals) == 2:
                    self.maps[feature_name] = binary_map(feature)
                    self.feature_types[feature_name] = 'binary'
                else:
                    dt = clone(self.dt)
                    dt.fit(feature.values.reshape(-1, 1), y)
                    self.dt_models[feature_name] = dt
                    self.feature_types[feature_name] = 'continuous'
            else:
                if self.bit:
                    # lb = LabelBinarizer()
                    # lb.fit(feature.to_list())
                    # self.encoders[feature_name] = lb
                    self.feature_types[feature_name] = 'categorical'
                else:
                    encoder = OneHotEncoder(sparse_output=False)
                    encoded_feature = encoder.fit_transform(feature.values.reshape(-1, 1))
                    self.encoders[feature_name] = encoder
                    self.feature_types[feature_name] = 'categorical'
    
    def transform(self, X):
        assert set(self.feature_types.keys()) == set(X.columns), "X not compatible with the X BinaryTransformer was fitted on"
        
        transformed_X = pd.DataFrame()
        for i, feature_name in enumerate(X.columns):
            # print(i, feature_name, transformed_X.isnull().values.any())
            # if i == 1:
            #     transformed_X.reset_index(drop=True, inplace=True)
            feature = X[feature_name]
            if self.feature_types[feature_name] == 'binary':
                transformed_X.reset_index(drop=True, inplace=True)
                transformed_X = pd.concat([transformed_X, pd.DataFrame(X[feature_name].map(self.maps[feature_name]).to_numpy(), columns=[feature_name])], axis = 1)
                self.sizes[feature_name] = 1
              
            elif self.feature_types[feature_name] == 'continuous':
                dt_model = self.dt_models[feature_name]
                leaf_indices = dt_model.apply(feature.values.reshape(-1, 1))
                all_cats = get_leaf_node_indices(dt_model.tree_)
                if self.bit:
                    df_transformed, new_columns = bit_repr(pd.Series(leaf_indices, name = f'{feature_name}_leaf'), self.depth)
                    #self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    df_transformed.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, df_transformed], axis = 1)
                    self.sizes[feature_name] = df_transformed.shape[1]
                else:
                    ohe = OneHotEncoder(categories = [all_cats], sparse_output=False)
                    encoded = ohe.fit_transform(leaf_indices.reshape(-1, 1))
                    new_columns = ohe.get_feature_names_out([feature_name])
                    self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=new_columns)], axis = 1)
                    self.sizes[feature_name] = len(new_columns)
            else:
                if self.bit:
                    df_transformed, new_columns = bit_repr(feature, self.depth)
                    #self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    df_transformed.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, df_transformed], axis = 1)
                    self.sizes[feature_name] = df_transformed.shape[1]
                    # lb = self.encoders[feature_name]
                    # encoded = lb.transform(feature.to_list())
                    # new_columns = [f'{feature_name}_bit_{i}' for i in range(encoded.shape[1])]
                    # self.no_interaction.append(set(new_columns))
                    # transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=new_columns)], axis = 1)
                else:
                    ohe = self.encoders[feature_name]
                    encoded = ohe.transform(feature.values.reshape(-1, 1))
                    new_columns = ohe.get_feature_names_out([feature_name])
                    self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=new_columns)], axis = 1)
                    self.sizes[feature_name] = len(new_columns)
        return transformed_X.astype(int).replace({-1:-1, 0:-1, 1:1})
    
    def fit_and_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

    
class ClassFeaturizer(RegFeaturizer):
    def __init__(self, depth=2, bit=True, seed=0):
        super().__init__(depth, bit, seed)
        self.dt = DecisionTreeClassifier(max_depth=self.depth, random_state=seed)