import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import os

from interpretDistill.featurizer_utils import binary_map, bit_repr, get_leaf_node_indices
#from featurizer_utils import binary_map, bit_repr, get_leaf_node_indices

class RegFeaturizer:
    def __init__(self, depth=2, bit=1, empty_cat=1, seed=0):
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
        self.empty_cat=empty_cat
    
    def fit(self, X, y):
        for feature_name in X.columns:
            feature = X[feature_name]
            if pd.api.types.is_float_dtype(feature) and len(feature.unique()) > 10:
                unique_vals = feature.unique()
                if len(unique_vals) == 2:
                    self.maps[feature_name] = binary_map(feature)
                    self.feature_types[feature_name] = 'binary'
                else:
                    dt = clone(self.dt)
                    dt.fit(feature.values.reshape(-1, 1), y)
                    mapping = defaultdict(lambda: 0)
                    unique_values = sorted(np.unique(dt.apply(feature.values.reshape(-1, 1))))
                    mapping.update({val: i+self.empty_cat for i, val in enumerate(unique_values)})
                    self.dt_models[feature_name] = dt
                    self.maps[feature_name] = mapping
                    self.feature_types[feature_name] = 'continuous'
            else:
                if self.bit:
                    unique_values = sorted(feature.unique())
                    mapping = defaultdict(lambda: 0)
                    mapping.update({val: i+self.empty_cat for i, val in enumerate(unique_values)})
                    self.maps[feature_name] = mapping
                    self.feature_types[feature_name] = 'categorical'
                else:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
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
                    df_transformed, new_columns = bit_repr(pd.Series(leaf_indices, name = f'{feature_name}_leaf'), self.maps[feature_name], self.empty_cat)
                    #self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    df_transformed.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, df_transformed], axis = 1)
                    self.sizes[feature_name] = df_transformed.shape[1]
                else:
                    ohe = OneHotEncoder(categories = [all_cats], sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
                    encoded = ohe.fit_transform(leaf_indices.reshape(-1, 1))
                    new_columns = ohe.get_feature_names_out([feature_name])
                    self.no_interaction.append(set(new_columns))
                    transformed_X.reset_index(drop=True, inplace=True)
                    transformed_X = pd.concat([transformed_X, pd.DataFrame(encoded, columns=new_columns)], axis = 1)
                    self.sizes[feature_name] = len(new_columns)
            else:
                if self.bit:
                    df_transformed, new_columns = bit_repr(feature, self.maps[feature_name], self.empty_cat)
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
        # return transformed_X.astype(int).replace({-1:-1, 0:-1, 1:1})
        return transformed_X.astype(int).replace({-1:0, 0:0, 1:1})
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

class ClassFeaturizer(RegFeaturizer):
    def __init__(self, depth=2, bit=True, seed=0):
        super().__init__(depth, bit, seed)
        self.dt = DecisionTreeClassifier(max_depth=self.depth, random_state=seed)
        
class GMMBinaryMapper:
    def __init__(self, empty_cat=1, seed=0, max_gmm_components=3):
        self.encoders = {}
        self.maps = {}
        self.feature_types = {}
        self.no_interaction = []
        self.sizes = {}
        self.seed = seed
        self.empty_cat = empty_cat
        self.max_gmm_components = max_gmm_components
        
    def _plot_gmm_with_intersections(self, data, feature_name, gmm, intersections):
        x = np.linspace(data.min(), data.max(), 1000)
        pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.6, color='gray')
        plt.plot(x, pdf, '-k', lw=2)

        # Plot each Gaussian component
        for mean, cov in zip(gmm.means_, gmm.covariances_):
            component_pdf = (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-(x - mean)**2 / (2 * cov))
            plt.plot(x, component_pdf.reshape(-1, ), '--', lw=2)

        # Plot intersection lines
        for intersection in intersections:
            plt.axvline(x=intersection, color='r', linestyle='--')

        plt.title(f'GMM for feature: {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.show()
    
    def _fit_gmm_and_find_intersections(self, data, feature_name, plot):
        param_grid = {'n_components': np.arange(1, self.max_gmm_components+1)}
        gmm = GaussianMixture()
        grid_search = GridSearchCV(gmm, param_grid, cv=5)
        grid_search.fit(data.reshape(-1, 1))
        optimal_components = grid_search.best_params_['n_components']
        
        gmm = GaussianMixture(n_components=optimal_components)
        gmm.fit(data.reshape(-1, 1))
        
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        sorted_indices = np.argsort(means)
        sorted_means = means[sorted_indices]
        sorted_variances = variances[sorted_indices]
        
        intersections = []
        
        for i in range(len(sorted_means) - 1):
            mu1, var1 = sorted_means[i], sorted_variances[i]
            mu2, var2 = sorted_means[i + 1], sorted_variances[i + 1]

            term1 = mu1*var2 - mu2*var1
            term2 = np.sqrt(var1*var2) * np.sqrt((mu1 - mu2)**2 + (var2 - var1) * np.log(var2 / var1))
            term3 = var2 - var1
            intersections.append(((term1 + term2) / term3))
            
        if plot:
            self._plot_gmm_with_intersections(data, feature_name, gmm, intersections)
        
        return intersections
    
    def fit(self, X, plot=False):
        for feature_name in X.columns:
            feature = X[feature_name]
            if pd.api.types.is_float_dtype(feature) and len(feature.unique()) > 10:
                unique_vals = feature.unique()
                if len(unique_vals) == 2:
                    self.maps[feature_name] = binary_map(feature)
                    self.feature_types[feature_name] = 'binary'
                else:
                    intersections = self._fit_gmm_and_find_intersections(feature.values, feature_name, plot)
                    # mapping = defaultdict(lambda: 0)
                    # unique_values = sorted(np.unique(intersections))
                    # mapping.update({val: i + self.empty_cat for i, val in enumerate(unique_values)})
                    # self.maps[feature_name] = mapping
                    self.maps[feature_name] = intersections
                    self.feature_types[feature_name] = 'continuous'
            else:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
                encoded_feature = encoder.fit_transform(feature.values.reshape(-1, 1))
                self.encoders[feature_name] = encoder
                self.feature_types[feature_name] = 'categorical'
                
    def transform(self, X):
        transformed_features = []
        
        for feature_name in X.columns:
            feature = X[feature_name]
            if self.feature_types[feature_name] == 'binary':
                binary_map = self.maps[feature_name]
                transformed_feature = feature.map(binary_map).fillna(0)
                transformed_features.append(transformed_feature)
                
            elif self.feature_types[feature_name] == 'categorical':
                encoder = self.encoders[feature_name]
                transformed_feature = encoder.transform(feature.values.reshape(-1, 1))
                transformed_features.append(pd.DataFrame(transformed_feature, columns=encoder.get_feature_names_out([feature_name])))
            
            elif self.feature_types[feature_name] == 'continuous':
                intersections = self.maps[feature_name]
                regions = []
                
                # Create regions based on intersections
                for i in range(len(intersections) + 1):
                    if i == 0:
                        regions.append((feature <= intersections[i]).astype(int))
                    elif i == len(intersections):
                        regions.append((feature > intersections[i-1]).astype(int))
                    else:
                        regions.append(((feature > intersections[i-1]) & (feature <= intersections[i])).astype(int))
                
                for j, region in enumerate(regions):
                    transformed_features.append(pd.Series(region, name=f'{feature_name}_region{j+1}'))
        
        return pd.concat(transformed_features, axis=1)