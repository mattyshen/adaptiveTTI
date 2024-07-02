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

from interpretDistill.binary_mapper_utils import binary_map, bit_repr, get_leaf_node_indices
from interpretDistill.continuous import is_continuous
# from binary_mapper_utils import binary_map, bit_repr, get_leaf_node_indices
# from continuous import is_continuous

class DTRegBinaryMapper:
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
            if is_continuous(feature):
                dt = clone(self.dt)
                dt.fit(feature.values.reshape(-1, 1), y)
                self.dt_models[feature_name] = dt
                if self.bit:
                    mapping = defaultdict(lambda: 0)
                    unique_values = sorted(np.unique(dt.apply(feature.values.reshape(-1, 1))))
                    mapping.update({val: i+self.empty_cat for i, val in enumerate(unique_values)})
                    self.maps[feature_name] = mapping
                else:
                    all_cats = get_leaf_node_indices(dt.tree_)
                    leaf_indices = dt.apply(feature.values.reshape(-1, 1))

                    encoder = OneHotEncoder(categories = [all_cats], sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
                    encoder.fit(leaf_indices.reshape(-1, 1))
                    self.no_interaction.append(set(encoder.get_feature_names_out(input_features=[f'{feature_name}_leaf'])))
                    self.encoders[feature_name] = encoder

                self.feature_types[feature_name] = 'continuous'
            else:
                unique_vals = feature.unique()
                if len(unique_vals) == 2:
                    self.maps[feature_name] = binary_map(feature)
                    self.feature_types[feature_name] = 'binary'
                else:
                    if self.bit:
                        unique_values = sorted(feature.unique())
                        mapping = defaultdict(lambda: 0)
                        mapping.update({val: i+self.empty_cat for i, val in enumerate(unique_values)})
                        self.maps[feature_name] = mapping
                    else:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
                        encoded_feature = encoder.fit_transform(feature.values.reshape(-1, 1))
                        self.no_interaction.append(set(encoder.get_feature_names_out(input_features=[feature.name])))
                        self.encoders[feature_name] = encoder
                    self.feature_types[feature_name] = 'categorical'
    
    def transform(self, X):
        assert set(self.feature_types.keys()) == set(X.columns), "X not compatible with the X BinaryTransformer was fitted on"
        idx = X.index
        transformed_features = []
        
        for i, feature_name in enumerate(X.columns):
            feature = X[feature_name]
            if self.feature_types[feature_name] == 'binary':
                transformed_features.append(X[feature_name].map(self.maps[feature_name]).reset_index(drop=True))
                self.sizes[feature_name] = 1
                
            elif self.feature_types[feature_name] == 'continuous':
                dt_model = self.dt_models[feature_name]
                leaf_indices = dt_model.apply(feature.values.reshape(-1, 1))
                if self.bit:
                    df_transformed, new_columns = bit_repr(pd.Series(leaf_indices, name = f'{feature_name}_leaf'), self.maps[feature_name], self.empty_cat)
                    transformed_features.append(df_transformed.reset_index(drop=True))
                    self.sizes[feature_name] = df_transformed.shape[1]
                else:
                    ohe = self.encoders[feature_name]
                    encoded = ohe.transform(leaf_indices.reshape(-1, 1))
                    new_columns = ohe.get_feature_names_out([feature_name])
                    transformed_features.append(pd.DataFrame(encoded, columns=new_columns).reset_index(drop=True))
                    self.sizes[feature_name] = len(new_columns)
                    
            else:
                if self.bit:
                    df_transformed, new_columns = bit_repr(feature, self.maps[feature_name], self.empty_cat)
                    transformed_features.append(df_transformed.reset_index(drop=True))
                    self.sizes[feature_name] = df_transformed.shape[1]
                else:
                    ohe = self.encoders[feature_name]
                    encoded = ohe.transform(feature.values.reshape(-1, 1))
                    new_columns = ohe.get_feature_names_out([feature_name])
                    transformed_features.append(pd.DataFrame(encoded, columns=new_columns).reset_index(drop=True))
                    self.sizes[feature_name] = len(new_columns)
                    
        return pd.concat(transformed_features, axis=1).set_index([idx]).astype(int).replace({-1:0, 0:0, 1:1})
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

class DTClassBinaryMapper(DTRegBinaryMapper):
    def __init__(self, depth=2, bit=True, seed=0):
        super().__init__(depth, bit, seed)
        self.dt = DecisionTreeClassifier(max_depth=self.depth, random_state=seed)
        
class GMMBinaryMapper:
    def __init__(self, empty_cat=1, seed=0, max_gmm_components=4):
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

        for mean, cov in zip(gmm.means_, gmm.covariances_):
            component_pdf = (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-(x - mean)**2 / (2 * cov))
            plt.plot(x, component_pdf.reshape(-1, ), '--', lw=2)

        for intersection in intersections:
            plt.axvline(x=intersection, color='r', linestyle='--')

        plt.title(f'GMM for feature: {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.show()
    
    def _fit_gmm_and_find_intersections(self, data, feature_name, plot):
        param_grid = {'n_components': np.arange(1, self.max_gmm_components+1)}
        gmm = GaussianMixture(random_state=self.seed)
        grid_search = GridSearchCV(gmm, param_grid, cv=3)
        grid_search.fit(data.reshape(-1, 1))
        optimal_components = grid_search.best_params_['n_components']
        
        gmm = GaussianMixture(n_components=optimal_components, random_state=self.seed)
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
    
    def fit(self, X, y=None, plot=False):
        for feature_name in X.columns:
            feature = X[feature_name]
            if is_continuous(feature):
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
                    self.no_interaction.append(set([f'{feature_name}_region{j+1}' for j in range(len(intersections) + 1)]))
                    self.feature_types[feature_name] = 'continuous'
            else:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore' if not self.empty_cat else 'infrequent_if_exist')
                encoded_feature = encoder.fit_transform(feature.values.reshape(-1, 1))
                self.no_interaction.append(set(encoder.get_feature_names_out(input_features=[feature.name])))
                self.encoders[feature_name] = encoder
                self.feature_types[feature_name] = 'categorical'
                
    def transform(self, X):
        idx = X.index
        transformed_features = []
        
        for feature_name in X.columns:
            feature = X[feature_name]
            if self.feature_types[feature_name] == 'binary':
                binary_map = self.maps[feature_name]
                transformed_feature = feature.map(binary_map).fillna(0)
                transformed_features.append(transformed_feature.reset_index(drop=True))
                
            elif self.feature_types[feature_name] == 'categorical':
                encoder = self.encoders[feature_name]
                transformed_feature = encoder.transform(feature.values.reshape(-1, 1))
                transformed_features.append(pd.DataFrame(transformed_feature, columns=encoder.get_feature_names_out([feature_name])).reset_index(drop=True))
            
            elif self.feature_types[feature_name] == 'continuous':
                intersections = self.maps[feature_name]
                regions = []
                if len(intersections) > 0:
                    for i in range(len(intersections) + 1):
                        if i == 0:
                            regions.append((feature <= intersections[i]).astype(int))
                        elif i == len(intersections):
                            regions.append((feature > intersections[i-1]).astype(int))
                        else:
                            regions.append(((feature > intersections[i-1]) & (feature <= intersections[i])).astype(int))
                    
                    for j, region in enumerate(regions):
                        transformed_features.append(pd.Series(region, name=f'{feature_name}_region{j+1}').reset_index(drop=True))
        
        
        return pd.concat(transformed_features, axis=1).set_index([idx]).astype(int).replace({-1:0, 0:0, 1:1})
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)