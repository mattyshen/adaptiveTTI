import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from copy import deepcopy

import sklearn.datasets
import openml
from ucimlrepo import fetch_ucirepo 


def load_tabular_dataset(dataset_name, args):
    if dataset_name == 'ca_housing':
        dataset = sklearn.datasets.fetch_california_housing(as_frame = True)
        X = dataset["data"]
        y = dataset["target"]
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping = {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
        
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'abalone':
        abalone = fetch_ucirepo(id=1) 
        X = abalone.data.features
        y = abalone.data.targets.Rings
        y = pd.Series(y, name = 'Rings')
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping = {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'parkinsons':
        parkinsons_telemonitoring = fetch_ucirepo(id=189) 
        X = parkinsons_telemonitoring.data.features.drop(columns = "test_time")
        y = parkinsons_telemonitoring.data.targets.total_UPDRS
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'airfoil':
        airfoil_self_noise = fetch_ucirepo(id=291) 
        X = airfoil_self_noise.data.features 
        y = airfoil_self_noise.data.targets["scaled-sound-pressure"]
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'cpu_act':
        computer = openml.datasets.get_dataset(197)
        X, y, _, _ = computer.get_data(target=computer.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'concrete':
        concrete_compressive_strength = fetch_ucirepo(id=165)  
        X = concrete_compressive_strength.data.features 
        y = concrete_compressive_strength.data.targets["Concrete compressive strength"]
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'powerplant':
        combined_cycle_power_plant = fetch_ucirepo(id=294) 
        X = combined_cycle_power_plant.data.features 
        y = combined_cycle_power_plant.data.targets.PE
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'miami_housing':
        miami_housing = openml.datasets.get_dataset(43093)
        X, y, _, _ = miami_housing.get_data(target=miami_housing.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
         
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'traffic':
        #TODO
        X, y = None, None
    elif dataset_name == 'insurance':
        insurance = pd.read_csv("https://raw.githubusercontent.com/pycaret/datasets/main/data/common/insurance.csv")
        X = insurance.drop(columns="charges")
        y = insurance.charges
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'qsar':
        #high D
        qsar = openml.datasets.get_dataset(4048)
        X, y, _, _ = qsar.get_data(target=qsar.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'allstate':   
        #high D
        allstate = openml.datasets.get_dataset(42571)
        X, y, _, _ = allstate.get_data(target=allstate.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'mercedes':
        #high D
        mercedes = openml.datasets.get_dataset(42570)
        X, y, _, _ = mercedes.get_data(target=mercedes.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'transaction':
        transaction = openml.datasets.get_dataset(42572)
        X, y, _, _ = transaction.get_data(target=transaction.default_target_attribute, dataset_format="dataframe")
        
        is_continuous = X.apply(lambda col: pd.api.types.is_float_dtype(col) and len(col.unique()) > 20)
        X_cat = X.loc[:, ~is_continuous]
        cats = list(X_cat.columns)
        cat_mappings = {}
        for c in cats:
            unique_values = sorted(X_cat[c].unique())
            mapping= {val: i for i, val in enumerate(unique_values)}
            X[c] = X_cat[c].map(mapping)
            cat_mappings[c] = {str(val): i for i, val in enumerate(unique_values)}
            
        args.cat_mappings = cat_mappings
        args.task_type = 'regression'
    elif dataset_name == 'fMRI':
        return None
    elif dataset_name == 'ccle':
        return None
    elif dataset_name == 'enhancer':
        return None
    else:
        return None
    return X.copy(deep=True), y.copy(deep=True), args

def load_huggingface_dataset(dataset_name, subsample_frac=1.0):
    """Load text dataset from huggingface (with train/validation spltis) + return the relevant dataset key
    """
    # load dset
    if dataset_name == 'tweet_eval':
        dset = datasets.load_dataset('tweet_eval', 'hate')
    elif dataset_name == 'financial_phrasebank':
        train = datasets.load_dataset('financial_phrasebank', 'sentences_75agree',
                                      revision='main', split='train')
        idxs_train, idxs_val = train_test_split(
            np.arange(len(train)), test_size=0.33, random_state=13)
        dset = datasets.DatasetDict()
        dset['train'] = train.select(idxs_train)
        dset['validation'] = train.select(idxs_val)
    else:
        dset = datasets.load_dataset(dataset_name)

    # process dset
    dataset_key_text = 'text'
    if dataset_name == 'sst2':
        dataset_key_text = 'sentence'
    elif dataset_name == 'financial_phrasebank':
        dataset_key_text = 'sentence'
    elif dataset_name == 'imdb':
        del dset['unsupervised']
        dset['validation'] = dset['test']

    # subsample datak
    if subsample_frac > 0:
        n = len(dset['train'])
        dset['train'] = dset['train'].select(np.random.choice(
            range(n), replace=False,
            size=int(n * subsample_frac)
        ))
    return dset, dataset_key_text

def convert_text_data_to_counts_array(dset, dataset_key_text):
    v = CountVectorizer()
    X = v.fit_transform(dset['train'][dataset_key_text])
    y_train = dset['train']['label']
    X_test = v.transform(dset['validation'][dataset_key_text])
    y_test = dset['validation']['label']
    feature_names = v.get_feature_names_out().tolist()
    return X, X_test, y_train, y_test, feature_names