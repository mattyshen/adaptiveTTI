import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, KFold
import joblib
import imodels
import inspect
import os.path
import sys
import psutil
import imodelsx.cache_save_utils

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_to_repo)

import interpretDistill.model
import interpretDistill.data

def fit_model(model, X_train, y_train, feature_names, no_interaction, r):
    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        model.fit(X_train, y_train, feature_names=feature_names)
    elif "no_interaction" in fit_parameters and len(no_interaction) > 0:
        #ft_distill models
        model.fit(X_train, y_train, no_interaction=no_interaction)
    elif type(model) == imodels.importance.rf_plus.RandomForestPlusRegressor:
        model.fit(X_train, y_train.to_numpy())
    else:
        model.fit(X_train, y_train)

    return r, model

def fit_binary_mapper(binary_mapper, X_train, y_train, r):
    # fit the model
    fit_parameters = inspect.signature(binary_mapper.fit).parameters.keys()
    if "y" in fit_parameters and y_train is not None:
        binary_mapper.fit(X_train, y_train)
    else:
        binary_mapper.fit(X_train)

    return r, binary_mapper

def evaluate_model(model, model_name, comp, task, X_train, X_val, y_train, y_val, r):
    """Evaluate model performance on each split"""
    if task == 'regression':
        metrics = {
            "r2_score": r2_score,
        }
    else:
        metrics = {
            "accuracy": accuracy_score,
        }
    for split_name, (X_, y_) in zip(
        ["train", "val"], [(X_train, y_train), (X_val, y_val)]
    ):
        y_pred_ = model.predict(X_)
        for metric_name, metric_fn in metrics.items():
            r[f"{model_name}_{metric_name}_{split_name}_{comp}"] = metric_fn(y_, y_pred_)

    return r


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", 
                 "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"],
        default="ca_housing", 
        help="name of dataset"
    )
    parser.add_argument(
        "--subsample_frac", type=float, default=0.25, help="fraction of samples to use for val set"
    )

    # training misc args
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["random_forest", "rf_plus", "figs", "xgboost", "resnet", "ft_transformer", "ft_distill", "xgboost30"],
        default="ft_transformer",
        help="name of (teacher, if distillation) model",
    )
    parser.add_argument(
        "--binary_mapper_name", 
        type=str, 
        choices=["gmm_binary_mapper", "dt_binary_mapper", "no_binary_mapper"],
        default="no_binary_mapper", 
        help="type of binary_mapper to discretize dataset"
    )
    parser.add_argument(
        "--binary_mapper_bit", type=int, default=0, help="binary_mapper model to use bit representation"
    )
    parser.add_argument(
        "--binary_mapper_depth", type=int, default=2, help="depth of decision tree for binary_mapper model/2**bmd number of gaussian components in GMM"
    )
    parser.add_argument(
        "--binary_mapper_frac", type=float, default=0.5, help="fraction of train samples to fit binary_mapper"
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="max depth of tree based models (RF, XGB)")
    parser.add_argument(
        "--max_features", type=float, default=1, help="max features of tree based models (RF, XGB)")
    parser.add_argument(
        "--max_rules", type=int, default=60, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=30, help="max trees of FIGS model"
    )
    parser.add_argument(
        "--pre_interaction", 
        type=str,
        choices=["l0", "l0l2", "l1", "l1l2"],
        default="l0l2", 
        help="type of feature selection in ft_distill model pre-interaction expansion"
    )
    parser.add_argument(
        "--pre_max_features", type=float, default=0.5, help="max frac or max number of features allowed in pre-interaction with l0 based model"
    )
    parser.add_argument(
        "--post_interaction", 
        type=str,
        choices=["l0", "l0l2", "l1", "l1l2"],
        default="l0l2", 
        help="type of feature selection in ft_distill model post-interaction expansion"
    )
    parser.add_argument(
        "--post_max_features", type=float, default=30, help="max frac or max number of features allowed in post-interaction with l0 based model"
    )
    parser.add_argument(
        "--size_interactions", type=int, default=3, help="size of largest interactions for ft_distill models"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs for DL based models"
    )
    parser.add_argument(
        "--gpu", type=int, choices=[0, 1, 2, 3], default=2, help="gpu"
    )
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    X, y, args = interpretDistill.data.load_tabular_dataset(args.dataset_name, args)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.subsample_frac, random_state=0)
    
    # load tabular data
    # https://csinva.io/imodels/util/data_util.html#imodels.util.data_util.get_clean_dataset
    # X_train, X_test, y_train, y_test, feature_names = imodels.get_clean_dataset('compas_two_year_clean', data_source='imodels', test_size=0.33)

    scores_ = [[]]
    
    kf = KFold(n_splits=3, random_state=405, shuffle=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_out, y_out = X_train.iloc[test_index, :], y_train.iloc[test_index]
        X_in, y_in = X_train.iloc[train_index, :], y_train.iloc[train_index]
        
        
        binary_mapper_k = deepcopy(interpretDistill.model.get_model(args.task_type, args.binary_mapper_name, args))
        no_interaction = []
        
        if binary_mapper_k is not None:
            
            print('Using binary mapper')

            X_in_bm, X_in_bmt, y_in_bm, y_in_bmt = train_test_split(X_in, y_in, test_size=args.binary_mapper_frac, random_state=args.seed)

            _, binary_mapper_k = fit_binary_mapper(binary_mapper_k, X_in_bmt, y_in_bmt, 0)

            X_in_bmt = binary_mapper_k.transform(X_in_bmt)

            X_in_bm = binary_mapper_k.transform(X_in_bm)

            X_in = pd.concat([X_in_bmt, X_in_bm]).reset_index(drop=True)
            y_in = pd.concat([y_in_bmt, y_in_bm]).reset_index(drop=True)

            no_interaction = binary_mapper_k.no_interaction
    
        model_k = deepcopy(interpretDistill.model.get_model(args.task_type, args.model_name, args))
            
        feature_names = list(X_in.columns)

        _, model_k = fit_model(model_k, X_in, y_in, feature_names, no_interaction, 0)
        
        if binary_mapper_k is not None:
            X_out = binary_mapper_k.transform(X_out)

        scores_[0].append(r2_score(y_out, model_k.predict(X_out)))
    
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )
    r['teacher_avg_kfold_val_r2'] = np.mean(scores_[0])
    
    r['teacher_kfold_val_r2'] = scores_[0]
    
    binary_mapper = deepcopy(interpretDistill.model.get_model(args.task_type, args.binary_mapper_name, args))
    no_interaction = []
    
    if binary_mapper is not None:
    
        X_train_bm, X_train_bmt, y_train_bm, y_train_bmt = train_test_split(X_train, y_train, test_size=args.binary_mapper_frac, random_state=args.seed)

        r, binary_mapper = fit_binary_mapper(binary_mapper, X_train_bmt, y_train_bmt, r)

        X_train_bmt = binary_mapper.transform(X_train_bmt)

        X_train_bm = binary_mapper.transform(X_train_bm)

        X_train = pd.concat([X_train_bmt, X_train_bm]).reset_index(drop=True)
        y_train = pd.concat([y_train_bmt, y_train_bm]).reset_index(drop=True)

        no_interaction = binary_mapper.no_interaction
    
    model = deepcopy(interpretDistill.model.get_model(args.task_type, args.model_name, args))
    
    feature_names = list(X_train.columns)

    r, model_train = fit_model(model, X_train, y_train, feature_names, no_interaction, r)
    
    if binary_mapper is not None:
        X_val = binary_mapper.transform(X_val)
    
    r = evaluate_model(model, 'teacher', 'true', args.task_type, X_train, X_val, y_train, y_val, r)
    
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")