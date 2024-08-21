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
from sklearn.linear_model import RidgeCV
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
import joblib
import imodels
import inspect
import os.path
import sys
import psutil
import imodelsx.cache_save_utils
import time

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_to_repo)

import interpretDistill.model
import interpretDistill.data
from interpretDistill.continuous import is_continuous
from interpretDistill.fourierDistill import FTDistillRegressorCV
from interpretDistill.figs_d import FIGSRegressor

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
    if "y" in fit_parameters and y_train is not None and "train" not in fit_parameters:
        binary_mapper.fit(X_train, y_train)
    elif "train" in fit_parameters:
        binary_mapper.fit(X=X_train, y=y_train, train=True)
    else:
        binary_mapper.fit(X_train)

    return r, binary_mapper

def evaluate_model(model, model_name, task, X_train, X_val, y_train, y_val, r):
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
            r[f"{model_name}_{metric_name}_{split_name}"] = metric_fn(y_, y_pred_)

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
    #FIGS model automatically inside FIGS Binary Mapper
    # parser.add_argument(
    #     "--figs_name",
    #     type=str,
    #     choices=["figs", "figscv"],
    #     default="figs",
    #     help="name of figs model",
    # )
    # parser.add_argument(
    #     "--ftd_name",
    #     type=str,
    #     choices=["ft_distill"],
    #     default="ft_distill",
    #     help="name of ftd model",
    # )
    parser.add_argument(
        "--binary_mapper_name", type=str, default="figs_binary_mapper"
    )
    parser.add_argument(
        "--binary_mapper_frac", type=float, default=0.5, help="fraction of train samples to fit binary_mapper"
    )
    parser.add_argument(
        "--max_rules", type=int, default=60, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=30, help="max trees of FIGS model"
    )
    parser.add_argument(
        "--max_features", type=float, default=1, help="max features of tree based models (RF, XGB)")
    parser.add_argument(
        "--max_depth", type=int, default=6, help="max depth of XGBoost, RF, RF+ model"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="num estimators of XGBoost, RF, RF+ model"
    )
    # parser.add_argument(
    #     "--pre_interaction", 
    #     type=str,
    #     choices=["l0", "l0l2", "l1", "l1l2"],
    #     default="l0l2", 
    #     help="type of feature selection in ft_distill model pre-interaction expansion"
    # )
    parser.add_argument(
        "--pre_max_features", type=float, default=1, help="max frac or max number of features allowed in pre-interaction with l0 based model"
    )
    parser.add_argument(
        "--max_min_interaction_size", type=int, default=3, help="max min interaction size"
    )
    # parser.add_argument(
    #     "--post_interaction", 
    #     type=str,
    #     choices=["l0", "l0l2", "l1", "l1l2"],
    #     default="l0l2", 
    #     help="type of feature selection in ft_distill model post-interaction expansion"
    # )
    # parser.add_argument(
    #     "--post_max_features", type=float, default=30, help="max frac or max number of features allowed in post-interaction with l0 based model"
    # )
    # parser.add_argument(
    #     "--size_interactions", type=int, default=3, help="size of largest interactions for ft_distill models"
    # )
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

    scores_ = [[], []] #figs, ftd
    
    kf = KFold(n_splits=3, random_state=405, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_out, y_out = X_train.iloc[test_index, :], y_train.iloc[test_index]
        X_in, y_in = X_train.iloc[train_index, :], y_train.iloc[train_index]
        
        
        binary_mapper_k = deepcopy(interpretDistill.model.get_model(args.task_type, args.binary_mapper_name, args))
        no_interaction = []
        
        if args.binary_mapper_frac < 0.01:
            _, binary_mapper_k = fit_binary_mapper(binary_mapper_k, X_in, y_in, 0)
            X_in_figs = binary_mapper_k.transform(X_in)
            
        else:

            X_in_bm, X_in_bmt, y_in_bm, y_in_bmt = train_test_split(X_in, y_in, test_size=args.binary_mapper_frac, random_state=args.seed)

            _, binary_mapper_k = fit_binary_mapper(binary_mapper_k, X_in_bmt, y_in_bmt, 0)

            X_in_bmt = binary_mapper_k.transform(X_in_bmt)

            X_in_bm = binary_mapper_k.transform(X_in_bm)

            X_in_figs = pd.concat([X_in_bmt, X_in_bm]) #.reset_index(drop=True)
            y_in = pd.concat([y_in_bmt, y_in_bm]) #.reset_index(drop=True)

        no_interaction = binary_mapper_k.no_interaction
        
        si = min(binary_mapper_k.max_interaction_size, args.max_min_interaction_size)
        model_k = FTDistillRegressorCV(pre_max_features=args.pre_max_features,
                                       post_max_features=binary_mapper_k.num_interactions, 
                                       size_interactions = si, 
                                       re_fit_alpha = 10**np.linspace(-3, 3, 100)) #deepcopy(interpretDistill.model.get_model(args.task_type, args.model_name, args))

        _, model_k = fit_model(model_k, X_in_figs, y_in, list(X_in_figs.columns), no_interaction, 0)
        
        X_out_figs = binary_mapper_k.transform(X_out)

        scores_[0].append(r2_score(y_out, binary_mapper_k.predict(X_out)))
        scores_[1].append(r2_score(y_out, model_k.predict(X_out_figs)))
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )
    r['figs_kfold_val_r2'] = scores_[0]
    r['ftd_kfold_val_r2'] = scores_[1]
    r['figs_avg_kfold_val_r2'] = np.mean(scores_[0])
    r['ftd_avg_kfold_val_r2'] = np.mean(scores_[1])
    
    binary_mapper = deepcopy(interpretDistill.model.get_model(args.task_type, args.binary_mapper_name, args))
    no_interaction = []
    
    if args.binary_mapper_frac < 0.01:
        _, binary_mapper = fit_binary_mapper(binary_mapper, X_train, y_train, r)
        X_train_figs_decoup = binary_mapper.transform(X_train)
        X_train_figs_inter = binary_mapper.transform_figs_inter(X_train_figs_decoup)
            
    else:
        X_train_bm, X_train_bmt, y_train_bm, y_train_bmt = train_test_split(X_train, y_train, test_size=args.binary_mapper_frac, random_state=args.seed)

        _, binary_mapper = fit_binary_mapper(binary_mapper, X_train_bmt, y_train_bmt, r)

        X_train_bmt = binary_mapper.transform(X_train_bmt)

        X_train_bm = binary_mapper.transform(X_train_bm)

        X_train_figs_decoup = pd.concat([X_train_bmt, X_train_bm], axis = 0) #.reset_index(drop=True)
        X_train_figs_inter = binary_mapper.transform_figs_inter(X_train_figs_decoup)
        
        y_train = pd.concat([y_train_bmt, y_train_bm], axis = 0) #.reset_index(drop=True)

    no_interaction = binary_mapper.no_interaction
    
    si = min(binary_mapper.max_interaction_size, args.max_min_interaction_size)

    model = FTDistillRegressorCV(pre_max_features=args.pre_max_features,
                               post_max_features=binary_mapper.num_interactions, 
                               size_interactions = si,
                               re_fit_alpha = 10**np.linspace(-3, 3, 100))
    ftd_start_time = time.time()
    r, model_train = fit_model(model, X_train_figs_decoup, y_train, list(X_train_figs_decoup.columns), no_interaction, r)
    ftd_training_time = time.time() - ftd_start_time
    
    ridge_figs_decoup = RidgeCV(alphas = 10**np.linspace(-3, 3, 100))
    ridge_figs_inter = RidgeCV(alphas = 10**np.linspace(-3, 3, 100))
    xgb = xgb.XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
    
    ridge_figs_decoup.fit(X_train_figs_decoup, y_train)
    ridge_figs_inter.fit(X_train_figs_inter, y_train)
    xgb.fit(X_train, y_train)
    
    
    
    
    # figs = FIGSRegressor(max_trees = args.max_trees, max_rules = args.max_rules)
    # figs.fit(X_train, y_train)
    # print(f'figs train r2: {r2_score(y_train, figs.predict(X_train))}')
    # print(f'figs val r2: {r2_score(y_val, figs.predict(X_val))}')
    
    # ss = StandardScaler()
    # continuous_mask = X_train.apply(lambda col: is_continuous(col))
    # X_train_cat = X_train.loc[:, ~continuous_mask]
    # X_train_cts = X_train.loc[:, continuous_mask]
    # cts_feats = X_train_cts.columns
    # ss.fit(X_train_cts)
    # X_train_cts = pd.DataFrame(ss.transform(X_train_cts), columns = cts_feats, index = X_train_cts.index)
    # X_train_cts = pd.DataFrame(X_train_cts, columns = cts_feats, index = X_train_cts.index)
    # rulefit = RidgeCV(alphas = 10**np.linspace(-3, 3, 100))
    # X_train_rulefit = pd.concat([X_train_cts, X_train_cat, X_train_figs], axis = 1)
    # print(X_train_rulefit)
    # rulefit.fit(X_train_rulefit, y_train)
    
    X_val_figs_decoup = binary_mapper.transform(X_val)
    X_val_figs_inter = binary_mapper.transform_figs_inter(X_val_figs_decoup)
    
    r = evaluate_model(model, 'ftd', args.task_type, X_train_figs_decoup, X_val_figs_decoup, y_train, y_val, r)
    r = evaluate_model(binary_mapper, 'figs', args.task_type, X_train, X_val, y_train, y_val, r)
    r = evaluate_model(ridge_figs_decoup, 'ridge_figs_decoup', args.task_type, X_train_figs_decoup, X_val_figs_decoup, y_train, y_val, r)
    r = evaluate_model(ridge_figs_inter, 'ridge_figs_inter', args.task_type, X_train_figs_inter, X_val_figs_inter, y_train, y_val, r)
    r = evaluate_model(xgb, 'xgboost', args.task_type, X_train, X_val, y_train, y_val, r)
    
#     X_val_cat = X_val.loc[:, ~continuous_mask]
#     X_val_cts = X_val.loc[:, continuous_mask]
#     #X_val_cts = pd.DataFrame(ss.transform(X_val_cts), columns = cts_feats, index = X_val_cts.index)
    
#     X_val_rulefit = pd.concat([X_val_cts, X_val_cat, X_val_figs], axis = 1)
#     print(X_val_rulefit)
#     r = evaluate_model(rulefit, 'rulefit', args.task_type, X_train_rulefit, X_val_rulefit, y_train, y_val, r)

    figs_interactions = sorted([tuple(inter) for inter, weight in binary_mapper.interactions], key = len)
    ftd_interactions = model.post_interaction_features

    list1 = [set(fp) for fp in figs_interactions]
    list2 = [set(fp) for fp in ftd_interactions]
    frozensets_list1 = set(frozenset(s) for s in list1)
    frozensets_list2 = set(frozenset(s) for s in list2)

    common_frozensets = frozensets_list1.intersection(frozensets_list2)

    common_count = len(common_frozensets)

    r['num_common_interactions'] = common_count
    r['num_interactions'] = binary_mapper.num_interactions
    r['figs_max_interaction_size'] = binary_mapper.max_interaction_size
    r['ftd_max_interaction_size'] = min(binary_mapper.max_interaction_size, 5)
    r['figs_training_time'] = binary_mapper.figs_training_time
    r['ftd_training_time'] = ftd_training_time
    
    print(f'num common interactions found: {common_count}')
    print(f'common interactions found: {common_frozensets}')
    print(f'figs: {frozensets_list1}')
    print(f'ftd: {frozensets_list2}')

    
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")