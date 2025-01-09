import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, f1_score, precision_recall_curve
from scipy.special import softmax
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
import xgboost as xgb

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.preprocessing import StandardScaler
import joblib
import imodels
import inspect
import os.path
import sys
import psutil
import imodelsx.cache_save_utils
import time
import torch

sys.path.append('/home/mattyshen/iCBM')

from CUB.template_model import End2EndModel, Inception3, MLP

# from CUB.dataset import load_data
# from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, DEVICE, get_device, set_device
# from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_to_repo)

import idistill.model
import idistill.data
from idistill.ftd import FTDistillRegressorCV
from idistill.figs_distiller import FIGSRegressor

def fit_model(model, X_train, y_train, feature_names, r):
    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        model.fit(X_train, y_train, feature_names=feature_names)
    else:
        model.fit(X_train, y_train)

    return r, model

def evaluate_model(model, X_train, X_val, y_train, y_val, comp, seed, r):
    """Evaluate model performance on each split"""
    metrics = {
            "accuracy": accuracy_score,
        }
    for split_name, (X_, y_) in zip(
        ["trainval", "test"], [(X_train, y_train), (X_val, y_val)]
    ):
        y_pred_ = model.predict(X_)
        if len(y_pred_.shape) > 1 and y_pred_.shape[1] > 1:
            #handle regressors
            y_pred_ = np.argmax(y_pred_, axis=1)
        for i, (metric_name, metric_fn) in enumerate(metrics.items()):
            print(metric_fn(y_, y_pred_))
            r[f"{comp}_seed{seed}_{metric_name}_{split_name}"] = metric_fn(y_, y_pred_)

    return r

def load_csvs(path):

    X_train = pd.read_csv(f'{path}/X_trainval.csv', index_col=0)
    X_train_hat = pd.read_csv(f'{path}/X_trainval_hat.csv', index_col=0)
    X_test = pd.read_csv(f'{path}/X_test.csv', index_col=0)
    X_test_hat = pd.read_csv(f'{path}/X_test_hat.csv', index_col=0)
    y_train = pd.read_csv(f'{path}/y_trainval.csv', index_col=0)
    y_train_hat = pd.read_csv(f'{path}/y_trainval_hat.csv', index_col=0)
    y_test = pd.read_csv(f'{path}/y_test.csv', index_col=0)
    y_test_hat = pd.read_csv(f'{path}/y_test_hat.csv', index_col=0)

    return X_train, X_train_hat, X_test, X_test_hat, y_train, y_train_hat, y_test, y_test_hat

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def find_thresh(linkage_matrix, min_clusters=10, max_clusters=15, step=0.1, count = 0):
    if count > 3:
        return None, 0
    threshold = 1.5
    while threshold < 10:
        clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
        num_clusters = len(set(clusters))
        if min_clusters <= num_clusters <= max_clusters:
            return threshold, num_clusters
        threshold += step
    print('find_thresh recursive call beginning')
    return find_thresh(linkage_matrix, min_clusters=min_clusters, max_clusters=max_clusters+5, step=0.1, count = count+1)
    #return None, 0

def cluster_concepts(X, num_clusters):
    distance_matrix = 1 - X_train_hat.corr().abs()
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    threshold, _ = find_thresh(linkage_matrix, min_clusters=num_clusters-5, max_clusters=num_clusters, step=0.1)
        
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    
    feature_groups = {}
    for i, cluster_id in enumerate(clusters):
        feature_groups.setdefault(cluster_id, []).append(distance_matrix.columns[i])
    
    return feature_groups

def process_X(X_train, X_train_hat, X_test, X_test_hat, prepro, num_clusters, thresh=0):
    if prepro == "probs":
        return X_train_hat, X_test_hat
    elif prepro == 'cluster':
        f_gs = cluster_concepts(X_train_hat, num_clusters)
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int)
    elif prepro == 'global':
        opt_thresh = find_optimal_threshold(X_train.values.reshape(-1, ), X_train_hat.values.reshape(-1, ))
        
        return (X_train_hat > opt_thresh).astype(int), (X_test_hat > opt_thresh).astype(int)
        
    elif prepro == 'binary' and thresh > 0:
        return (X_train_hat > thresh).astype(int), (X_test_hat > thresh).astype(int),
    else:
        optimal_thresholds = []
        for class_idx in range(X_train_hat.shape[1]):
            y_true_class = X_train.iloc[:, class_idx]
            y_probs_class = X_train_hat.iloc[:, class_idx]
            optimal_thresholds.append(find_optimal_threshold(y_true_class, y_probs_class))
        optimal_thresholds = np.array(optimal_thresholds)
        
        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), 
    
def process_y(y_train, y_train_hat, y_test, y_test_hat, prepro):
    if prepro == "probs":
        return softmax(y_train_hat, axis=1), softmax(y_test_hat, axis=1)
    elif prepro == "classes":
        return pd.DataFrame(y_train_hat.idxmax(axis=1).astype(int)), pd.DataFrame(y_test_hat.idxmax(axis=1).astype(int))
    else:
        return y_train_hat, y_test_hat

# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--task_type", 
        type=str, 
        choices=["regression", "classification"],
        default="regression", 
        help="Type of task"
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        choices=["FIGSHydraRegressor", "FIGSRegressor", "XGBRegressor", "FTDHydraRegressor", "FTDHydraRegressorCV","FTDRegressorCV", "FIGSClassifier", "XGBClassifier", "FTDClassifierCV"],
        default="FIGSRegressor", 
        help="Model Name"
    )
    parser.add_argument(
        "--X_type", 
        type=str, 
        choices=["probs", "binary", "cluster", "global"],
        default="binary", 
        help="Type of X"
    )
    parser.add_argument(
        "--thresh", 
        type=float, 
        default=0, 
        help="Use thresh as binary threshold"
    )
    parser.add_argument(
        "--Y_type", 
        type=str, 
        choices=["probs", "classes", "logits"],
        default="logits", 
        help="Type of Y"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )

    parser.add_argument(
        "--max_rules", type=int, default=60, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=30, help="max trees of FIGS model"
    )
    parser.add_argument(
        "--max_depth", type=int, default=3, help="max depth of tree based models"
    )
    parser.add_argument(
        "--pre_interaction", 
        type=str,
        choices=["l0", "l0l2", "None"],
        default="l0l2", 
        help="type of feature selection in ft_distill model pre-interaction expansion"
    )
    parser.add_argument(
        "--pre_max_features", type=float, default=1, help="max fraction or max number of features allowed in pre-interaction with l0 based model"
    )
    parser.add_argument(
        "--post_interaction", 
        type=str,
        choices=["l0", "l0l2", "None"],
        default="l0l2", 
        help="type of feature selection in ft_distill model post-interaction expansion"
    )
    parser.add_argument(
        "--post_max_features", type=float, default=30, help="max frac or max number of features allowed in post-interaction with l0 based model"
    )
    parser.add_argument(
        "--mo", type=bool, default=False, help="multi-output parameter for FTDRegressorCV"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda:0",
        help="GPU device"
    )
    parser.add_argument(
        "--num_clusters", 
        type=int,
        default=15, 
        help="max number of clusters of concepts"
    )
    parser.add_argument(
        "--concepts_to_edit", 
        type=str,
        default='', 
        help="string of concepts to update of format 'X,Y,Z' for integers X, Y, Z"
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

    
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )
    
    for seed in range(1, 4):
        #ORIGINAL
        
        #load in tabular frozen predictions for distillation and logging original cbm accuracies
        X_train, X_train_hat, X_test, X_test_hat, y_train, y_train_hat, y_test, y_test_hat = load_csvs(f'/home/mattyshen/DistillationEdit/data/cub_tabular/seed0_Joint0.01SigmoidModel__Seed{seed}')
        X_train_model, X_test_model = process_X(X_train, X_train_hat, X_test, X_test_hat, args.X_type, args.num_clusters, args.thresh)
        y_train_model, y_test_model = process_y(y_train, y_train_hat, y_test, y_test_hat, args.Y_type)
        
        #log cbm prediction accuracies for train+val and test (using tabular frozen predictions)
        r[f"cbm_true_seed{seed}_accuracy_trainval"] = accuracy_score(y_train_hat.idxmax(axis=1).astype(int), y_train)
        r[f"cbm_true_seed{seed}_accuracy_test"] = accuracy_score(y_test_hat.idxmax(axis=1).astype(int), y_test)
        
        
        #create distiller model using tabular frozen predictions
        model = idistill.model.get_model(args.task_type, args.model_name, args)
        r, model = fit_model(model, X_train_model, y_train_model, None, r)
        
        #log distiller prediction accuracies for train+val and test
        print('distiller true')
        r = evaluate_model(model, X_train_model, X_test_model, y_train, y_test, "distiller_true", seed, r)
        print('distiller cbm')
        r = evaluate_model(model, X_train_model, X_test_model, y_train_hat.idxmax(axis=1).astype(int), y_test_hat.idxmax(axis=1).astype(int), "distiller_cbm", seed, r)
        
        cbm_train_mask = y_train_hat.idxmax(axis = 1).astype(int).to_numpy().reshape(-1, ) == y_train.values.reshape(-1, )
        cbm_test_mask = y_test_hat.idxmax(axis = 1).astype(int).to_numpy().reshape(-1, ) == y_test.values.reshape(-1, )
        
        distiller_train_mask = np.argmax(model.predict(X_train_model), axis=1) == y_train.values.reshape(-1, )
        distiller_test_mask = np.argmax(model.predict(X_test_model), axis=1) == y_test.values.reshape(-1, )
        
        r[f"%_correct_seed{seed}_overlap_trainval"] = np.mean(cbm_train_mask == distiller_train_mask)
        r[f"%_correct_seed{seed}_overlap_test"] = np.mean(cbm_test_mask == distiller_test_mask)
        
        #EDITED
        
        concepts_to_edit = list(map(int, args.concepts_to_edit.split(',')))
        
        #load model_seed in
        sys.path.append('/home/mattyshen/iCBM')
        sec_model = torch.load(f'/home/mattyshen/iCBM/CUB/best_models/Joint0.01SigmoidModel__Seed{seed}/outputs/best_model_{seed}.pth').sec_model
        sec_model.to(args.device)
        #sec_model = model.sec_model
        #model.eval()
        sec_model.eval()
        sys.path.append(path_to_repo)

        # sec_model = torch.load(f'/home/mattyshen/iCBM/CUB/best_models/Joint0.01SigmoidModel__Seed{seed}/outputs/best_model_{seed}.pth').sec_model
        # sec_model = sec_model.to(args.device)
        # sec_model.eval()
        
        #editing concept predictions for second half model of CBM
        X_train_hat.iloc[:, concepts_to_edit] = X_train.iloc[:, concepts_to_edit]
        X_test_hat.iloc[:, concepts_to_edit] = X_test.iloc[:, concepts_to_edit]
        #editing concept predictions for distiller model
        X_train_model.iloc[:, concepts_to_edit] = X_train.iloc[:, concepts_to_edit].astype(X_train_model.iloc[:, 0].dtype)
        X_test_model.iloc[:, concepts_to_edit] = X_test.iloc[:, concepts_to_edit].astype(X_test_model.iloc[:, 0].dtype)
        
        y_train_edited_cbm = sec_model(torch.tensor(X_train_hat.values, dtype=torch.float32).to(args.device))
        y_test_edited_cbm = sec_model(torch.tensor(X_test_hat.values, dtype=torch.float32).to(args.device))
        
        y_train_edited_cbm = pd.DataFrame(y_train_edited_cbm.detach().cpu().numpy())
        y_test_edited_cbm = pd.DataFrame(y_test_edited_cbm.detach().cpu().numpy())
        
        r[f"edited_cbm_true_seed{seed}_accuracy_trainval"] = accuracy_score(y_train_edited_cbm.idxmax(axis=1).astype(int), y_train)
        r[f"edited_cbm_true_seed{seed}_accuracy_test"] = accuracy_score(y_test_edited_cbm.idxmax(axis=1).astype(int), y_test)
        
        #log distiller prediction accuracies for train+val and test
        print('distiller edited true')
        r = evaluate_model(model, X_train_model, X_test_model, y_train, y_test, "edited_distiller_true", seed, r)
        print('distiller edited cbm')
        r = evaluate_model(model, X_train_model, X_test_model, y_train_edited_cbm.idxmax(axis=1).astype(int), y_test_edited_cbm.idxmax(axis=1).astype(int), "edited_distiller_cbm", seed, r)
        cbm_train_mask = y_train_edited_cbm.idxmax(axis = 1).astype(int).to_numpy().reshape(-1, ) == y_train.values.reshape(-1, )
        cbm_test_mask = y_test_edited_cbm.idxmax(axis = 1).astype(int).to_numpy().reshape(-1, ) == y_test.values.reshape(-1, )
        
        distiller_train_mask = np.argmax(model.predict(X_train_model), axis=1) == y_train.values.reshape(-1, )
        distiller_test_mask = np.argmax(model.predict(X_test_model), axis=1) == y_test.values.reshape(-1, )
        
        r[f"edited_%_correct_seed{seed}_overlap_trainval"] = np.mean(cbm_train_mask == distiller_train_mask)
        r[f"edited_%_correct_seed{seed}_overlap_test"] = np.mean(cbm_test_mask == distiller_test_mask)

        
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")