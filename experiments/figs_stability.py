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
from scipy.spatial.distance import squareform

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
        print(max_clusters)
        return find_thresh(linkage_matrix, min_clusters=min_clusters, max_clusters=(max_clusters-5*4)-1, step=step, count = 0)
    threshold = 4
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
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    threshold, _ = find_thresh(linkage_matrix, min_clusters=num_clusters-5, max_clusters=num_clusters, step=0.1)
        
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    
    feature_groups = {}
    for i, cluster_id in enumerate(clusters):
        feature_groups.setdefault(cluster_id, []).append(distance_matrix.columns[i])
    
    return feature_groups

def process_X(X_train, X_train_hat, X_test, X_test_hat, prepro, num_clusters, thresh=0):
    if prepro == "probs":
        return X_train_hat, X_test_hat, None
    elif prepro == 'cluster':
        f_gs = cluster_concepts(X_train_hat, num_clusters)
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
    elif prepro == 'global':
        f_gs = cluster_concepts(X_train_hat, num_clusters)
        opt_thresh = find_optimal_threshold(X_train.values.reshape(-1, ), X_train_hat.values.reshape(-1, ))
        
        return (X_train_hat > opt_thresh).astype(int), (X_test_hat > opt_thresh).astype(int), f_gs
    elif prepro == 'gpt1':
        f_gs = {1:['c'+str(i) for i in range(1, 5)]+['c'+str(i) for i in range(53, 55)]+['c'+str(i) for i in range(100, 104)],
                2:['c'+str(i) for i in range(5, 11)]+['c'+str(i) for i in range(110, 113)]+['c78'],
                3:['c'+str(i) for i in range(11, 17)]+['c'+str(i) for i in range(26, 32)]+['c'+str(i) for i in range(85, 88)]+['c'+str(i) for i in range(65, 71)]+['c'+str(i) for i in range(104, 110)],
                4:['c'+str(i) for i in range(17, 24)]+['c'+str(i) for i in range(40, 51)]+['c'+str(i) for i in range(24, 26)]+['c'+str(i) for i in range(71, 78)]+['c'+str(i) for i in range(60, 65)],
                5:['c'+str(i) for i in range(32, 38)]+['c'+str(i) for i in range(88, 91)],
                6:['c38', 'c39', 'c51','c52']+['c'+str(i) for i in range(55, 60)],
                7:['c'+str(i) for i in range(97, 100)],
                8:['c'+str(i) for i in range(91, 97)]+['c'+str(i) for i in range(79, 85)]
        }
        
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
    elif prepro == 'gpt2':
        f_gs = {1:['c'+str(i) for i in range(1, 5)]+['c'+str(i) for i in range(53, 55)]+['c32']+['c'+str(i) for i in range(78, 85)],
                2:['c'+str(i) for i in range(5, 11)]+['c'+str(i) for i in range(110, 113)]+['c'+str(i) for i in range(33, 38)]+['c'+str(i) for i in range(88, 91)],
                3:['c'+str(i) for i in range(91, 97)]+['c'+str(i) for i in range(11, 17)]+['c'+str(i) for i in range(26, 32)]+['c'+str(i) for i in range(85, 88)]+['c'+str(i) for i in range(17, 24)]+['c'+str(i) for i in range(60, 65)],
                4:['c'+str(i) for i in range(40, 51)]+['c24', 'c25']+['c'+str(i) for i in range(104, 110)]+['c'+str(i) for i in range(55, 60)]+['c'+str(i) for i in range(65, 78)],
                5:['c38', 'c39', 'c51','c52'],
                6:['c'+str(i) for i in range(100, 104)]+['c'+str(i) for i in range(97, 100)]
        }
        
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
    elif prepro == 'gpt3':
        f_gs = {1:['c'+str(i) for i in range(1, 5)]+['c53', 'c54']+['c'+str(i) for i in range(100, 104)],
                2:['c78', 'c32']+['c'+str(i) for i in range(5, 11)]+['c'+str(i) for i in range(110, 113)]+['c'+str(i) for i in range(88, 91)]+['c'+str(i) for i in range(33, 38)],
                3:['c'+str(i) for i in range(91, 97)]+['c'+str(i) for i in range(11, 24)]+['c'+str(i) for i in range(26, 32)]+['c'+str(i) for i in range(85, 88)]+['c'+str(i) for i in range(55, 78)]+['c'+str(i) for i in range(104, 110)],
                4:['c'+str(i) for i in range(40, 51)]+['c24', 'c25', 'c38', 'c39', 'c51', 'c52'],
                5:['c'+str(i) for i in range(79, 85)],
                6:['c'+str(i) for i in range(97, 100)]
        }
        
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
    
    elif prepro == 'gpt4':
        f_gs = {1:[1,3,5,7,8,11,12,15,16,19,20,21,22,24,26,27,40,41,42,43,46,49,50,53,55,56,60,62,64,65,66,69,70,71,72,77,78,79,80,83,84,85,87,88,90,91,92,93,94,95,102,106,107,108,110],
                2:[4,6,10,13,14,17,18,23,25,29,54,73,76,86,89,97,98,104,105,111],
                3:[2,9,28,30,32,33,34,35,36,37,38,39,47,48,51,52,81,96,99],
                4:[31,44,45,57,58,59,61,63,67,68,74,75,82,100,101,103,109,112]
        }
        
        for k in f_gs.keys():
            f_gs[k] = ['c'+str(i) for i in f_gs[k]]
        
        optimal_thresholds = np.zeros(X_train.shape[1])
        
        for k in f_gs.keys():
            idxs = [int(s[1:]) - 1 for s in f_gs[k]]
            optimal_thresholds[idxs] = find_optimal_threshold(X_train[f_gs[k]].values.reshape(-1, ), X_train_hat[f_gs[k]].values.reshape(-1, ))

        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
        
    elif prepro == 'binary' and thresh > 0:
        f_gs = cluster_concepts(X_train_hat, num_clusters)
        return (X_train_hat > thresh).astype(int), (X_test_hat > thresh).astype(int), f_gs
    else:
        f_gs = cluster_concepts(X_train_hat, num_clusters)
        optimal_thresholds = []
        for class_idx in range(X_train_hat.shape[1]):
            y_true_class = X_train.iloc[:, class_idx]
            y_probs_class = X_train_hat.iloc[:, class_idx]
            optimal_thresholds.append(find_optimal_threshold(y_true_class, y_probs_class))
        optimal_thresholds = np.array(optimal_thresholds)
        
        return (X_train_hat > optimal_thresholds).astype(int), (X_test_hat > optimal_thresholds).astype(int), f_gs
    
def process_y(y_train, y_train_hat, y_test, y_test_hat, prepro):
    if prepro == "probs":
        return softmax(y_train_hat, axis=1), softmax(y_test_hat, axis=1)
    elif prepro == "classes":
        return pd.DataFrame(y_train_hat.idxmax(axis=1).astype(int)), pd.DataFrame(y_test_hat.idxmax(axis=1).astype(int))
    else:
        return y_train_hat, y_test_hat
    
def extract_interactions(model):
    """
    Extracts all feature interactions from the FIGS model by parsing through each additive tree.

    Parameters:
        model: A FIGS model containing an attribute `trees_`.
               Each tree is comprised of hierarchically linked `Node` objects.

    Returns:
        interactions: A list of sets, where each set contains the features involved in an interaction.
    """
    interactions = []

    def traverse_tree(node, current_features, current_depth):
        """
        Recursively traverse a tree to collect feature interactions.

        Parameters:
            node: The current `Node` object in the tree.
            current_features: A set of features encountered so far in the current path.
        """
        if node.left is None and node.right is None:
            interactions.append(current_features)
            return

        # Add the current feature to the set of features for this path
        current_features.add('c' + str(node.feature+1))

        # If the node has children, traverse them
        if node.left is not None:
            traverse_tree(node.left, current_features.copy(), current_depth=current_depth+1)
        if node.right is not None:
            traverse_tree(node.right, current_features.copy(), current_depth=current_depth+1)

    # Loop through each tree in the model
    # traverse_tree(model.trees_[0], set(), current_depth=0)
    # return interactions
    for tree in model.trees_:
        # Start traversal for each tree
        traverse_tree(tree, set(), current_depth=0)

    return interactions

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
        choices=["probs", "binary", "cluster", "global", "gpt1", "gpt2", "gpt3", "gpt4"],
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
        "--num_clusters", 
        type=int,
        default=5, 
        help="max number of clusters of concepts"
    )
    parser.add_argument(
        "--model_seed", 
        type=int,
        choices=[1, 2, 3],
        default=1, 
        help="seed of mdoel predictions to bootstrap"
    )
    parser.add_argument(
        "--num_bootstraps", 
        type=int,
        default=20, 
        help="number of bootstraps"
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

    all_interactions = {}

    for i in range(args.num_bootstraps):
        print(f'bootstrap: {i}')
        np.random.seed(i)
        X_train, X_train_hat, X_test, X_test_hat, y_train, y_train_hat, y_test, y_test_hat = load_csvs(f'/home/mattyshen/DistillationEdit/data/cub_tabular/seed0_Joint0.01SigmoidModel__Seed{args.model_seed}')
        X_train_model, X_test_model, clusters = process_X(X_train, X_train_hat, X_test, X_test_hat, args.X_type, args.num_clusters, args.thresh)
        y_train_model, y_test_model = process_y(y_train, y_train_hat, y_test, y_test_hat, args.Y_type)

        cur_bootstrap = pd.concat([X_train_model, y_train_model], axis = 1).sample(X_train_model.shape[0], replace=True)

        X_bs = cur_bootstrap.iloc[:, np.arange(0, X_train_model.shape[1])]
        y_bs = cur_bootstrap.iloc[:, np.arange(X_train_model.shape[1], X_train_model.shape[1]+y_train_model.shape[1])]

        model = idistill.model.get_model(args.task_type, args.model_name, args)
        r, model = fit_model(model, X_bs, y_bs, None, r)

        cur_interactions = extract_interactions(model)
        cur_interactions = list(set(frozenset(item) for item in cur_interactions))

        for inter in cur_interactions:
            if inter not in all_interactions.keys():
                all_interactions[inter] = 1
            else:
                all_interactions[inter] += 1
                
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open(f'/home/mattyshen/DistillationEdit/results/figs_stability/stafigs_seed{args.model_seed}_nbootstraps{args.num_bootstraps}.p', 'wb') as fp:
        pickle.dump(all_interactions, fp, protocol=pickle.HIGHEST_PROTOCOL)

        
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")