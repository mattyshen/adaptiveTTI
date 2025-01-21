import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
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

### TODO: fill in import statements ###

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_to_repo)

import idistill.model
import idistill.data
from idistill.ftd import FTDistillRegressorCV
from idistill.figs_distiller import FIGSRegressor

def distill_model(distiller, X_train_teacher, X_train_teacher, r, feature_names = None):
    """Distill the teacher model using the distiller model"""
    
    fit_parameters = inspect.signature(distiller.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        distiller.fit(X_train_teacher, X_train_teacher, feature_names=feature_names)
    else:
        distiller.fit(X_train_teacher, X_train_teacher)

    return r, model

def predict_distiller(distiller, X):
    ### TODO: handle distiller prediction outputs to match metrics ###

    return y_pred

def predict_teacher(teacher, X):
    ### TODO: handle teacher prediction outputs ###

    return y_pred

def evaluate_distiller(distiller, X_train, X_test, y_train, y_test, metric, task, r):
    """Evaluate distiller performance on each split"""
    
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    for split_name, (X_, y_) in zip(
        ["train", "test"], [(X_train, y_train), (X_test, y_test)]
    ):
        y_pred_ = predict_distiller(distiller, X_)
        r[f"distiller_{task}_{split_name}_{metric}"] = metric_fn(y_, y_pred_)

    return r

def evaluate_teacher(y_train_teacher, y_test_teacher, y_train, y_test, metric, task, r):
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    for split_name, (y_teacher_, y_) in zip(
        ["train", "test"], [(y_train_teacher, y_train), (y_test_teacher, y_test)]
    ):
        r[f"teacher_{task}_{split_name}_{metric}"] = metric_fn(y_teacher_, y_)
    
    return r

def load_teacher_model(model_path):
    ### TODO: load in teacher model using model_path ###
    
    return model

def generate_tabular_distillation_data(model, train_path, test_path):
    ### TODO: generate teacher train and test data using model, train_path, and test_path ###
    
    return X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher, y_train, y_test
    
def process_distillation_data(X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher):
    ### TODO: process (i.e. binarize) data for distillation ###
    
    return X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher

def process_teacher_eval(y_teacher):
    ### TODO: process teacher model predictions for evaluations (sometimes we distill a teacher model using a regressor, but want to evaluate class prediction accuracy) ###
    
    return y_teacher_eval



# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=join(path_to_repo, "models"),
        help="path to teacher model",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=join(path_to_repo, "data"),
        help="path to training data",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=join(path_to_repo, "data"),
        help="path to training data",
    )
    parser.add_argument(
        "--task_type", 
        type=str, 
        choices=["regression", "classification"],
        default="regression", 
        help="type of task"
    )
    parser.add_argument(
        "--distiller_name", 
        type=str,
        choices=["FIGSRegressor", "FIGSClassifier", "XGBRegressor", "XGBClassifier"],
        default="FIGSRegressor", 
        help="distiller name"
    )
    parser.add_argument(
        "--max_rules", type=int, default=100, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=20, help="max trees of FIGS model"
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="max depth of tree based models"
    )
    parser.add_argument(
        "--metric", type=str, default="accuracy", help="metric to log distillation and prediction performance"
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
    
    model = load_teacher_model(args.model_path)
    
    X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher, y_train, y_test = generate_tabular_distillation_data(model, args.train_path, args.test_path)
    
    X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher = process_distillation_data(X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher)
    
    y_train_teacher_eval = process_teacher_eval(y_train_teacher)
    y_test_teacher_eval = process_teacher_eval(y_test_teacher)
    
    model = idistill.model.get_model(args.task_type, args.distiller_name, args)
    
    figs_distiller = distill_model(figs_distiller, X_train_teacher, y_train_teacher, r)
    
    r = evaluate_distiller(figs_distiller, X_train_teacher, X_test_teacher, y_train_teacher_eval, y_test_teacher_eval, args.metric, "distillation", r)
    r = evaluate_distiller(figs_distiller, X_train, X_test, y_train, y_test, args.metric, "prediction", r)
    
    r = evaluate_teacher(y_train_teacher_eval, y_test_teacher_eval, y_train, y_test, args.metric, "prediction", r)
    
    ### TODO: FIGS based concept editing ###
        
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")