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

def distill_model(student, X_train_teacher, y_train_teacher, r, feature_names = None):
    """Distill the teacher model using the student model
    
        Paramaters: 
            student: student model
            X_train_teacher (n_train, n_concepts): teacher model's predicted concept outputs (logits, probabilties, etc) for training data
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs (logits, probabilities, etc) for training data
            r: default dictionary to log experiment metrics
            feature_names: feature names of X_train_teacher
            
        Returns:
            r: default dictionary to log experiment metrics
            student: trained/distilled student model
    
    """
    
    fit_parameters = inspect.signature(student.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        student.fit(X_train_teacher, y_train_teacher, feature_names=feature_names)
    else:
        student.fit(X_train_teacher, y_train_teacher)

    return r, student

def evaluate_student(student, X_train, X_test, y_train, y_test, metric, task, r):
    """Evaluate student performance on each split
    
        Paramaters: 
            student: student model
            X_train (n_train, n_concepts): teacher model's predicted concept outputs (logits, probabilties, etc) for training data
            X_test (n_test, n_concepts): teacher model's predicted concept outputs (logits, probabilties, etc) for test data
            y_train (n_train, 1): teacher model's predicted task outputs in evaluation form (i.e. if classification task, y_train_teacher must be class predictions, not class logits) OR true outputs for train data
            y_test (n_test, 1): teacher model's predicted task outputs in evaluation form OR true outputs for test data
            metric: metric to log 
            task: task to log (i.e. if evaluating distillation performance, task would be `distillation`)
            r: default dictionary to log experiment metrics
            
        Returns:
            r: default dictionary to log experiment metrics
            
    """
    
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
        y_pred_ = process_student_eval(student.predict(X_))
        r[f"student_{task}_{split_name}_{metric}"] = metric_fn(y_, y_pred_)

    return r

def evaluate_teacher(y_train_teacher, y_test_teacher, y_train, y_test, metric, task, r):
    """Evaluate teacher performance on each split
    
        Paramaters: 
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs in evaluation form (i.e. if classification task, y_train_teacher must be class predictions, not class logits) for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs in evaluation form for test data
            y_train (n_train, 1): true outputs for train data
            y_test (n_test, 1): true outputs for test data
            metric: metric to log 
            task: type of distillation task (likely regression)
            r: default dictionary to log experiment metrics
            
        Returns:
            r: default dictionary to log experiment metrics
            
    """
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

def evaluate_test_student(student, X_test, y_test, metric, task, r):
    """Evaluate student performance on each split"""
    
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    y_pred_ = process_student_eval(student.predict(X_test))
    r[f"student_{task}_test_{metric}"] = metric_fn(y_test, y_pred_)

    return r

def evaluate_test_teacher(y_test_teacher, y_test, metric, task, r):
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    r[f"teacher_{task}_test_{metric}"] = metric_fn(y_test_teacher, y_test)
    
    return r

def predict_teacher(teacher, X, gpu=0):
    """Make prediction from concepts to outputs with teacher model
    
        Paramaters: 
            teacher: teacher model
            X (n, n_concepts): concept data
            gpu: gpu cuda device if applicable
            
        Returns:
            y_pred: teacher model predictions for X 
            
    """
    ### TODO: handle teacher prediction outputs (X is intended to be concept design matrix)###

    return y_pred

def load_teacher_model(teacher_path, gpu=0):
    """Load in teacher model
    
        Paramaters: 
            teacher_path: path where teacher model is stored
            gpu: gpu cuda device if applicable
            
        Returns:
            model: teacher model
            
    """
    ### TODO: load in teacher model using teacher_path ###
    
    return model

def generate_tabular_distillation_data(teacher, train_path, test_path, gpu=0):
    """Generate tabular concept and output data using teacher model for distillation and evaluation
    
        Paramaters: 
            teacher: teacher model
            train_path: path where training data is stored
            test_path: path where test data is stored
            gpu: gpu cuda device if applicable
            
        Returns:
            X_train_teacher (n_train, n_concepts): predicted concepts by teacher model for training data
            X_test_teacher (n_test, n_concepts): predicted concepts by teacher model for test data
            X_train (n_train, n_concepts): true concept training data (likely 0, 1)
            X_test (n_test, n_concepts): true concept test data (likely 0, 1)
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs for test data
            y_train (n_train, 1): true outputs for train data
            y_test (n_test, 1): true outputs for test data
            
    """
    ### TODO: generate teacher train and test data using model, train_path, and test_path ###
    
    return X_train_teacher, X_test_teacher, X_train, X_test, y_train_teacher, y_test_teacher, y_train, y_test
    
def process_distillation_data(X_train_teacher, X_test_teacher, X_train, X_test, y_train_teacher, y_test_teacher):
    """Process teacher data for distillation (likely binarizing the data)
    
        Paramaters: 
            X_train_teacher (n_train, n_concepts): predicted concepts by teacher model for training data
            X_test_teacher (n_test, n_concepts): predicted concepts by teacher model for test data
            X_train (n_train, n_concepts): true concept training data (likely 0, 1)
            X_test (n_test, n_concepts): true concept test data (likely 0, 1)
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs in evaluation form (i.e. if classification task, y_train_teacher must be class predictions, not class logits) for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs in evaluation form for test data
            
        Returns:
            X_train_teacher (n_train, n_concepts): processed predicted concepts by teacher model for distillation/student train data (likely 0, 1)
            X_test_teacher (n_test, n_concepts): processed predicted concepts by teacher model for student test data (likely 0, 1)
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs for test data
            
    """
    ### TODO: process (i.e. binarize) data for distillation ###
    
    return X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher

def process_student_eval(y_student):
    """Process student data outputs for evaluation (i.e. if we're using a regressor for distilling a classification model, need to argmax for evaluation)
    
        Paramaters: 
            y_student (n, n_outputs): student model predictions
            
        Returns:
            y_pred (n, ...): student model predictions in evaluation form (if regression, then perhaps y_pred = y_student)
            
    """
    ### TODO: handle student prediction outputs to match metrics ###

    return y_pred

def process_teacher_eval(y_teacher):
    """Process teacher data outputs for evaluation (i.e. if process teacher logits into classes)
    
        Paramaters: 
            y_student (n, n_outputs): teacher model predictions
            
        Returns:
            y_pred (n, ...): teacher model predictions in evaluation form (if regression, then perhaps y_pred = y_teacher)
            
    """
    ### TODO: process teacher model predictions for evaluations (sometimes we distill a teacher model using a regressor, but want to evaluate class prediction accuracy) ###
    
    return y_teacher_eval

def split_list_by_sizes(list1, list2):
    result = []
    for row1, row2 in zip(list1, list2):
        sizes = [len(sublist) for sublist in row1]
        row_result = []
        start = 0
        for size in sizes:
            end = start + size
            row_result.append(list(row2[start:end]))
            start = end
        result.append(row_result)
    return result


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
        "--teacher_path",
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
        "--student_name", 
        type=str,
        choices=["FIGSRegressorCV", "FIGSRegressor", "XGBRegressor", "DecisionTreeRegressor", "RandomForestRegressor"],
        default="FIGSRegressorCV", 
        help="student name"
    )
    parser.add_argument("-n_trees_list", 
                        help="delimited max_trees_list input for FIGS CV",
                        default="30,40",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("-n_rules_list", 
                        help="delimited max_rules_list input for FIGS CV", 
                        default="125,200",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("-n_depth_list", 
                        help="delimited max_rules_list input for FIGS CV",
                        default="4",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("-min_impurity_decrease_list", 
                        help="delimited min_impurity_decrease_list input for FIGS CV",
                        default="0",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument(
        "--max_rules", type=int, default=200, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=30, help="max trees of FIGS & XGB model"
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="max depth of tree based models"
    )
    parser.add_argument(
        "--metric", type=str, default="accuracy", help="metric to log distillation and prediction performance"
    )
    parser.add_argument(
        "--num_interactions_intervention", type=int, default=3, help="max interactions to intervene on"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device"
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
    
    teacher = load_teacher_model(args.teacher_path, args.gpu)
    
    X_train_t, X_test_t, X_train, X_test, y_train_t, y_test_t, y_train, y_test = generate_tabular_distillation_data(teacher, args.train_path, args.test_path, args.gpu)
    
    X_train_d, X_test_d, y_train_d, y_test_d = process_distillation_data(X_train_t, X_test_t, X_train, X_test, y_train_t, y_test_t)
    
    y_train_t_eval = process_teacher_eval(y_train_t)
    y_test_t_eval = process_teacher_eval(y_test_t)
    
    figs_student = idistill.model.get_model(args.task_type, args.student_name, args)
    
    r, figs_student = distill_model(figs_student, X_train_d, y_train_d, r)
    
    r['max_trees'] = figs_student.max_trees
    r['max_rules'] = figs_student.max_rules
    r['max_depth'] = figs_student.max_depth
    try:
        r['n_trees'] = len(figs_student.figs.trees_)
        r['n_rules'] = figs_student.figs.complexity_
    except:
        r['n_trees'] = len(figs_student.trees_)
        r['n_rules'] = figs_student.complexity_
        
    r = evaluate_student(figs_student, X_train_d, X_test_d, y_train_t_eval, y_test_t_eval, args.metric, "distillation", r)
    r = evaluate_student(figs_student, X_train_d, X_test_d, y_train, y_test, args.metric, "prediction", r)
    
    r = evaluate_teacher(y_train_t_eval, y_test_t_eval, y_train, y_test, args.metric, "prediction", r)
    
    ### adaptive FIGS concept editing ###
    
    figs_student.extract_interactions()
    
    r['depth'] = max([max([len(i[0]) for i in t]) for t in figs_student.interactions])
    
    train_q5 = np.quantile(X_train_t, 0.05, axis = 0)
    train_q95 = np.quantile(X_train_t, 0.95, axis = 0)

    X_test_d_a_edit = X_test_d.copy()
    X_test_d_r_edit = X_test_d.copy()

    X_test_t_a_edit = X_test_t.copy()
    X_test_t_r_edit = X_test_t.copy()

    
    cti_adap_test = figs_student.extract_atti(X_test_d, args.num_interactions_intervention)

    cti_rand_test = [np.random.choice(np.arange(X_test_d.shape[1]), X_test_d.shape[1], replace=False) for i in range(X_test_d.shape[0])]
    cti_rand_test = split_list_by_sizes(cti_adap_test, cti_rand_test)
    
    
    if 'linear' in args.teacher_path or 'Linear' in args.teacher_path:
        X_test_d_l_edit = X_test_d.copy()
        X_test_t_l_edit = X_test_t.copy()
        
        test_l_edit = np.einsum('nc, yc -> nyc', X_test_t.values, teacher.sec_model.linear.weight.cpu().detach().numpy())

        cti_l_test_arr = np.argsort(np.var(np.abs(test_l_edit), axis = 1), axis = 1)[:, ::-1]
        cti_l_test = [row for row in cti_l_test_arr]
        cti_l_test = split_list_by_sizes(cti_adap_test, cti_l_test)
        
    if args.num_interactions_intervention == 0:
        num_iters = len(figs_student.trees_)
    else:
        num_iters = args.num_interactions_intervention
        
    for i in range(num_iters):
        for n in range(X_test_d.shape[0]):

            X_test_d_a_edit.iloc[n, cti_adap_test[n][i]] = X_test.iloc[n, cti_adap_test[n][i]]
            X_test_d_r_edit.iloc[n, cti_rand_test[n][i]] = X_test.iloc[n, cti_rand_test[n][i]]

            X_test_t_a_edit.iloc[n, cti_adap_test[n][i]] = train_q5[cti_adap_test[n][i]]*(X_test.iloc[n, cti_adap_test[n][i]] == 0) + train_q95[cti_adap_test[n][i]]*(X_test.iloc[n, cti_adap_test[n][i]])
            X_test_t_r_edit.iloc[n, cti_rand_test[n][i]] = train_q5[cti_rand_test[n][i]]*(X_test.iloc[n, cti_rand_test[n][i]] == 0) + train_q95[cti_rand_test[n][i]]*(X_test.iloc[n, cti_rand_test[n][i]])
            
            if 'linear' in args.teacher_path or 'Linear' in args.teacher_path:
                X_test_d_l_edit.iloc[n, cti_l_test[n][i]] = X_test.iloc[n, cti_l_test[n][i]]
                X_test_t_l_edit.iloc[n, cti_l_test[n][i]] = train_q5[cti_l_test[n][i]]*(X_test.iloc[n, cti_l_test[n][i]] == 0) + train_q95[cti_l_test[n][i]]*(X_test.iloc[n, cti_l_test[n][i]])

        y_test_t_eval_a_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_a_edit, args.gpu))
        y_test_t_eval_r_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_r_edit, args.gpu))
            
        r = evaluate_test_student(figs_student, X_test_d_a_edit, y_test_t_eval_a_interv, args.metric, f"distill_adap_interv_iter{i}", r)
        r = evaluate_test_student(figs_student, X_test_d_r_edit, y_test_t_eval_r_interv, args.metric, f"distill_rand_interv_iter{i}", r)
        
        r = evaluate_test_student(figs_student, X_test_d_a_edit, y_test, args.metric, f"pred_adap_interv_iter{i}", r)
        r = evaluate_test_student(figs_student, X_test_d_r_edit, y_test, args.metric, f"pred_rand_interv_iter{i}", r)

        r = evaluate_test_teacher(y_test_t_eval_a_interv, y_test, args.metric, f"pred_adap_interv_iter{i}", r)
        r = evaluate_test_teacher(y_test_t_eval_r_interv, y_test, args.metric, f"pred_rand_interv_iter{i}", r)
        
        if 'linear' in args.teacher_path or 'Linear' in args.teacher_path:
            y_test_t_eval_l_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_l_edit, args.gpu))
            
            r = evaluate_test_student(figs_student, X_test_d_l_edit, y_test_t_eval_l_interv, args.metric, f"distill_lin_interv_iter{i}", r)
            r = evaluate_test_student(figs_student, X_test_d_l_edit, y_test, args.metric, f"pred_lin_interv_iter{i}", r)
            r = evaluate_test_teacher(y_test_t_eval_l_interv, y_test, args.metric, f"pred_lin_interv_iter{i}", r)
        
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")