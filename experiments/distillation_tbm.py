import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import OneHotEncoder
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
import ast

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_to_repo)

import idistill.model

from idistill.tbm_simple_transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    if metric == "neg_mean_squared_error": # the "neg" is to follow the sklearn CV scoring naming convention
        metric = "mse"
        
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
        y_pred_ = process_student_eval(student.predict(X_), args.teacher_name, args.task_type)
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
    if metric == "neg_mean_squared_error":  # the "neg" is to follow the sklearn CV scoring naming convention
        metric = "mse"
        
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
    
    if metric == "neg_mean_squared_error": # the "neg" is to follow the sklearn CV scoring naming convention
        metric = "mse"
        
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    y_pred_ = process_student_eval(student.predict(X_test), args.teacher_name, args.task_type)
    r[f"student_{task}_test_{metric}"] = metric_fn(y_test, y_pred_)

    return r

def evaluate_test_teacher(y_test_teacher, y_test, metric, task, r):
    
    if metric == "neg_mean_squared_error": # the "neg" is to follow the sklearn CV scoring naming convention
        metric = "mse"
        
    metrics = {
            "accuracy": accuracy_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "f1": f1_score,
        
        }
    
    metric_fn = metrics[metric]
    
    r[f"teacher_{task}_test_{metric}"] = metric_fn(y_test_teacher, y_test)
    
    return r

def predict_teacher(model, X, task_type):
    """Make prediction from concepts to outputs with teacher model
    
        Paramaters: 
            model: teacher model
            X (n, n_concepts): concept data
            task_type: regression or classification
            
        Returns:
            y_pred: teacher model predictions for X 
            
    """
    
    if task_type == "regression":
        y_pred = model.predict(X).reshape(-1, 1) #[N, 1]
    else:
        y_pred = model.predict_proba(X) #[N, #_class]
            
    return y_pred

def load_teacher_model(task_type, teacher_type):
    """Load in teacher model
    
        Paramaters:
            task_type: regression or classification
            teacher_type: transformer, linear, mlp1, or mlp2
            
        Returns:
            model: teacher model
            
    """
    if teacher_type == "transformer":
        model = None # we will load it in in `generate_tabular_distillation_data`
    else:    
        if task_type == "regression":
            if teacher_type == "linear":
                model = LinearRegression()
            elif teacher_type == "mlp1":
                model = MLPRegressor(hidden_layer_sizes=(50,), early_stopping=True, random_state=42)
            elif teacher_type == "mlp2":
                model = MLPRegressor(hidden_layer_sizes=(50,50,), early_stopping=True, random_state=42)
        else:
            if teacher_type == "linear":
                model = LogisticRegression(random_state=42)
            elif teacher_type == "mlp1":
                model = MLPClassifier(hidden_layer_sizes=(50,), early_stopping=True, random_state=42)
            elif teacher_type == "mlp2":
                model = MLPClassifier(hidden_layer_sizes=(50,50,), early_stopping=True, random_state=42)
    
    return model


def generate_tabular_distillation_data(model, saved_data_path, task_name, teacher_type, task_type, task_output_classes, teacher_path, do_intervention):
    """Generate tabular concept and output data using teacher model for distillation and evaluation
    
        Paramaters: 
            model: teacher model
            saved_data_path: path where data is stored
            task_name: dataset name
            teacher_type: transformer, linear, mlp1, or mlp2
            task_type: regression or classification
            task_output_classes: number of output classes in the task
            teacher_path: path where the model is saved (only applicable to transformer teacher type)
            do_intervention: if True, load in the human annotated concept vectors
            
        Returns:
            X_train_teacher (n_train, n_concepts): predicted concepts by teacher model for training data
            X_test_teacher (n_test, n_concepts): predicted concepts by teacher model for test data
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs for test data
            y_train (n_train, 1): true outputs for train data
            y_test (n_test, 1): true outputs for test data
            categories: list of categories of each concept
            X_test (n_test, n_concepts): true concept test data
    """
    
    def remove_dup(input_lst):
        res = []
        for x in input_lst:
            if x not in res:
                res.append(x)
        return res
    
    def remove_dup_concepts(concepts, cats):
        new_concepts, new_cats = [], []
        for x, y in zip(concepts, cats):
            if x not in new_concepts:
                new_concepts.append(x)
                new_cats.append(y)
        return new_concepts, new_cats
        
    concept_file = join(saved_data_path, "{}_{}".format(task_name, teacher_type), "{}_concepts.csv".format(task_name))
    train_file = join(saved_data_path, "{}_{}".format(task_name, teacher_type), "{}_train_df.csv".format(task_name))
    
    concept_df = pd.read_csv(concept_file)
    train_df = pd.read_csv(train_file)
    
    concepts = concept_df['Concept Name'].to_list()
    categories = [y for y in concept_df['Response Mapping'].apply(lambda x: remove_dup([str(float(cat)) for cat in ast.literal_eval(x).values()]))]
    concepts, categories = remove_dup_concepts(concepts, categories)
    
    if do_intervention == "False":
        test_file = join(saved_data_path, "{}_{}".format(task_name, teacher_type), "{}_test_df.csv".format(task_name))
        X_test = None
        test_df = pd.read_csv(test_file)
        X_test_teacher = test_df[concepts]
        y_test = test_df['label'].to_numpy().reshape(-1, 1)
    else:
        test_file = join(saved_data_path, "human_annotation", "{}_{}".format(task_name, teacher_type), "full_test_subset.csv".format(task_name))
        if task_name == "agnews":
            gt_test_file = join(saved_data_path, "human_annotation", "{}_{}".format(task_name, teacher_type), "X_test_v3.csv".format(task_name))
        else:
            gt_test_file = join(saved_data_path, "human_annotation", "{}_{}".format(task_name, teacher_type), "X_test.csv".format(task_name))
        X_test = pd.read_csv(gt_test_file)
        test_df = pd.read_csv(test_file)
        if task_name == "cebab":
            exclude = [16, 19, 21] # bad samples as indicated by human annotators
        else:
            exclude = []
        good_idxs = [i for i in range(len(test_df)) if i not in exclude]
        X_test_concept_col_idxs = [list(X_test.columns).index(c) for c in concepts]
        test_df_concept_col_idxs = [list(test_df.columns).index(c) for c in concepts]
        
        X_test = X_test.iloc[good_idxs, X_test_concept_col_idxs]
        X_test_teacher = test_df.iloc[good_idxs, test_df_concept_col_idxs]
        y_test = test_df.iloc[good_idxs, list(test_df.columns).index('label')].to_numpy().reshape(-1, 1)
    

    
    X_train_teacher = train_df[concepts]  
    y_train = train_df['label'].to_numpy().reshape(-1, 1)
    
    if teacher_type == "transformer":
        is_regression = True if task_type == "regression" else False
        
        tbm_train_dataset = TBMDataset(train_file, concept_file, is_regression=is_regression)
        train_dataloader = DataLoader(tbm_train_dataset, batch_size=128, shuffle=False, num_workers=1)
        vocab_size = tbm_train_dataset._get_vocab_size()
        
        tbm_test_dataset = TBMDataset(test_file, concept_file, is_regression=is_regression)
        test_dataloader = DataLoader(tbm_test_dataset, batch_size=128, shuffle=False, num_workers=1)
        
        # laod in model
        model = SimpleTransformer(vocab_size=vocab_size, embedding_dim=52, num_heads=4, num_layers=2, num_classes=task_output_classes).to(device)
        model.load_state_dict(torch.load(join(saved_data_path, "{}_{}".format(task_name, teacher_type), teacher_path))['model_state_dict'])
        
        if task_type == "regression":
            y_train_teacher = predict(model, train_dataloader, device, is_regression).reshape(-1, 1)
            y_test_teacher = predict(model, test_dataloader, device, is_regression).reshape(-1, 1)
        else:
            y_train_teacher = predict_proba(model, train_dataloader, device, is_regression)
            y_test_teacher = predict_proba(model, test_dataloader, device, is_regression)

    else:
        model.fit(X_train_teacher, y_train)
        
        filename = f'{task_name}_{teacher_type}_model.pkl'
        joblib.dump(model, filename)
            
        if task_type == "regression":
            y_train_teacher = model.predict(X_train_teacher).reshape(-1, 1) #[N, 1]
            y_test_teacher = model.predict(X_test_teacher).reshape(-1, 1)
        else:
            y_train_teacher = model.predict_proba(X_train_teacher) #[N, #_class]
            y_test_teacher = model.predict_proba(X_test_teacher)
    
    return X_train_teacher, X_test_teacher, y_train_teacher, y_test_teacher, y_train, y_test, categories, X_test
    
def process_distillation_data(X_train_teacher, X_test_teacher, X_test, y_train_teacher, y_test_teacher, categories):
    """Process teacher data for distillation (likely binarizing the data)
    
        Paramaters: 
            X_train_teacher (n_train, n_concepts): predicted concepts by teacher model for training data
            X_test_teacher (n_test, n_concepts): predicted concepts by teacher model for test data
            X_test (n_test, n_concepts): true concept test data
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs in evaluation form (i.e. if classification task, y_train_teacher must be class predictions, not class logits) for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs in evaluation form for test data
            categories: list of categories of each concept
            
        Returns:
            encoded_X_train_t (n_train, n_concepts): processed predicted concepts by teacher model for distillation/student train data (likely 0, 1)
            encoded_X_test_t (n_test, n_concepts): processed predicted concepts by teacher model for student test data (likely 0, 1)
            y_train_teacher (n_train, n_outputs): teacher model's predicted task outputs for train data
            y_test_teacher (n_test, n_outputs): teacher model's predicted task outputs for test data
            encoded_X_test (n_test, n_concepts): true concept test data (likely 0, 1)
            ohe_column_names: one hot encoding feature names
            encoder: one hot encoder
            
    """
    
    encoder = OneHotEncoder(categories=categories, handle_unknown='ignore')
    encoder.fit(X_train_teacher.astype(str))  # Fit only on training data

    encoded_X_train_t = pd.DataFrame(
        encoder.transform(X_train_teacher.astype(str)).toarray(),
        columns=encoder.get_feature_names_out()
    )
    
    encoded_X_test_t = pd.DataFrame(
        encoder.transform(X_test_teacher.astype(str)).toarray(),
        columns=encoder.get_feature_names_out()
    )
    
    if X_test is not None:
        encoded_X_test = pd.DataFrame(
            encoder.transform(X_test.astype(str)).toarray(),
            columns=encoder.get_feature_names_out()
        )
    else:
        encoded_X_test = None
    
    ohe_column_names = encoder.get_feature_names_out()
    
    return encoded_X_train_t, encoded_X_test_t, y_train_teacher, y_test_teacher, encoded_X_test, ohe_column_names, encoder

'''def create_map_ohe_idx_to_concept_idx(ohe_column_names, saved_data_path, task_name, teacher_type):
    concept_file = join(saved_data_path, "{}_{}".format(task_name, teacher_type), "{}_concepts.csv".format(task_name))
    concept_df = pd.read_csv(concept_file)
    concepts = concept_df['Concept Name'].to_list()
    
    idx_map, score_map = {}, {}
    for i, x in enumerate(ohe_column_names):
        if x[-4] == '-':
            concept_name = x[:-5]
            score = float(x[-4:])
        else:
            concept_name = x[:-4]
            score = float(x[-3:])
        #concept_name, score = x.split('_')
        idx_map[i] = concepts.index(concept_name)
        score_map[i] = float(score)
    return idx_map, score_map'''

def create_map_ohe_idx_to_concept_idx(ohe_column_names, saved_data_path, task_name, teacher_type):
    concept_file = join(saved_data_path, f"{task_name}_{teacher_type}", f"{task_name}_concepts.csv")
    concept_df = pd.read_csv(concept_file)
    concepts = concept_df['Concept Name'].to_list()
    
    idx_map, score_map = {}, {}
    for i, x in enumerate(ohe_column_names):
        # Safer: split from the right, once
        try:
            concept_name, score_str = x.rsplit('_', 1)
            score = float(score_str)
        except ValueError:
            raise ValueError(f"Column name '{x}' is not in expected format '<concept>_<score>'")

        if concept_name not in concepts:
            raise ValueError(f"Concept name '{concept_name}' not found in concept list.")

        idx_map[i] = concepts.index(concept_name)
        score_map[i] = score

    return idx_map, score_map
    
def process_student_eval(y_student, teacher_type, task_type):
    """Process student data outputs for evaluation (i.e. if we're using a regressor for distilling a classification model, need to argmax for evaluation)
    
        Paramaters: 
            y_student (n, n_outputs): student model predictions
            teacher_type: transformer, linear, mlp1, or mlp2
            task_type: regression or classification
            
        Returns:
            y_pred (n, ...): student model predictions in evaluation form (if regression, then perhaps y_pred = y_student)
            
    """
    if task_type == "regression":
        y_pred = y_student
    else:
        y_pred = np.argmax(y_student, axis=1)
    return y_pred

def process_teacher_eval(y_teacher, teacher_type, task_type):
    """Process teacher data outputs for evaluation (i.e. if process teacher logits into classes)
    
        Paramaters: 
            y_teacher (n, n_outputs): teacher model predictions
            teacher_type: transformer, linear, mlp1, or mlp2
            task_type: regression or classification
            
        Returns:
            y_teacher_eval (n, ...): teacher model predictions in evaluation form (if regression, then perhaps y_pred = y_teacher)
            
    """
    if task_type == "regression":
        y_teacher_eval = y_teacher
    else:
        y_teacher_eval = np.argmax(y_teacher, axis=1)
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

def map_into_idx_list(idx_map, orig_idx_list):
    res = [idx_map[x] for x in orig_idx_list]
    return res

def map_into_data_arr(idx_map, orig_idx_list):
    res = [idx_map[x] for x in orig_idx_list]
    return np.array(res)
        
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
        default=join(path_to_repo, "data/tbm_gpt4"),
        help="path to teacher model",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=join(path_to_repo, "data/tbm_gpt4"),
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
        "--task_name", 
        type=str, 
        choices=["cebab", "news_partisanship", "agnews"],
        default="cebab", 
        help="name of task"
    )
    parser.add_argument(
        "--teacher_name", 
        type=str,
        choices=["linear", "mlp1", "mlp2", "transformer"],
        default="linear", 
        help="teacher name"
    )
    parser.add_argument(
        "--student_name", 
        type=str,
        choices=["FIGSRegressorCV", "FIGSRegressor", "XGBRegressor"],
        default="FIGSRegressorCV", 
        help="student name"
    )
    parser.add_argument(
        "--do_intervention", 
        type=str,
        choices=["True", "False"],
        default="False", 
        help="whether to do intervention after distillation"
    )
    parser.add_argument("--n_trees_list", 
                        help="delimited max_trees_list input for FIGS CV",
                        default="50",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("--n_rules_list", 
                        help="delimited max_rules_list input for FIGS CV", 
                        default="100",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("--n_depth_list", 
                        help="delimited n_depth_list input for FIGS CV",
                        default="3",
                        type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument("--min_impurity_decrease_list", 
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
        "--metric", type=str, choices=["neg_mean_squared_error", "accuracy", "r2", "f1"], default="r2", help="metric to log distillation and prediction performance"
    )
    parser.add_argument(
        "--num_interactions_intervention", type=int, default=0, help="max interactions to intervene on"
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
    
    teacher = load_teacher_model(args.task_type, args.teacher_name)
    task_output_classes = 1 if args.task_name == "cebab" else 4
    X_train_t, X_test_t, y_train_t, y_test_t, y_train, y_test, categories, X_test = generate_tabular_distillation_data(teacher, args.train_path, args.task_name, args.teacher_name, 
                                                                                                               args.task_type, task_output_classes, args.teacher_path, args.do_intervention)
    X_train_d, X_test_d, y_train_d, y_test_d, X_test_ohe, ohe_column_names, encoder = process_distillation_data(X_train_t, X_test_t, X_test, y_train_t, y_test_t, categories)
    y_train_t_eval = process_teacher_eval(y_train_t, args.teacher_name, args.task_type)
    y_test_t_eval = process_teacher_eval(y_test_t, args.teacher_name, args.task_type)
    
    figs_student = idistill.model.get_model(args.task_type, args.student_name, args)

    ### distillation ###
    print('distilling')
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
    
    ohe2concept_idx_map, ohe2score_map = create_map_ohe_idx_to_concept_idx(ohe_column_names, args.train_path, args.task_name, args.teacher_name)
    
    ### adaptive FIGS concept editing ###
    figs_student.extract_interactions()
    
    r['depth'] = max([max([len(i[0]) for i in t]) for t in figs_student.interactions])
    
    X_test_d_a_edit = X_test_d.copy()
    X_test_d_r_edit = X_test_d.copy()

    X_test_t_a_edit = X_test_t.copy()
    X_test_t_r_edit = X_test_t.copy()

    cti_adap_test = figs_student.extract_atti(X_test_d, args.num_interactions_intervention)
    
    cti_rand_test = [np.random.choice(np.arange(X_test_d.shape[1]), X_test_d.shape[1], replace=False) for i in range(X_test_d.shape[0])]
    cti_rand_test = split_list_by_sizes(cti_adap_test, cti_rand_test)
    
    if 'linear' in [args.teacher_path, args.teacher_name] or 'Linear' in [args.teacher_path, args.teacher_name]:
        X_test_d_l_edit = X_test_d.copy()
        X_test_t_l_edit = X_test_t.copy()
        
        test_l_edit = np.einsum('nc, yc -> nyc', X_test_t.values, teacher.coef_)
        
        if args.task_type == "regression":
            cti_l_test_arr = np.argsort(np.max(np.abs(test_l_edit), axis = 1), axis = 1)[:, ::-1]
        else:
            cti_l_test_arr = np.argsort(np.var(np.abs(test_l_edit), axis = 1), axis = 1)[:, ::-1]
        cti_l_test = [row for row in cti_l_test_arr]
        cti_l_test = split_list_by_sizes(cti_adap_test, cti_l_test)
    
    
    if args.num_interactions_intervention == 0:
        num_iters = len(figs_student.figs.trees_)
    else:
        num_iters = args.num_interactions_intervention


    cti_adap_test_concept_space = [[map_into_idx_list(ohe2concept_idx_map, cti_adap_test[n][i]) for i in range(len(cti_adap_test[n]))] for n in range(X_test_d.shape[0])]
    cti_rand_test_concept_space = [[map_into_idx_list(ohe2concept_idx_map, cti_rand_test[n][i]) for i in range(len(cti_rand_test[n]))] for n in range(X_test_d.shape[0])]
    
    for i in range(num_iters):
        for n in range(X_test_d.shape[0]):

            #X_test_d_a_edit.iloc[n, cti_adap_test[n][i]] = X_test_ohe.iloc[n, cti_adap_test[n][i]]
            #X_test_d_r_edit.iloc[n, cti_rand_test[n][i]] = X_test_ohe.iloc[n, cti_rand_test[n][i]]
            
            adap_idxs = map_into_idx_list(ohe2concept_idx_map, cti_adap_test[n][i])
            rand_idxs = map_into_idx_list(ohe2concept_idx_map, cti_rand_test[n][i])
            
            X_test_t_a_edit.iloc[n, adap_idxs] = X_test.iloc[n, adap_idxs]
            X_test_t_r_edit.iloc[n, rand_idxs] = X_test.iloc[n, rand_idxs]

            X_test_d_a_edit = encoder.transform(X_test_t_a_edit.astype(str)).toarray()
            X_test_d_r_edit = encoder.transform(X_test_t_r_edit.astype(str)).toarray()
            
            if 'linear' in [args.teacher_path, args.teacher_name] or 'Linear' in [args.teacher_path, args.teacher_name]:
                l_idxs = map_into_idx_list(ohe2concept_idx_map, cti_l_test[n][i])
                
                #X_test_d_l_edit.iloc[n, cti_l_test[n][i]] = X_test_ohe.iloc[n, cti_l_test[n][i]]
                X_test_t_l_edit.iloc[n, l_idxs] = X_test.iloc[n, l_idxs]
                X_test_d_l_edit = encoder.transform(X_test_t_l_edit.astype(str)).toarray()

        y_test_t_eval_a_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_a_edit, args.task_type), args.teacher_name, args.task_type)
        y_test_t_eval_r_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_r_edit, args.task_type), args.teacher_name, args.task_type)
        
        r = evaluate_test_student(figs_student, X_test_d_a_edit, y_test, args.metric, f"pred_adap_interv_iter{i}", r)
        r = evaluate_test_student(figs_student, X_test_d_r_edit, y_test, args.metric, f"pred_rand_interv_iter{i}", r)
        
        r = evaluate_test_teacher(y_test_t_eval_a_interv, y_test, args.metric, f"pred_adap_interv_iter{i}", r)
        r = evaluate_test_teacher(y_test_t_eval_r_interv, y_test, args.metric, f"pred_rand_interv_iter{i}", r)
        
        if 'linear' in [args.teacher_path, args.teacher_name] or 'Linear' in [args.teacher_path, args.teacher_name]:
            y_test_t_eval_l_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_l_edit, args.task_type), args.teacher_name, args.task_type)
            
            r = evaluate_test_student(figs_student, X_test_d_l_edit, y_test, args.metric, f"pred_lin_interv_iter{i}", r)
            r = evaluate_test_teacher(y_test_t_eval_l_interv, y_test, args.metric, f"pred_lin_interv_iter{i}", r)
        
    
    # save the interaction groups
    feature_groups_df = pd.DataFrame({'cti_adap_test': cti_adap_test_concept_space, 'cti_rand_test': cti_rand_test_concept_space, 'cti_l_test': cti_l_test})
    feature_groups_df.to_csv(f'{args.task_name}_{args.teacher_name}_interv_groups.csv', index=False)
    
    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")