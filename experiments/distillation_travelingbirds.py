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
from idistill.whitebox_figs import FIGSRegressorCV

def distill_model(student, X_train_teacher, y_train_teacher, r, feature_names = None):
    """Distill the teacher model using the student model"""
    
    fit_parameters = inspect.signature(student.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        student.fit(X_train_teacher, y_train_teacher, feature_names=feature_names)
    else:
        student.fit(X_train_teacher, y_train_teacher)

    return r, student

def evaluate_student(student, X_train, X_test, y_train, y_test, metric, task, r):
    """Evaluate student performance on each split"""
    
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

def predict_teacher(teacher, X, gpu=0):
    ### TODO: handle teacher prediction outputs (X is intended to be concept design matrix, output is intended to be logits)###

    y_pred_torch = teacher.sec_model(torch.tensor(X.values, dtype=torch.float32).to(f'cuda:{gpu}'))
    y_pred = pd.DataFrame(y_pred_torch.detach().cpu().numpy())
        
    return y_pred

def load_teacher_model(teacher_path, gpu=0):
    ### TODO: load in teacher model using teacher_path ###
    
    sys.path.append('/home/mattyshen/ConceptBottleneck')
    try:
        teacher = torch.load(teacher_path, weights_only=False)
    except:
        teacher = torch.load(join(join(path_to_repo, "models"), teacher_path), weights_only=False)
    teacher.to(f'cuda:{gpu}')
    teacher.eval()
    sys.path.append('/home/mattyshen/DistillationEdit')
    
    return teacher

def generate_tabular_distillation_data(teacher, train_path, test_path, gpu=0):
    ### TODO: generate teacher train and test data using model, train_path, and test_path ###
    
    sys.path.append('/home/mattyshen/ConceptBottleneck/CUB')
    from dataset import load_data
    from config import BASE_DIR
    
    def get_cub_data(teacher, path, data = 'train', override_train = True, batch_size = 32):
        with torch.no_grad():
            if data == 'test':
                test_dir = path
                #print(test_dir)
                # loader = load_data([test_dir], True, False, batch_size, image_dir='images',
                #                    n_class_attr=2, override_train=override_train)
                loader = load_data([test_dir], True, False, batch_size, image_dir='AdversarialData/CUB_fixed/test',
                                   n_class_attr=2)
            else:
                train_dir = path
                val_dir = '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/val.pkl'
                #print(train_dir, val_dir)
                # loader = load_data([train_dir, val_dir], True, False, batch_size, image_dir='images',
                #                    n_class_attr=2, override_train=override_train)
                loader = load_data([train_dir, val_dir], True, False, batch_size, image_dir='AdversarialData/CUB_fixed/train',
                                    n_class_attr=2)
                
            torch.manual_seed(0)
            
            attrs_true = []
            attrs_hat = []
            labels_true = []
            labels_hat = []
            for data_idx, data in enumerate(loader):
                inputs, labels, attr_labels = data
                attr_labels = torch.stack(attr_labels).t()

                inputs_var = torch.autograd.Variable(inputs).to(f'cuda:{gpu}')
                labels_var = torch.autograd.Variable(labels).to(f'cuda:{gpu}')
                outputs = teacher(inputs_var)
                class_outputs = outputs[0]

                attr_outputs = outputs[1:] #[torch.nn.Sigmoid()(o) for o in outputs[1:]]
                #attr_outputs_sigmoid = attr_outputs

                attrs_hat.append(torch.stack(attr_outputs).squeeze(2).detach().cpu().numpy())
                attrs_true.append(attr_labels.T)
                labels_hat.append(class_outputs.detach().cpu().numpy())
                labels_true.append(labels)

            X_hat = pd.DataFrame(np.concatenate(attrs_hat, axis=1).T, columns = [f'c{i}' for i in range(1, 113)])
            X = pd.DataFrame(np.concatenate(attrs_true, axis = 1).T, columns = [f'c{i}' for i in range(1, 113)])

            y = pd.Series(np.concatenate([l.numpy().reshape(-1, ) for l in labels_true]))
            y_hat = pd.DataFrame(np.concatenate(labels_hat, axis = 0))

            del attrs_hat
            del labels
            del labels_hat
            del loader
            del data
            del inputs
            del outputs
            del class_outputs
            del attr_outputs
            del inputs_var
            del labels_var
            torch.cuda.empty_cache()

            return X_hat, X, y_hat, y

    X_train_teacher, X_train, y_train_teacher, y_train = get_cub_data(teacher, train_path)
    X_test_teacher, X_test, y_test_teacher, y_test = get_cub_data(teacher, test_path, data = 'test')
    
    sys.path.append('/home/mattyshen/DistillationEdit')
    
    return X_train_teacher, X_test_teacher, X_train, X_test, y_train_teacher, y_test_teacher, y_train, y_test
    
def process_distillation_data(X_train_teacher, X_test_teacher, X_train, X_test, y_train_teacher, y_test_teacher):
    ### TODO: process (i.e. binarize, F1-max binarize) data for distillation ###
    
    thresh = 0
    
    return (X_train_teacher > thresh).astype(int), (X_test_teacher > thresh).astype(int), y_train_teacher, y_test_teacher

def process_student_eval(y_student):
    ### TODO: handle student prediction outputs to match metrics ###
    
    y_pred = np.argmax(y_student, axis = 1)

    return y_pred

def process_teacher_eval(y_teacher):
    ### TODO: process teacher model predictions for evaluations (sometimes we distill a teacher model using a regressor, but want to evaluate class prediction accuracy) ###
    
    y_teacher_eval = y_teacher.idxmax(axis = 1).astype(int).values
    
    return y_teacher_eval

def extract_interactions(student):

    interactions = []

    def traverse_tree(node, current_features, current_depth):

        if node.left is None and node.right is None:
            tree_interactions.append((current_features, np.var(np.abs(node.value))))
            return
        if node.left is not None:
            current_features_l = current_features.copy()
            current_features_l.append('c' + str(node.feature+1))
            traverse_tree(node.left, current_features_l.copy(), current_depth=current_depth+1)
        if node.right is not None:
            current_features_r = current_features.copy()
            current_features_r.append('!c' + str(node.feature+1))
            traverse_tree(node.right, current_features_r.copy(), current_depth=current_depth+1)
            
    try:
        trees = student.trees_
    except:
        trees = student.figs.trees_

    for tree in trees:
        tree_interactions = []
        traverse_tree(tree, [], current_depth=0)
        interactions.append(tree_interactions)
        
    return interactions

def get_argmax_max(vals, index):
    
    maxes = np.partition(vals, -2, axis=1)[:, -index]
    argmaxes = np.argsort(vals, axis=1)[:, -index]
    return maxes, argmaxes

def extract_adaptive_intervention(student, X, interactions, number_of_top_paths, tol = 0.0001):
    
    test_pred_intervention = student.predict(X, by_tree = True)

    concepts_to_edit = [[] for _ in range(X.shape[0])]
    variances = np.var(np.abs(test_pred_intervention), axis = 1)

    for idx in range(number_of_top_paths):
        maxes, argmaxes = get_argmax_max(variances, idx+1)
        for i, (tree_idx, var) in enumerate(zip(argmaxes, maxes)):
            for paths in interactions[tree_idx]:
                if abs(paths[1] - var) < tol:
                    concept_indexes = [int(p[1:])-1 if p[0] != '!' else int(p[2:])-1 for p in paths[0]]
                    concepts_to_edit[i].append(concept_indexes)
                    
    concepts_to_edit = [sum(element, []) for element in concepts_to_edit]
    concepts_to_edit = [list(set(c)) for c in concepts_to_edit]
    
    return concepts_to_edit


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
        choices=["FIGSRegressorCV", "XGBRegressor"],
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
        "--max_rules", type=int, default=125, help="max rules of FIGS model"
    )
    parser.add_argument(
        "--max_trees", type=int, default=25, help="max trees of FIGS & XGB model"
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
    
    r['n_trees'] = len(figs_student.figs.trees_)
    r['n_rules'] = figs_student.figs.complexity_
    
    r = evaluate_student(figs_student, X_train_d, X_test_d, y_train_t_eval, y_test_t_eval, args.metric, "distillation", r)
    r = evaluate_student(figs_student, X_train_d, X_test_d, y_train, y_test, args.metric, "prediction", r)
    
    r = evaluate_teacher(y_train_t_eval, y_test_t_eval, y_train, y_test, args.metric, "prediction", r)
    
    ### adaptive FIGS concept editing ###
    
    figs_interactions = extract_interactions(figs_student)
    
    r['depth'] = max([max([len(i[0]) for i in t]) for t in figs_interactions])

    X_train_d_edit = X_train_d.copy()
    X_train_t_edit = X_train_t.copy()

    train_q5 = np.quantile(X_train_t, 0.05, axis = 0)
    train_q95 = np.quantile(X_train_t, 0.95, axis = 0)

    cti_train = extract_adaptive_intervention(figs_student, X_train_d, figs_interactions, args.num_interactions_intervention)

    for i in range(len(cti_train)):
        X_train_d_edit.iloc[i, cti_train[i]] = X_train.iloc[i, cti_train[i]]
        X_train_t_edit.iloc[i, cti_train[i]] = train_q5[cti_train[i]]*(X_train.iloc[i, cti_train[i]] == 0) + train_q95[cti_train[i]]*(X_train.iloc[i, cti_train[i]] == 1)

    cti_test = extract_adaptive_intervention(figs_student, X_test_d, figs_interactions, args.num_interactions_intervention)

    X_test_d_edit = X_test_d.copy()
    X_test_t_edit = X_test_t.copy()

    for i in range(len(cti_test)):
        X_test_d_edit.iloc[i, cti_test[i]] = X_test.iloc[i, cti_test[i]]
        X_test_t_edit.iloc[i, cti_test[i]] = train_q5[cti_test[i]]*(X_test.iloc[i, cti_test[i]] == 0) + train_q95[cti_test[i]]*(X_test.iloc[i, cti_test[i]] == 1)

    y_train_t_eval_interv = process_teacher_eval(predict_teacher(teacher, X_train_t_edit, args.gpu))
    y_test_t_eval_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_edit, args.gpu))

    r = evaluate_student(figs_student, X_train_d_edit, X_test_d_edit, y_train_t_eval_interv, y_test_t_eval_interv, args.metric, "distillation_adap_interv", r)
    r = evaluate_student(figs_student, X_train_d_edit, X_test_d_edit, y_train, y_test, args.metric, "prediction_adap_interv", r)

    r = evaluate_teacher(y_train_t_eval_interv, y_test_t_eval_interv, y_train, y_test, args.metric, "prediction_adap_interv", r)
    
    ### random FIGS concept editing ###
    
    cti_r_train = [np.random.choice(np.arange(0, 112), size=len(c), replace=False) for c in cti_train]
    cti_r_test = [np.random.choice(np.arange(0, 112), size=len(c), replace=False) for c in cti_test]
    
    X_train_d_r_edit = X_train_d.copy()
    X_train_t_r_edit = X_train_t.copy()

    for i in range(len(cti_r_train)):
        X_train_d_r_edit.iloc[i, cti_r_train[i]] = X_train.iloc[i, cti_r_train[i]]
        X_train_t_r_edit.iloc[i, cti_r_train[i]] = train_q5[cti_r_train[i]]*(X_train.iloc[i, cti_r_train[i]] == 0) + train_q95[cti_r_train[i]]*(X_train.iloc[i, cti_r_train[i]] == 1)

    X_test_d_r_edit = X_test_d.copy()
    X_test_t_r_edit = X_test_t.copy()

    for i in range(len(cti_r_test)):
        X_test_d_r_edit.iloc[i, cti_r_test[i]] = X_test.iloc[i, cti_r_test[i]]
        X_test_t_r_edit.iloc[i, cti_r_test[i]] = train_q5[cti_r_test[i]]*(X_test.iloc[i, cti_r_test[i]] == 0) + train_q95[cti_r_test[i]]*(X_test.iloc[i, cti_r_test[i]] == 1)

    y_train_t_eval_r_interv = process_teacher_eval(predict_teacher(teacher, X_train_t_r_edit, args.gpu))
    y_test_t_eval_r_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_r_edit, args.gpu))

    r = evaluate_student(figs_student, X_train_d_r_edit, X_test_d_r_edit, y_train_t_eval_r_interv, y_test_t_eval_r_interv, args.metric, "distillation_rand_interv", r)
    r = evaluate_student(figs_student, X_train_d_r_edit, X_test_d_r_edit, y_train, y_test, args.metric, "prediction_rand_interv", r)

    r = evaluate_teacher(y_train_t_eval_r_interv, y_test_t_eval_r_interv, y_train, y_test, args.metric, "prediction_rand_interv", r)
    
    ### linear CBM concept editing ###
    
    if 'linear' in args.teacher_path:
        #TODO: access linear weight from concept-to-prediction portion of the CBM model
        #example: CBM
        train_l_edit = np.einsum('nc, yc -> nyc', X_train_t.values, teacher.sec_model.linear.weight.cpu().detach().numpy())
        test_l_edit = np.einsum('nc, yc -> nyc', X_test_t.values, teacher.sec_model.linear.weight.cpu().detach().numpy())
        
        cti_l_train_arr = np.argsort(np.max(train_l_edit, axis = 1), axis = 1)[:, (-1 * r['depth'] * args.num_interactions_intervention):]
        cti_l_train = [row for row in cti_l_train_arr]

        cti_l_test_arr = np.argsort(np.max(test_l_edit, axis = 1), axis = 1)[:, (-1 * r['depth'] * args.num_interactions_intervention):]
        cti_l_test = [row for row in cti_l_test_arr]

        X_train_d_l_edit = X_train_d.copy()
        X_train_t_l_edit = X_train_t.copy()

        for i in range(len(cti_l_train)):
            X_train_d_l_edit.iloc[i, cti_l_train[i]] = X_train.iloc[i, cti_l_train[i]]
            X_train_t_l_edit.iloc[i, cti_l_train[i]] = train_q5[cti_l_train[i]]*(X_train.iloc[i, cti_l_train[i]] == 0) + train_q95[cti_l_train[i]]*(X_train.iloc[i, cti_l_train[i]] == 1)

        X_test_d_l_edit = X_test_d.copy()
        X_test_t_l_edit = X_test_t.copy()

        for i in range(len(cti_l_test)):
            X_test_d_l_edit.iloc[i, cti_l_test[i]] = X_test.iloc[i, cti_l_test[i]]
            X_test_t_l_edit.iloc[i, cti_l_test[i]] = train_q5[cti_l_test[i]]*(X_test.iloc[i, cti_l_test[i]] == 0) + train_q95[cti_l_test[i]]*(X_test.iloc[i, cti_l_test[i]] == 1)

        y_train_t_eval_l_interv = process_teacher_eval(predict_teacher(teacher, X_train_t_l_edit, args.gpu))
        y_test_t_eval_l_interv = process_teacher_eval(predict_teacher(teacher, X_test_t_l_edit, args.gpu))

        r = evaluate_student(figs_student, X_train_d_l_edit, X_test_d_l_edit, y_train_t_eval_l_interv, y_test_t_eval_l_interv, args.metric, "distillation_lin_interv", r)
        r = evaluate_student(figs_student, X_train_d_l_edit, X_test_d_l_edit, y_train, y_test, args.metric, "prediction_lin_interv", r)

        r = evaluate_teacher(y_train_t_eval_l_interv, y_test_t_eval_l_interv, y_train, y_test, args.metric, "prediction_lin_interv", r)
        
    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    #joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")