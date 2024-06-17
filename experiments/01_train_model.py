import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import inspect
import os.path
import imodelsx.cache_save_utils
import sys

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#os.chdir(path_to_repo)
#os.chdir('/home/mattyshen/interpretableDistillation')
sys.path.append(path_to_repo)

print(os.listdir())

import interpretDistill.model
import interpretDistill.data


def fit_model(model, X_train, y_train, feature_names, r):
    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    if "feature_names" in fit_parameters and feature_names is not None:
        model.fit(X_train, y_train, feature_names=feature_names)
    else:
        model.fit(X_train, y_train)

    return r, model


def evaluate_model(model, model_name, comp, task, X_train, X_test, y_train, y_test, r):
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
        ["train", "test"], [(X_train, y_train), (X_test, y_test)]
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
        "--task_type", type=str, default="regression", help="prediction task"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ca_housing", help="name of dataset"
    )
    parser.add_argument(
        "--subsample_frac", type=float, default=0.2, help="fraction of samples to use for test set"
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
        choices=["featurizer", "random_forest", "figs", "xgboost", "resnet", "ft_transformer", "ft_distill"],
        default="ft_transformer",
        help="name of (teacher, if distillation) model",
    )
    parser.add_argument(
        "--featurizer_name", type=str, default="no_featurizer", help="type of featurizer to discretize dataset"
    )
    parser.add_argument(
        "--featurizer_frac", type=float, default=0.33, help="fraction of train samples to fit featurizer"
    )
    parser.add_argument(
        "--featurizer_overlap", type=int, default=1, help="dictate whether featurizer samples and train samples are combined"
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="max depth of featurizer"
    )
    parser.add_argument(
        "--bit", type=int, default=1, help="dictate if featurizer uses bit representation"
    )
    parser.add_argument(
        "--max_depth", type=int, default=4, help="max depth of tree based models (RF, XGB)")
    parser.add_argument(
        "--max_rules", type=int, default=4, help="max rules of FIGS model"
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
        "--n_epochs", type=int, default=100, help="number of epochs for DL based models"
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
    print('-----')
    print(type(args), type(var)
    print('-----')
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
    # torch.manual_seed(args.seed)

    # load text data
    dset, dataset_key_text = interpretDistill.data.load_huggingface_dataset(
        dataset_name=args.dataset_name, subsample_frac=args.subsample_frac
    )
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = interpretDistill.data.convert_text_data_to_counts_array(dset, dataset_key_text)

    # load tabular data
    # https://csinva.io/imodels/util/data_util.html#imodels.util.data_util.get_clean_dataset
    # X_train, X_test, y_train, y_test, feature_names = imodels.get_clean_dataset('compas_two_year_clean', data_source='imodels', test_size=0.33)

    # load model
    featurizer = interpretDistill.model.get_model(args.task_type, args.featurizer_name, args)
    
    if featurizer is not None:
        X_train, X_f, y_train, y_f = train_test_split(
        X_train, y_train, test_size=args.featurizer_frac, random_state=args.seed
    )
        X_f = featurizer.fit_transform(X_f, y_f)
        
        X_train = featurizer.transform(X_train)
        X_test = featurizer.transform(X_test)
        
        if args.featurizer_overlap:
            X_train = pd.concat([X_train, X_f]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_f]).reset_index(drop=True)
    
    model = interpretDistill.model.get_model(args.task_type, args.model_name, args)
    
    if featurizer is not None:
        model_f = interpretDistill.model.get_model(args.task_type, args.model_name, args)
        

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # fit
    r, model = fit_model(model, X_train, y_train, feature_names, r)
    r = evaluate_model(model, args.model_name, 'true', args.tasktype, X_train, X_test, y_train, y_test, r)
    
    if featurizer is not None:
        r, model_f = fit_model(model_f, X_train, y_train, feature_names, r)
        r = evaluate_model(model_f, args.model_name+'_f', 'true', args.tasktype, X_train, X_test, y_train, y_test, r)


    # save results
    print(f'save_dir_unique: {save_dir_unique}')
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")