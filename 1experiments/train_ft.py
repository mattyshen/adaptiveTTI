# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal
import time

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import json
import sys
from sklearn.metrics import r2_score

warnings.resetwarnings()

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

sys.path.append('..')

from interpretDistill.fourierDistill import *
from interpretDistill.binaryTransformer import *

n_inter = 3
k_cv = 3

for cur_seed in [0, 1, 2, 3, 4]:
    print(f'current seed: {cur_seed}')
    dataset = sklearn.datasets.fetch_california_housing(as_frame = True)
    X: np.ndarray = dataset["data"]
    Y: np.ndarray = dataset["target"]

    all_idx = np.arange(len(Y))
    train_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state = cur_seed
    )
    # train_idx, val_idx = sklearn.model_selection.train_test_split(
    #     trainval_idx, train_size=0.8, random_state = 0
    # )

    bt_bin3 = BinaryTransformer(depth = 3, bit = False)
    X_train_bin3 = bt_bin3.fit_and_transform(X.loc[train_idx, :], Y.loc[train_idx])
    X_test_bin3 = bt_bin3.transform(X.loc[test_idx, :])

    bt_bit3 = BinaryTransformer(depth = 3, bit = True)
    X_train_bit3 = bt_bit3.fit_and_transform(X.loc[train_idx, :], Y.loc[train_idx])
    X_test_bit3 = bt_bit3.transform(X.loc[test_idx, :])

    bt_bit4 = BinaryTransformer(depth = 4, bit = True)
    X_train_bit4 = bt_bit4.fit_and_transform(X.loc[train_idx, :], Y.loc[train_idx])
    X_test_bit4 = bt_bit4.transform(X.loc[test_idx, :])

    y_train = Y.loc[train_idx]
    y_test = Y.loc[test_idx]

    train_time = []

    ftd_bin3 = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
    ftd_bit3 = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
    ftd_bit4 = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)

    start = time.time()
    ftd_bin3.fit(X_train_bin3, y_train, bt_bin3.no_interaction)
    end = time.time()
    train_time.append(end - start)
    start = time.time()
    ftd_bit3.fit(X_train_bit3, y_train)
    end = time.time()
    train_time.append(end - start)
    start = time.time()
    ftd_bit4.fit(X_train_bit4, y_train)
    end = time.time()
    train_time.append(end - start)

    model_list = [ftd_bin3, ftd_bit3, ftd_bit4]
    model_names = ['(bin3, true, train)', '(bit3, true, train)', '(bit4, true, train)']

    r2_df = pd.DataFrame()
    r2_df['Model'] = model_names

    r2_df['Train R2'] = [r2_score(ftd_bin3.predict(X_train_bin3), y_train),r2_score(ftd_bit3.predict(X_train_bit3), y_train), r2_score(ftd_bit4.predict(X_train_bit4), y_train)]
    r2_df['Test R2'] = [r2_score(ftd_bin3.predict(X_test_bin3), y_test),r2_score(ftd_bit3.predict(X_test_bit3), y_test), r2_score(ftd_bit4.predict(X_test_bit4), y_test)]

    r2_df['Train Time'] = train_time
    r2_df['Total Num Features'] = [len(m.regression_model.coef_) for m in model_list]
    r2_df['Num Selected Features'] = [sum(m.regression_model.coef_ != 0) for m in model_list]

    r2_df.to_csv(f'r2/binarize_prediction_seed{cur_seed}.csv')
