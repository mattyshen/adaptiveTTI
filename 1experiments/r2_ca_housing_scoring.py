# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal

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
from sklearn.metrics import mean_squared_error, r2_score
import time

warnings.resetwarnings()

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

sys.path.append('..')

from interpretDistill.fourierDistill import *
from interpretDistill.binaryTransformer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
delu.random.seed(0)

#depth
dt_depth = int(sys.argv[1])
#bit
bit_boolean = str(sys.argv[2])

if bit_boolean == 'bit':
    word_access = 'bit'
else:
    word_access = 'bin'
    
print(dt_depth, word_access)

# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]

task_type: TaskType = "regression"
n_classes = None
dataset = sklearn.datasets.fetch_california_housing(as_frame = True)
X: np.ndarray = dataset["data"]
Y: np.ndarray = dataset["target"]

all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8, random_state = 0
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8, random_state = 0
)

X_b = {}
bt = BinaryTransformer(depth = dt_depth, bit = bit_boolean)
X_b['train'] = bt.fit_and_transform(X.loc[train_idx, :], Y.loc[train_idx])
X_b['val'] = bt.transform(X.loc[val_idx, :])
X_b['test'] = bt.transform(X.loc[test_idx, :])


# >>> Continuous features.
#X_cont: np.ndarray = X[['carat', 'depth', 'table', 'x', 'y', 'z']].to_numpy().astype(np.float32)
n_cont_features = 0

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, but,
# for the demonstration purposes, it is possible to generate them.

cat_cardinalities = [len(X_b['train'][c].value_counts()) for c in X_b['train'].columns]
print(cat_cardinalities)


for k in X_b.keys():
    X_b[k] = (X_b[k] + 1) // 2

# >>> Labels.
# Regression labels must be represented by float32.

Y = Y.to_numpy()
if task_type == "regression":
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

data_numpy = {
    "train": {"x_cat": X_b['train'].to_numpy().astype(np.int64), "y": Y[train_idx]},
    "val": {"x_cat": X_b['val'].to_numpy().astype(np.int64), "y": Y[val_idx]},
    "test": {"x_cat": X_b['test'].to_numpy().astype(np.int64), "y": Y[test_idx]},
}

if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
data = {
    part: {'y': torch.as_tensor(data_numpy[part]['y'], device=device)}
    for part in data_numpy
}

if True:
    data["train"]["x_cat"] = torch.from_numpy(data_numpy["train"]["x_cat"]).to(torch.int64).to(device)
    data["val"]["x_cat"] = torch.from_numpy(data_numpy["val"]["x_cat"]).to(torch.int64).to(device)
    data["test"]["x_cat"] = torch.from_numpy(data_numpy["test"]["x_cat"]).to(torch.int64).to(device)

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()
        
with open(f'predictions/ftt_{word_access}_depth{str(dt_depth)}_preds.json') as json_file:
    b_preds = json.load(json_file)

with open('predictions/ftt_orig_preds.json') as json_file:
    orig_preds = json.load(json_file)
    
for s in b_preds.keys():
    for pt in b_preds[s].keys():
        b_preds[s][pt] = np.array(b_preds[s][pt])
        
for s in orig_preds.keys():
    for pt in orig_preds[s].keys():
        orig_preds[s][pt] = np.array(orig_preds[s][pt])
        
def dict_to_series(preds):
    ret = []
    for i in ['train', 'val', 'test']:
        ret.append(pd.Series(preds[i]['y_hat'], name = 'MedHouseVal'))
    return tuple(ret)

y_train_b, y_val_b, y_test_b = dict_to_series(b_preds)

y_train_orig, y_val_orig, y_test_orig = dict_to_series(orig_preds)

#train: (b, b) + (b, orig)
#val: (b, b) + (b, orig)
#train val: (b, b) + (b, orig)

n_inter = 3
k_cv = 2

ftd_bo_train = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
ftd_bb_train = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
ftd_bo_val = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
ftd_bb_val = FTDistillCV(size_interactions = n_inter , k_cv = k_cv)
ftd_bo_tv = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)
ftd_bb_tv = FTDistillCV(size_interactions = n_inter, k_cv = k_cv)

train_time = []

if bit_boolean:
    start = time.time()
    ftd_bo_train.fit(X_b['train'], y_train_orig)
    end = time.time()
    train_time.append(end-start)
    print('bo_train concluded')
    start = time.time()
    ftd_bb_train.fit(X_b['train'], y_train_b)
    end = time.time()
    train_time.append(end-start)
    print('bb_train concluded')
    start = time.time()
    ftd_bo_val.fit(X_b['val'], y_val_orig)
    end = time.time()
    train_time.append(end-start)
    print('bo_val concluded')
    start = time.time()
    ftd_bb_val.fit(X_b['val'], y_val_b)
    end = time.time()
    train_time.append(end-start)
    print('bb_val concluded')
    start = time.time()
    ftd_bo_tv.fit(pd.concat([X_b['train'], X_b['val']], axis = 0), pd.concat([y_train_orig, y_val_orig], axis = 0))
    end = time.time()
    train_time.append(end-start)
    print('bo_tv concluded')
    start = time.time()
    ftd_bb_tv.fit(pd.concat([X_b['train'], X_b['val']], axis = 0), pd.concat([y_train_b, y_val_b], axis = 0))
    end = time.time()
    train_time.append(end-start)
    print('bb_tv concluded')
else:
    start = time.time()
    ftd_bo_train.fit(X_b['train'], y_train_orig, bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bo_train concluded')
    start = time.time()
    ftd_bb_train.fit(X_b['train'], y_train_b, bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bb_train concluded')
    start = time.time()
    ftd_bo_val.fit(X_b['val'], y_val_orig, bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bo_val concluded')
    start = time.time()
    ftd_bb_val.fit(X_b['val'], y_val_b, bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bb_val concluded')
    start = time.time()
    ftd_bo_tv.fit(pd.concat([X_b['train'], X_b['val']], axis = 0), pd.concat([y_train_orig, y_val_orig], axis = 0), bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bo_tv concluded')
    start = time.time()
    ftd_bb_tv.fit(pd.concat([X_b['train'], X_b['val']], axis = 0), pd.concat([y_train_b, y_val_b], axis = 0), bt.no_interaction)
    end = time.time()
    train_time.append(end-start)
    print('bb_tv concluded')

ftd_list = [ftd_bo_train, ftd_bb_train, ftd_bo_val, ftd_bb_val, ftd_bo_tv, ftd_bb_tv]
ftd_names = [f'({word_access}, orig, train)', f'({word_access}, {word_access}, train)', f'({word_access}, orig, val)', f'({word_access}, {word_access}, val)', f'({word_access}, orig, train+val)', f'({word_access}, {word_access}, train+val)']

r2_true_df = pd.DataFrame(columns = ['Model', 'Train R2', 'Val R2', 'Test R2'])

for i, j in zip(['train', 'val', 'test'], ['Train R2', 'Val R2', 'Test R2']):
    r2_true_df[j] = [r2_score(m.predict(X_b[i]), orig_preds[i]['y_true']) for m in ftd_list]
    
r2_true_df['Model'] = ftd_names

r2_true_df.loc[len(r2_true_df)] = ['FTTransformer']+[r2_score(orig_preds[i]['y_hat'], orig_preds[i]['y_true']) for i in ['train', 'val', 'test']]
r2_true_df.loc[len(r2_true_df)] = [f'FTTransformer {word_access}']+[r2_score(b_preds[i]['y_hat'], b_preds[i]['y_true']) for i in ['train', 'val', 'test']]

train_time.append(-1)
train_time.append(-1)

r2_true_df['Train Time'] = train_time
#[ftd_bo_train, ftd_bb_train, ftd_bo_val, ftd_bb_val, ftd_bo_tv, ftd_bb_tv]
total_num_features = [len(ftd_bo_train.regression_model.coef_), len(ftd_bb_train.regression_model.coef_), len(ftd_bo_val.regression_model.coef_), len(ftd_bb_val.regression_model.coef_), len(ftd_bo_tv.regression_model.coef_), len(ftd_bb_tv.regression_model.coef_)]
total_num_features.append(-1)
total_num_features.append(-1)
r2_true_df['Total Num Features'] = total_num_features

num_selected_features = [sum(ftd_bo_train.regression_model.coef_ != 0), sum(ftd_bb_train.regression_model.coef_!= 0), sum(ftd_bo_val.regression_model.coef_!= 0), sum(ftd_bb_val.regression_model.coef_!= 0), sum(ftd_bo_tv.regression_model.coef_!= 0), sum(ftd_bb_tv.regression_model.coef_!= 0)]
num_selected_features.append(-1)
num_selected_features.append(-1)

r2_true_df['Num Selected Features'] = num_selected_features

r2_true_df.to_csv(f'r2/{word_access}_depth{str(dt_depth)}_distillation_true_R2.csv')

r2_hat_df = pd.DataFrame(columns = ['Model', 'Train R2', 'Val R2', 'Test R2'])

for i, j in zip(['train', 'val', 'test'], ['Train R2', 'Val R2', 'Test R2']):
    r2_hat_df[j] = [r2_score(m.predict(X_b[i]), orig_preds[i]['y_hat']) for m in ftd_list]
    
r2_hat_df['Model'] = ftd_names

r2_hat_df.loc[len(r2_hat_df)] = ['FTTransformer']+[r2_score(orig_preds[i]['y_hat'], orig_preds[i]['y_hat']) for i in ['train', 'val', 'test']]
r2_hat_df.loc[len(r2_hat_df)] = [f'FTTransformer {word_access}']+[r2_score(b_preds[i]['y_hat'], b_preds[i]['y_hat']) for i in ['train', 'val', 'test']]

r2_hat_df['Train Time'] = train_time
r2_hat_df['Total Num Features'] = total_num_features
r2_hat_df['Num Selected Features'] = num_selected_features

r2_hat_df.to_csv(f'r2/{word_access}_depth{str(dt_depth)}_distillation_hat_R2.csv')

print(f'r2/{word_access}_depth{str(dt_depth)}_distillation_true_R2.csv', f'r2/{word_access}_depth{str(dt_depth)}_distillation_hat_R2.csv')