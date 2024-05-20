# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import pandas as pd
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

warnings.resetwarnings()

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
delu.random.seed(0)

# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]

task_type: TaskType = "regression"
n_classes = None
dataset = pd.read_csv('/home/mattyshen/interpretableDistillation/simulations/Tabular_Gemstone_Price/data/train.csv').drop(columns =['id'])
X: np.ndarray = dataset.drop(columns = ['price']) #.to_numpy()
Y: np.ndarray = dataset['price'].to_numpy()

# >>> Continuous features.
X_cont: np.ndarray = X[['carat', 'depth', 'table', 'x', 'y', 'z']].to_numpy().astype(np.float32)
n_cont_features = X_cont.shape[1]

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, but,
# for the demonstration purposes, it is possible to generate them.
X_cat = X[['cut', 'color']]
for column in X_cat.columns:
    label_encoder = sklearn.preprocessing.LabelEncoder()
    X_cat[column] = label_encoder.fit_transform(X_cat[column]).astype(np.int64)
    
cat_cardinalities = [len(X_cat[c].value_counts()) for c in X_cat.columns]
X_cat = X_cat.to_numpy().astype(np.int64)

# >>> Labels.
# Regression labels must be represented by float32.
if task_type == "regression":
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

# >>> Split the dataset.
all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8
)
data_numpy = {
    "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
    "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
    "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
}
if X_cat is not None:
    data_numpy["train"]["x_cat"] = X_cat[train_idx]
    data_numpy["val"]["x_cat"] = X_cat[val_idx]
    data_numpy["test"]["x_cat"] = X_cat[test_idx]
    
# >>> Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# (A) Simple preprocessing strategy.
# preprocessing = sklearn.preprocessing.StandardScaler().fit(
#     data_numpy['train']['x_cont']
# )

# (B) Fancy preprocessing strategy.
# The noise is added to improve the output of QuantileTransformer in some cases.
X_cont_train_numpy = data_numpy["train"]["x_cont"]
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution="normal",
    subsample=10**9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

for part in data_numpy:
    data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

# >>> Label preprocessing.
if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

if X_cat is not None:
    data["train"]["x_cat"] = data["train"]["x_cat"].to(torch.int64)
    data["val"]["x_cat"] = data["val"]["x_cat"].to(torch.int64)
    data["test"]["x_cat"] = data["test"]["x_cat"].to(torch.int64)

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()
        
# The output size.
d_out = n_classes if task_type == "multiclass" else 1

# # NOTE: uncomment to train MLP
mlp = MLP(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=384,
    dropout=0.1,
).to(device)
mlp_optim = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-5)

# # NOTE: uncomment to train ResNet
res = ResNet(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=192,
    d_hidden=None,
    d_hidden_multiplier=2.0,
    dropout1=0.3,
    dropout2=0.0,
).to(device)
res_optim = torch.optim.AdamW(res.parameters(), lr=3e-4, weight_decay=1e-5)

ftt = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **FTTransformer.get_default_kwargs(),
).to(device)
ftt_optim = ftt.make_default_optimizer()

models = [mlp, res, ftt]
optimizers = [mlp_optim, res_optim, ftt_optim]

def apply_model(model, batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, (MLP, ResNet)):
        x_cat_ohe = (
            [
                F.one_hot(column, cardinality)
                for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)
            ]
            if "x_cat" in batch
            else []
        )
        return model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)

    elif isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")


loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(model, part: str, ret=False) -> float:
    model.eval()

    eval_batch_size = 8096
    y_pred = (
        torch.cat(
            [
                apply_model(model, batch)
                for batch in delu.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()
    if ret:
        return y_pred

    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * Y_std)
    return score  # The higher -- the better.


#print(f'Test score before training: {evaluate("test"):.4f}')
print(f'MLP val score before training: {evaluate(mlp, "val"):.4f}')
print(f'ResNet val score before training: {evaluate(res, "val"):.4f}')
print(f'FTTransformer val score before training: {evaluate(ftt, "val"):.4f}')

# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
# n_epochs = 20
# patience = 2
model_names = ['mlp', 'resnet', 'fttransformer']
n_epochs = 1000
patience = 16

batch_size = 25000
epoch_size = math.ceil(len(train_idx) / batch_size)
timer = delu.tools.Timer()
early_stoppings = [delu.tools.EarlyStopping(patience, mode="max"), delu.tools.EarlyStopping(patience, mode="max"), delu.tools.EarlyStopping(patience, mode="max")]
best = [{"val": -math.inf,"epoch": -1,}, {"val": -math.inf,"epoch": -1,}, {"val": -math.inf,"epoch": -1,}]

print(f"Device: {device.type.upper()}")
print("-" * 88 + "\n")
timer.run()
for epoch in range(n_epochs):
    for batch in tqdm(
        delu.iter_batches(data["train"], batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):
        for model, optimizer in zip(models, optimizers):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model(model, batch), batch["y"])
            loss.backward()
            optimizer.step()
    for i, (model, e_s) in enumerate(zip(models, early_stoppings)):
        val_score = evaluate(model, "val")
        print(f"(val) {val_score:.4f} [time] {timer}")

        e_s.update(val_score)
        if e_s.should_stop():
            break

        if val_score > best[i]["val"]:
            #print("🌸 New best epoch! 🌸")
            best[i] = {"val": val_score, "epoch": epoch}
            torch.save(model.state_dict(), f'models/{model_names[i]}_best_val')
        print()

print("\n\nResult:")
print(best)

defaults = [MLP(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=384,
    dropout=0.1,
), ResNet(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=192,
    d_hidden=None,
    d_hidden_multiplier=2.0,
    dropout1=0.3,
    dropout2=0.0,
), FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **FTTransformer.get_default_kwargs(),
)]

import json
for i, model in enumerate(models):
    model_class = defaults[i]
    model_class.load_state_dict(torch.load(f'models/{model_names[i]}_best_val'))
    model_class = model_class.to(device)
    model_data_preds = data_numpy.copy()
    for k in model_data_preds.keys():
        model_data_preds[k]['y_hat'] = evaluate(model_class, k, ret=True)
        
    for k in model_data_preds.keys():
        for j in model_data_preds[k].keys():
            if type(model_data_preds[k][j]) != type([2, 4]):
                model_data_preds[k][j] = model_data_preds[k][j].tolist()
            
    with open(f'predictions/{model_names[i]}_data_predictions.json', 'w') as fp:
        json.dump(model_data_preds, fp)