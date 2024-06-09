# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import pandas as pd
import os
import sys
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

sys.path.append('..')

from interpretDistill.fourierDistill import *
from interpretDistill.binaryTransformer import *

#depth
dt_depth = int(sys.argv[1])
#bit
bit_boolean = str(sys.argv[2])

if bit_boolean == 'bit':
    word_access = 'bit'
    bit_boolean = True
else:
    word_access = 'bin'
    bit_boolean = False
    
print(dt_depth, bit_boolean, word_access)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

delu.random.seed(0)

# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]

task_type: TaskType = "regression"
n_classes = None

for cur_seed in [0, 1, 2, 3, 4]:
    print(f'current seed: {cur_seed}')
    dataset = sklearn.datasets.fetch_california_housing(as_frame = True)
    X = dataset["data"]
    Y = dataset["target"]
    # >>> Split the dataset.
    all_idx = np.arange(len(Y))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state = cur_seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8, random_state = cur_seed
    )

    X_b = {}
    bt = BinaryTransformer(depth = dt_depth, bit = bit_boolean)
    X_b['train'] = bt.fit_and_transform(X.loc[train_idx, :], Y.loc[train_idx])
    X_b['val'] = bt.transform(X.loc[val_idx, :])
    X_b['test'] = bt.transform(X.loc[test_idx, :])
    
    print(X_b['train'].shape)


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

    # >>> Feature preprocessing.
    # NOTE
    # The choice between preprocessing strategies depends on a task and a model.

    # (A) Simple preprocessing strategy.
    # preprocessing = sklearn.preprocessing.StandardScaler().fit(
    #     data_numpy['train']['x_cont']
    # )

    # (B) Fancy preprocessing strategy.
    # The noise is added to improve the output of QuantileTransformer in some cases.
    # X_cont_train_numpy = data_numpy["train"]["x_cont"]
    # noise = (
    #     np.random.default_rng(0)
    #     .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    #     .astype(X_cont_train_numpy.dtype)
    # )
    # preprocessing = sklearn.preprocessing.QuantileTransformer(
    #     n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    #     output_distribution="normal",
    #     subsample=10**9,
    # ).fit(X_cont_train_numpy + noise)
    # del X_cont_train_numpy

    # for part in data_numpy:
    #     data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

    # >>> Label preprocessing.
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

    # The output size.
    d_out = n_classes if task_type == "multiclass" else 1

    ftt_b = FTTransformer(
        n_cont_features=0,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(),
    ).to(device)
    ftt_b_optim = ftt_b.make_default_optimizer()

    models = [ftt_b]
    optimizers = [ftt_b_optim]

    def apply_model(model, batch: Dict[str, Tensor]) -> Tensor:
        if isinstance(model, (MLP, ResNet)):
            x_cat_ohe = ([F.one_hot(column, cardinality) for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)])
            return model(torch.column_stack(x_cat_ohe).to(torch.float32)).squeeze(-1)

        elif isinstance(model, FTTransformer):
            return model(None, batch.get("x_cat")).squeeze(-1)

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
    def evaluate(model, part: str, ret = False) -> float:
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



    print(f'FTTransformer val score before training: {evaluate(ftt_b, "val"):.4f}')

    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2

    model_names = [f'ftt_{word_access}_depth{dt_depth}']

    n_epochs = 100
    patience = 16

    batch_size = 256
    epoch_size = math.ceil(len(train_idx) / batch_size)
    timer = delu.tools.Timer()
    early_stoppings = [delu.tools.EarlyStopping(patience, mode="max"), delu.tools.EarlyStopping(patience, mode="max")]
    best = [{"val": -math.inf,"epoch": -1,}, {"val": -math.inf,"epoch": -1,}]

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
                best[i] = {"val": val_score, "epoch": epoch}
                torch.save(model.state_dict(), f'models/{model_names[i]}_seed{cur_seed}')
            print()

    print("\n\nResult:")
    print(best)

    defaults = [FTTransformer(
        n_cont_features=0,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(),
    )]

    import json

    for i, model in enumerate(models):
        model_class = defaults[i]
        model_class.load_state_dict(torch.load(f'models/{model_names[i]}_seed{cur_seed}'))
        model_class = model_class.to(device)
        model_data_preds = data_numpy.copy()
        for k in model_data_preds.keys():
            model_data_preds[k]['y_hat'] = evaluate(model_class, k, ret=True)

        for k in model_data_preds.keys():
            for j in model_data_preds[k].keys():
                if type(model_data_preds[k][j]) != type([2, 4]):
                    model_data_preds[k][j] = model_data_preds[k][j].tolist()

        with open(f'predictions/{model_names[i]}_preds_seed{cur_seed}.json', 'w') as fp:
            json.dump(model_data_preds, fp)