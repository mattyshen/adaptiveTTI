import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu 
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

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

class TabDLM:
    def __init__(self, 
                 model_type, 
                 task_type, 
                 n_epochs=100,
                 patience=16,
                 batch_size=256,
                 n_classes=1,
                 val_prop=0.2,
                 model_params={}, 
                 device=None, 
                 cuda=0,
                 verbose=True,
                 seed=0):
        if device is None:
            self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model_type = model_type
        self.task_type = task_type
        self.n_classes = n_classes 
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_prop = val_prop
        self.model_params = model_params

        self.loss_fn = (
                        F.binary_cross_entropy_with_logits
                        if task_type in ["binclass", "binary"]
                        else F.cross_entropy
                        if task_type == "multiclass"
                        else F.mse_loss
                        )
        self.seed = seed
        
        self.preprocessing = None

    def fit(self, X_train, y_train):
        
        if self.task_type == "regression":
            Y = y_train.to_numpy().astype(np.float32)
        else:
            assert self.n_classes is not None
            Y = y_train.to_numpy().astype(np.int64)
            assert set(y_train.tolist()) == set(
                range(self.n_classes)
            ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"
            
        train_idx, val_idx = sklearn.model_selection.train_test_split(np.arange(len(Y)), train_size=self.val_prop, random_state=self.seed)
        
        X_cont = X_train.select_dtypes(include=['float64', 'float32']).astype(np.float32)
        X_cat = X_train.drop(columns = X_cont.columns).astype(np.float32)
        
        data_numpy = {
            "train": {"x_cont": X_cont.iloc[train_idx, :].to_numpy().astype(np.float32), "y": Y[train_idx]},
            "val": {"x_cont": X_cont.iloc[val_idx, :].to_numpy().astype(np.float32), "y": Y[val_idx]},
        }
        if len(X_cat.columns) > 0:
            data_numpy["train"]["x_cat"] = X_cat.iloc[train_idx, :].to_numpy()
            data_numpy["val"]["x_cat"] = X_cat.iloc[val_idx, :].to_numpy()
        if len(X_cont.columns) > 0:
            X_cont_train_numpy = data_numpy["train"]["x_cont"]
            noise = (
                np.random.default_rng(0)
                .normal(0.0, 1e-5, X_cont_train_numpy.shape)
                .astype(X_cont_train_numpy.dtype)
            )
            self.preprocessing = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
                output_distribution="normal",
                subsample=10**9,
            ).fit(X_cont_train_numpy + noise)

            del X_cont_train_numpy

            for part in data_numpy:
                data_numpy[part]["x_cont"] = self.preprocessing.transform(data_numpy[part]["x_cont"])

        # >>> Label preprocessing.
        if self.task_type == "regression":
            self.Y_mean = data_numpy["train"]["y"].mean().item()
            self.Y_std = data_numpy["train"]["y"].std().item()
            for part in data_numpy:
                data_numpy[part]["y"] = (data_numpy[part]["y"] - self.Y_mean) / self.Y_std

        # >>> Convert data to tensors.
        data = {
            part: {k: torch.as_tensor(v, device=self.device) for k, v in data_numpy[part].items()}
            for part in data_numpy
        }

        d_out = self.n_classes if self.task_type == "multiclass" else 1

        self.n_cont_features = len(X_cont.columns)
        cat_unique_vals = [list(X_cat[c].value_counts().index) for c in X_cat.columns]
        self.cat_cardinalities = [len(c) for c in cat_unique_vals]
        
        self.all_cat_bin = set([x for xs in cat_unique_vals for x in xs]) == {0, 1}
                
        if self.model_type == 'MLP':
            if self.all_cat_bin:
                self.model = MLP(d_in=self.n_cont_features + len(self.cat_cardinalities),
                                     d_out=d_out,
                                     n_blocks=2,
                                     d_block=384,
                                     dropout=0.1,
                                     **self.model_params)
            else:
                self.model = MLP(d_in=self.n_cont_features + sum(self.cat_cardinalities),
                                     d_out=d_out,
                                     n_blocks=2,
                                     d_block=384,
                                     dropout=0.1,
                                     **self.model_params)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        elif self.model_type == 'ResNet':
            if self.all_cat_bin:
                self.model = ResNet(d_in=self.n_cont_features + len(self.cat_cardinalities),
                                    d_out=d_out,
                                    n_blocks=2,
                                    d_block=192,
                                    d_hidden=None,
                                    d_hidden_multiplier=2.0,
                                    dropout1=0.11,
                                    dropout2=0.1,
                                    **self.model_params)
            else:
                self.model = ResNet(d_in=self.n_cont_features + sum(self.cat_cardinalities),
                                    d_out=d_out,
                                    n_blocks=2,
                                    d_block=192,
                                    d_hidden=None,
                                    d_hidden_multiplier=2.0,
                                    dropout1=0.11,
                                    dropout2=0.1,
                                    **self.model_params)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        elif self.model_type == 'FTTransformer':
            if len(self.model_params.keys()) == 0:
                self.model_params = FTTransformer.get_default_kwargs()
            self.model = FTTransformer(n_cont_features=self.n_cont_features,
                                       cat_cardinalities=self.cat_cardinalities,
                                       d_out=d_out,
                                       **self.model_params)
            self.optimizer = self.model.make_default_optimizer()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(self.device)
        
        epoch_size = math.ceil(len(train_idx) / self.batch_size)
        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(self.patience, mode="max")
        best = {
            "val": -math.inf,
            "epoch": -1,
        }

        print(f"Device: {self.device.type.upper()}")
        print("-" * 88 + "\n")
        timer.run()
        for epoch in range(self.n_epochs):
            for batch in tqdm(
                delu.iter_batches(data["train"], self.batch_size, shuffle=True),
                desc=f"Epoch {epoch}",
                total=epoch_size,
                disable=(not self.verbose)
            ):
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self._apply_model(batch), batch["y"])
                loss.backward()
                self.optimizer.step()

            val_score = self._evaluate(data, "val")
            
            if self.verbose:
                print(f"(val) {val_score:.4f} [time] {timer}")

            early_stopping.update(val_score)
            if early_stopping.should_stop():
                break

            if val_score > best["val"]:
                if self.verbose:
                    print("New best epoch!")
                best = {"val": val_score, "epoch": epoch}
            if self.verbose:
                print()
        if self.verbose:
            print("\n\nResult:")
            print(best)
        best['train'] = self._evaluate(data, "train")
        self.best = best
        
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        
    def predict(self, X):
        self.model.eval()
        
        X_cont = X.select_dtypes(include=['float64', 'float32']).astype(np.float32)
        X_cat = X.drop(columns = X_cont.columns).astype(np.float32)
        
        data_numpy = {
            "test": {"x_cont": X_cont.to_numpy().astype(np.float32)},
        }
        
        if len(X_cat.columns) > 0:
            data_numpy["test"]["x_cat"] = X_cat.to_numpy().astype(np.float32)
        if self.preprocessing is not None:
            data_numpy['test']["x_cont"] = self.preprocessing.transform(data_numpy['test']["x_cont"])

        # >>> Label preprocessing.
        # if self.task_type == "regression":
        #     self.Y_mean = data_numpy["train"]["y"].mean().item()
        #     self.Y_std = data_numpy["train"]["y"].std().item()
        #     for part in data_numpy:
        #         data_numpy[part]["y"] = (data_numpy[part]["y"] - self.Y_mean) / self.Y_std

        # >>> Convert data to tensors.
        data = {
            'test': {k: torch.as_tensor(v, device=self.device) for k, v in data_numpy['test'].items()}
        }
        
        with torch.no_grad():
            predictions = (
                    torch.cat(
                        [
                            self._apply_model(batch)
                            for batch in delu.iter_batches(data['test'], self.batch_size)
                        ]
                    )
                    .cpu()
                    .numpy()
                )
        if self.task_type == "regression":
            return predictions * self.Y_std + self.Y_mean
        else:
            return predictions

    def _evaluate(self, data, part):
        with torch.no_grad():
            self.model.eval()

            eval_batch_size = 2048
            y_pred = (
                torch.cat(
                    [
                        self._apply_model(batch)
                        for batch in delu.iter_batches(data[part], eval_batch_size)
                    ]
                )
                .cpu()
                .numpy()
            )
            y_true = data[part]["y"].cpu().numpy()

            if self.task_type == "binclass":
                y_pred = np.round(scipy.special.expit(y_pred))
                score = sklearn.metrics.accuracy_score(y_true, y_pred)
            elif self.task_type == "multiclass":
                y_pred = y_pred.argmax(1)
                score = sklearn.metrics.accuracy_score(y_true, y_pred)
            else:
                assert self.task_type == "regression"
                score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * self.Y_std)
            return score  # The higher -- the better.
        
    def _apply_model(self, batch: Dict[str, Tensor]) -> Tensor:
        if isinstance(self.model, (MLP, ResNet)):
            if self.all_cat_bin:
                x_cat_ohe = (
                [
                    column
                    for column, cardinality in zip(batch["x_cat"].T, self.cat_cardinalities)
                ]
                if "x_cat" in batch
                else []
            )
            else:
                x_cat_ohe = (
                    [
                        F.one_hot(column, cardinality)
                        for column, cardinality in zip(batch["x_cat"].T, self.cat_cardinalities)
                    ]
                    if "x_cat" in batch
                    else []
                )
            return self.model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)

        elif isinstance(self.model, FTTransformer):
            if self.n_cont_features != 0:
                return self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
            else:
                return self.model(None, batch.get("x_cat").long()).squeeze(-1)
        else:
            raise RuntimeError(f"Unknown model type: {type(self.model)}")