from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import l0learn
import l0bnb
from itertools import chain, combinations
from scipy.special import expit

from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 

from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
from imodels.util.arguments import check_fit_arguments
from imodels.util.data_util import encode_categories


class Node:
    def __init__(
        self,
        feature: int = None,
        threshold: int = None,
        value=None,
        value_sklearn=None,
        idxs=None,
        is_root: bool = False,
        left=None,
        impurity: float = None,
        impurity_reduction: float = None,
        tree_num: int = None,
        node_id: int = None,
        right=None,
    ):
        """Node class for splitting"""

        # split or linear
        self.is_root = is_root
        self.idxs = idxs
        self.tree_num = tree_num
        self.node_id = None
        self.feature = feature
        self.impurity = impurity
        self.impurity_reduction = impurity_reduction
        self.value_sklearn = value_sklearn

        # different meanings
        self.value = value  # for split this is mean, for linear this is weight

        # split-specific
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_temp = None
        self.right_temp = None

    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        if self.is_root:
            return f"X_{self.feature} <= {self.threshold:0.3f} (Tree #{self.tree_num} root)"
        elif self.left is None and self.right is None:
            return f"Val: {self.value[0][0]:0.3f} (leaf)"
        else:
            return f"X_{self.feature} <= {self.threshold:0.3f} (split)"

    def print_root(self, y):
        try:
            one_count = pd.Series(y).value_counts()[1.0]
        except KeyError:
            one_count = 0
        one_proportion = (
            f" {one_count}/{y.shape[0]} ({round(100 * one_count / y.shape[0], 2)}%)"
        )

        if self.is_root:
            return f"X_{self.feature} <= {self.threshold:0.3f}" + one_proportion
        elif self.left is None and self.right is None:
            return f"ΔRisk = {self.value[0][0]:0.2f}" + one_proportion
        else:
            return f"X_{self.feature} <= {self.threshold:0.3f}" + one_proportion

    def __repr__(self):
        return self.__str__()


class distillFIGS(BaseEstimator):
    """distillFIGS"""

    def __init__(
        self,
        reg: str = 'l1',
        lam: float = 1.0,
        ratio: float = 1.0,
    ):
        """
        Params
        ------
        reg: str
            Type of regularization used when selecting features interactions and trees
        lam: float
            Strength of regularization parameter
        ratio: float
            Ratio of L1 and L2 penalization for Elastic Net type regularization
        """
        super().__init__()
        self.reg = reg
        self.lam = lam
        self.ratio = ratio

    def _init_decision_function(self):
        """Sets decision function based on _estimator_type"""
        # used by sklearn GridSearchCV, BaggingClassifier
        if isinstance(self, ClassifierMixin):

            def decision_function(x):
                return self.predict_proba(x)[:, 1]

        elif isinstance(self, RegressorMixin):
            decision_function = self.predict

    def _construct_node_with_stump(
        self,
        X,
        y,
        idxs,
        tree_num,
        sample_weight=None,
        compare_nodes_with_sample_weight=True,
        max_features=None,
    ):
        """
        Params
        ------
        compare_nodes_with_sample_weight: Deprecated
            If this is set to true and sample_weight is passed, use sample_weight to compare nodes
            Otherwise, use sample_weight only for picking a split given a particular node
        """

        # array indices
        SPLIT = 0
        LEFT = 1
        RIGHT = 2

        # fit stump
        stump = tree.DecisionTreeRegressor(max_depth=1, max_features=max_features)
        sweight = None
        if sample_weight is not None:
            sweight = sample_weight[idxs]
        stump.fit(X[idxs], y[idxs], sample_weight=sweight)

        # these are all arrays, arr[0] is split node
        # note: -2 is dummy
        feature = stump.tree_.feature
        threshold = stump.tree_.threshold

        impurity = stump.tree_.impurity
        n_node_samples = stump.tree_.n_node_samples
        value = stump.tree_.value

        # no split
        if len(feature) == 1:
            # print('no split found!', idxs.sum(), impurity, feature)
            return Node(
                idxs=idxs,
                value=value[SPLIT],
                tree_num=tree_num,
                feature=feature[SPLIT],
                threshold=threshold[SPLIT],
                impurity=impurity[SPLIT],
                impurity_reduction=None,
            )

        # manage sample weights
        idxs_split = X[:, feature[SPLIT]] <= threshold[SPLIT]
        idxs_left = idxs_split & idxs
        idxs_right = ~idxs_split & idxs
        if sample_weight is None:
            n_node_samples_left = n_node_samples[LEFT]
            n_node_samples_right = n_node_samples[RIGHT]
        else:
            n_node_samples_left = sample_weight[idxs_left].sum()
            n_node_samples_right = sample_weight[idxs_right].sum()
        n_node_samples_split = n_node_samples_left + n_node_samples_right

        # calculate impurity
        impurity_reduction = (
            impurity[SPLIT]
            - impurity[LEFT] * n_node_samples_left / n_node_samples_split
            - impurity[RIGHT] * n_node_samples_right / n_node_samples_split
        ) * n_node_samples_split

        node_split = Node(
            idxs=idxs,
            value=value[SPLIT],
            tree_num=tree_num,
            feature=feature[SPLIT],
            threshold=threshold[SPLIT],
            impurity=impurity[SPLIT],
            impurity_reduction=impurity_reduction,
        )
        # print('\t>>>', node_split, 'impurity', impurity, 'num_pts', idxs.sum(), 'imp_reduc', impurity_reduction)

        # manage children
        node_left = Node(
            idxs=idxs_left,
            value=value[LEFT],
            impurity=impurity[LEFT],
            tree_num=tree_num,
        )
        node_right = Node(
            idxs=idxs_right,
            value=value[RIGHT],
            impurity=impurity[RIGHT],
            tree_num=tree_num,
        )
        node_split.setattrs(
            left_temp=node_left,
            right_temp=node_right,
        )
        return node_split

    def fit(
        self,
        X,
        y=None,
        feature_names=None,
        verbose=False,
        sample_weight=None,
        categorical_features=None,
    ):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """
        
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        X = pd.DataFrame(X, columns = feature_names)
        
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            
        # X has to be binary: {0,1}. TODO: Investigate if binary: {-1, 1} has the same interpretation
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        def compute_subset_product(subset, data):
            return pd.Series(1, index = data.index) if not subset else data[list(subset)].product(axis=1)

        column_powerset = powerset(feature_names)
        
        Chi = pd.DataFrame()

        for subset in column_powerset:
            Chi[subset] = compute_subset_product(subset, X)
            
        #TODO: adjust for init argument
        interaction_selector = Lasso(fit_intercept = False, alpha = self.lam)
        interaction_selector.fit(Chi, y)
        
        #TODO: make sure all i_s have a coef_ AND handle rounding cases
        Chi_s = Chi.loc[:, interaction_selector.coef_ != 0]
        
        def split_constructor(columns, beta):
            if len(columns) == 0:
                return Node(is_root = True,
                           feature = 'Intercept',
                           value = np.array([[beta]]),
                           threshold = 0.5,
                           left = None,
                           right = None)
            elif len(columns) == 1:
                return Node(is_root = True,
                           feature = columns[0],
                           value = np.array([[beta]]),
                           threshold = 0.5,
                           left = Node(is_root = False,
                                      value = np.array([[0]])),
                           right = Node(is_root = False,
                                      value = np.array([[beta]])))
            else:
                n = Node(is_root = False,
                           feature = columns[0],
                           value = None,
                           threshold = 0.5,
                           left = Node(is_root = False,
                                      value = np.array([[0]])))
                n.setattrs(right = split_constructor(columns[1:], beta))
                return n
            
        self.trees_ = []
        #TODO: handle rounding cases
        for c, beta in zip(Chi_s.columns, list(filter(lambda num: num != 0, interaction_selector.coef_))):
            self.trees_.append(split_constructor(c, beta))
        
        return self

    def _tree_to_str(self, root: Node, prefix=""):
        if root is None:
            return ""
        elif root.threshold is None:
            return ""
        pprefix = prefix + "\t"
        return (
            prefix
            + str(root)
            + "\n"
            + self._tree_to_str(root.left, pprefix)
            + self._tree_to_str(root.right, pprefix)
        )

    def _tree_to_str_with_data(self, X, y, root: Node, prefix=""):
        if root is None:
            return ""
        elif root.threshold is None:
            return ""
        pprefix = prefix + "\t"
        left = X[:, root.feature] <= root.threshold
        return (
            prefix
            + root.print_root(y)
            + "\n"
            + self._tree_to_str_with_data(X[left], y[left], root.left, pprefix)
            + self._tree_to_str_with_data(X[~left], y[~left], root.right, pprefix)
        )

    def __str__(self):
        if not hasattr(self, "trees_"):
            s = self.__class__.__name__
            s += "("
            s += "max_rules="
            s += repr(self.max_rules)
            s += ")"
            return s
        else:
            s = "> ------------------------------\n"
            s += "> distillFIGS-distilled Fast Interpretable Greedy-Tree Sums:\n"
            s += '> \tPredictions are made by summing the "Val" reached by traversing each tree.\n'
            s += "> \tFor classifiers, a sigmoid function is then applied to the sum.\n"
            s += "> ------------------------------\n"
            s += "\n\t+\n".join([self._tree_to_str(t) for t in self.trees_])
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                for i in range(len(self.feature_names_))[::-1]:
                    s = s.replace(f"X_{i}", self.feature_names_[i])
            return s

    def print_tree(self, X, y, feature_names=None):
        s = "------------\n" + "\n\t+\n".join(
            [self._tree_to_str_with_data(X, y, t) for t in self.trees_]
        )
        if feature_names is None:
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                feature_names = self.feature_names_
        if feature_names is not None:
            for i in range(len(feature_names))[::-1]:
                s = s.replace(f"X_{i}", feature_names[i])
        return s

    def predict(self, X, categorical_features=None):
        if hasattr(self, "_encoder"):
            X = self._encode_categories(X, categorical_features=categorical_features)
        #TODO: put this back in, but needs to stay as DF
        #X = check_array(X)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            return (preds > 0.5).astype(int)
        

    def predict_proba(self, X, categorical_features=None, use_clipped_prediction=False):
        """Predict probability for classifiers:
        Default behavior is to constrain the outputs to the range of probabilities, i.e. 0 to 1, with a sigmoid function.
        Set use_clipped_prediction=True to use prior behavior of clipping between 0 and 1 instead.
        """
        if hasattr(self, "_encoder"):
            X = self._encode_categories(X, categorical_features=categorical_features)
        X = check_array(X)
        if isinstance(self, RegressorMixin):
            return NotImplemented
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if use_clipped_prediction:
            # old behavior, pre v1.3.9
            # constrain to range of probabilities by clipping
            preds = np.clip(preds, a_min=0.0, a_max=1.0)
        else:
            # constrain to range of probabilities with a sigmoid function
            preds = expit(preds)
        return np.vstack((1 - preds, preds)).transpose()

    def _predict_tree(self, root: Node, X):
        """Predict for a single tree"""

        def _predict_tree_single_point(root: Node, x):
            if root.left is None and root.right is None:
                return root.value[0, 0]
            left = x[root.feature] <= root.threshold
            if left:
                if root.left is None:  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.left, x)
            else:
                if (root.right is None):  # we don't actually have to worry about this case
                    return root.value
                else:
                    return _predict_tree_single_point(root.right, x)

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = _predict_tree_single_point(root, X.iloc[i])
        return preds

    @property
    def feature_importances_(self):
        """Gini impurity-based feature importances"""
        check_is_fitted(self)

        avg_feature_importances = np.mean(
            self.importance_data_, axis=0, dtype=np.float64
        )

        return avg_feature_importances / np.sum(avg_feature_importances)

    def plot(
        self,
        cols=2,
        feature_names=None,
        filename=None,
        label="all",
        impurity=False,
        tree_number=None,
        dpi=150,
        fig_size=None,
    ):
        is_single_tree = len(self.trees_) < 2 or tree_number is not None
        n_cols = int(cols)
        n_rows = int(np.ceil(len(self.trees_) / n_cols))

        if feature_names is None:
            if hasattr(self, "feature_names_") and self.feature_names_ is not None:
                feature_names = self.feature_names_

        n_plots = int(len(self.trees_)) if tree_number is None else 1
        fig, axs = plt.subplots(n_plots, dpi=dpi)
        if fig_size is not None:
            fig.set_size_inches(fig_size, fig_size)

        n_classes = 1 if isinstance(self, RegressorMixin) else 2
        ax_size = int(len(self.trees_))
        for i in range(n_plots):
            r = i // n_cols
            c = i % n_cols
            if not is_single_tree:
                ax = axs[i]
            else:
                ax = axs
            try:
                dt = extract_sklearn_tree_from_figs(
                    self, i if tree_number is None else tree_number, n_classes
                )
                plot_tree(
                    dt,
                    ax=ax,
                    feature_names=feature_names,
                    label=label,
                    impurity=impurity,
                )
            except IndexError:
                ax.axis("off")
                continue
            ttl = f"Tree {i}" if n_plots > 1 else f"Tree {tree_number}"
            ax.set_title(ttl)
        if filename is not None:
            plt.savefig(filename)
            return
        plt.show()


class distillFIGSRegressor(distillFIGS, RegressorMixin):
    ...

"""
class FIGSClassifier(FIGS, ClassifierMixin):
    ...


class FIGSCV:
    def __init__(
        self,
        figs,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring=None,
        *args,
        **kwargs,
    ):
        if len(n_rules_list) != len(n_trees_list):
            raise ValueError(
                f"len(n_rules_list) = {len(n_rules_list)} != len(n_trees_list) = {len(n_trees_list)}"
            )

        self._figs_class = figs
        self.n_rules_list = np.array(n_rules_list)
        self.n_trees_list = np.array(n_trees_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        self.scores_ = []
        for _i, n_rules in enumerate(self.n_rules_list):
            est = self._figs_class(max_rules=n_rules, max_trees=self.n_trees_list[_i])
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)
            if len(self.scores_) == 0:
                self.figs = est
            elif mean_score > np.max(self.scores_):
                self.figs = est

            self.scores_.append(mean_score)
        self.figs.fit(X=X, y=y)

    def predict_proba(self, X):
        return self.figs.predict_proba(X)

    def predict(self, X):
        return self.figs.predict(X)

    @property
    def max_rules(self):
        return self.figs.max_rules

    @property
    def max_trees(self):
        return self.figs.max_trees


class FIGSRegressorCV(FIGSCV):
    def __init__(
        self,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring="r2",
        *args,
        **kwargs,
    ):
        super(FIGSRegressorCV, self).__init__(
            figs=FIGSRegressor,
            n_rules_list=n_rules_list,
            n_trees_list=n_trees_list,
            cv=cv,
            scoring=scoring,
            *args,
            **kwargs,
        )


class FIGSClassifierCV(FIGSCV):
    def __init__(
        self,
        n_rules_list: List[int] = [6, 12, 24, 30, 50],
        n_trees_list: List[int] = [5, 5, 5, 5, 5],
        cv: int = 3,
        scoring="accuracy",
        *args,
        **kwargs,
    ):
        super(FIGSClassifierCV, self).__init__(
            figs=FIGSClassifier,
            n_rules_list=n_rules_list,
            n_trees_list=n_trees_list,
            cv=cv,
            scoring=scoring,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    from sklearn import datasets

    X_cls, Y_cls = datasets.load_breast_cancer(return_X_y=True)
    X_reg, Y_reg = datasets.make_friedman1(100)

    categories = ["cat", "dog", "bird", "fish"]
    categories_2 = ["bear", "chicken", "cow"]

    X_cat = pd.DataFrame(X_reg)
    X_cat["pet1"] = np.random.choice(categories, size=(100, 1))
    X_cat["pet2"] = np.random.choice(categories_2, size=(100, 1))

    # X_cat.columns[-1] = "pet"
    Y_cat = Y_reg

    est = FIGSRegressor(max_rules=10)
    est.fit(X_cat, Y_cat, categorical_features=["pet1", "pet2"])
    est.predict(X_cat, categorical_features=["pet1", "pet2"])
    est.plot(tree_number=1)

    est = FIGSClassifier(max_rules=10)
    # est.fit(X_cls, Y_cls, sample_weight=np.arange(0, X_cls.shape[0]))
    est.fit(X_cls, Y_cls, sample_weight=[1] * X_cls.shape[0])
    est.predict(X_cls)

    est = FIGSRegressorCV()
    est.fit(X_reg, Y_reg)
    est.predict(X_reg)
    print(est.max_rules)
    est.figs.plot(tree_number=0)

    est = FIGSClassifierCV()
    est.fit(X_cls, Y_cls)
    est.predict(X_cls)
    print(est.max_rules)
    est.figs.plot(tree_number=0)

# %%
"""