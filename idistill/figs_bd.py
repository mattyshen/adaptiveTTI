from typing import List

import numpy as np
import pandas as pd

from sklearn import datasets

from imodels import FIGSRegressor, FIGSRegressorCV

class FIGSBDRegressor(FIGSRegressor):
    
    def extract_interactions(self):

        interactions = []

        def traverse_tree(node, current_features, current_depth):

            if node.left is None and node.right is None:
                if self.n_outputs > 1:
                    tree_interactions.append((current_features, np.var(np.abs(node.value))))
                else:
                    tree_interactions.append((current_features, np.max(np.abs(node.value))))
                return
            if node.left is not None:
                current_features_l = current_features.copy()
                current_features_l.append('!c' + str(node.feature+1))
                traverse_tree(node.left, current_features_l.copy(), current_depth=current_depth+1)
            if node.right is not None:
                current_features_r = current_features.copy()
                current_features_r.append('c' + str(node.feature+1))
                traverse_tree(node.right, current_features_r.copy(), current_depth=current_depth+1)

        for tree in self.trees_:
            tree_interactions = []
            traverse_tree(tree, [], current_depth=0)
            interactions.append(tree_interactions)

        self.interactions = interactions
        
    def _find_closest_keys(self, dictionary, targets):
        keys = np.array(list(dictionary.keys()))
        targets = np.array(targets)
        diffs = np.abs(keys[:, None] - targets)
        closest_key_indices = np.argmin(diffs, axis=0)
        closest_keys = keys[closest_key_indices]

        return closest_keys

    def extract_atti(self, X, number_of_top_paths=0):
        
        if not hasattr(self, "interactions"):
            self.extract_interactions()

        figs_dict = {}
        for i, tree in enumerate(self.interactions):
            tree_dict = {}
            for path, var in tree:
                tree_dict[var] = path
            figs_dict[i] = tree_dict

        test_pred_intervention = self.predict(X, by_tree = True)

        concepts_to_edit = [[] for _ in range(X.shape[0])]
        
        if self.n_outputs > 1:
            imp_heur = np.var(np.abs(test_pred_intervention), axis = 1)
        else:
            imp_heur = np.max(np.abs(test_pred_intervention), axis = 1)

        concepts = np.array([self._find_closest_keys(figs_dict[i], imp_heur[:, i]) for i in range(imp_heur.shape[1])])
        orderings_of_interventions = np.argsort(concepts.T, axis = 1)[:, ::-1]
        imp_heur_of_orderings_of_interventions = np.sort(concepts.T, axis = 1)[:, ::-1]

        if number_of_top_paths == 0:
            r = range(orderings_of_interventions.shape[1])
        else:
            r = range(number_of_top_paths)

        for t in r:
            for i, l in enumerate(orderings_of_interventions[:, t]):
                new_list = []
                for c in figs_dict[l][imp_heur_of_orderings_of_interventions[i, t]]:
                    new_list.append(int(c[1:])-1 if c[0] != '!' else int(c[2:])-1)
                concepts_to_edit[i].append(new_list)
                
        return concepts_to_edit

class FIGSBDRegressorCV(FIGSRegressorCV):
    def extract_interactions(self):

        interactions = []

        def traverse_tree(node, current_features, current_depth):

            if node.left is None and node.right is None:
                if self.n_outputs > 1:
                    tree_interactions.append((current_features, np.var(np.abs(node.value))))
                else:
                    tree_interactions.append((current_features, np.max(np.abs(node.value))))
                return
            if node.left is not None:
                current_features_l = current_features.copy()
                current_features_l.append('!c' + str(node.feature+1))
                traverse_tree(node.left, current_features_l.copy(), current_depth=current_depth+1)
            if node.right is not None:
                current_features_r = current_features.copy()
                current_features_r.append('c' + str(node.feature+1))
                traverse_tree(node.right, current_features_r.copy(), current_depth=current_depth+1)

        for tree in self.trees_:
            tree_interactions = []
            traverse_tree(tree, [], current_depth=0)
            interactions.append(tree_interactions)

        self.interactions = interactions
        
    def _find_closest_keys(self, dictionary, targets):
        keys = np.array(list(dictionary.keys()))
        targets = np.array(targets)
        diffs = np.abs(keys[:, None] - targets)
        closest_key_indices = np.argmin(diffs, axis=0)
        closest_keys = keys[closest_key_indices]

        return closest_keys

    def extract_atti(self, X, number_of_top_paths=0):
        
        if not hasattr(self, "interactions"):
            self.extract_interactions()

        figs_dict = {}
        for i, tree in enumerate(self.interactions):
            tree_dict = {}
            for path, var in tree:
                tree_dict[var] = path
            figs_dict[i] = tree_dict

        test_pred_intervention = self.predict(X, by_tree = True)

        concepts_to_edit = [[] for _ in range(X.shape[0])]
        
        if self.n_outputs > 1:
            imp_heur = np.var(np.abs(test_pred_intervention), axis = 1)
        else:
            imp_heur = np.max(np.abs(test_pred_intervention), axis = 1)

        concepts = np.array([self._find_closest_keys(figs_dict[i], imp_heur[:, i]) for i in range(imp_heur.shape[1])])
        orderings_of_interventions = np.argsort(concepts.T, axis = 1)[:, ::-1]
        imp_heur_of_orderings_of_interventions = np.sort(concepts.T, axis = 1)[:, ::-1]

        if number_of_top_paths == 0:
            r = range(orderings_of_interventions.shape[1])
        else:
            r = range(number_of_top_paths)

        for t in r:
            for i, l in enumerate(orderings_of_interventions[:, t]):
                new_list = []
                for c in figs_dict[l][imp_heur_of_orderings_of_interventions[i, t]]:
                    new_list.append(int(c[1:])-1 if c[0] != '!' else int(c[2:])-1)
                concepts_to_edit[i].append(new_list)
                
        return concepts_to_edit


        

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