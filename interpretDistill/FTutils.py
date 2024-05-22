import numpy as np
import pandas as pd
from itertools import chain, combinations

class TrieNode:
    def __init__(self, features = set()):
        self.left = None
        self.right = None
        self.features = features

def powerset_pruned(features, removals, sort = True):

    root = TrieNode()
    stack = [(root, 0)]
    final_features = []
    removal_vc = pd.Series([i for j in removals for i in j]).value_counts().index.to_list()
    features = removal_vc + list(set(features.to_list()) - set(removal_vc))
    
    while stack:
        
        node, depth = stack.pop()
        if depth == len(features):
            final_features.append(node.features.copy())
            continue
            
        potential_set = node.features.copy()
        potential_set.add(features[depth])
        
        if any([len(set(potential_set).intersection(s)) >= 2 for s in removals]):
            node.left = TrieNode(node.features.copy())
            stack.append((node.left, depth + 1))
        else:
            node.left = TrieNode(node.features.copy())
            node.right = TrieNode(potential_set)
            stack.append((node.right, depth + 1))
            stack.append((node.left, depth + 1))
            
    if sort:
        return sorted(final_features, key=len)
    return final_features

memo = {}

def compute_subset_product(subset, data):
    global memo
    
    if subset in memo:
        return memo[subset]
    
    result = pd.Series(1, index=data.index) if not subset else _compute_subset_product_helper(subset, data)
    
    memo[subset] = result
    
    return result

def _compute_subset_product_helper(subset, data):
    global memo
    
    if len(subset) == 1:
        return data[subset[0]]
    
    if subset in memo:
        return memo[subset]
    
    if set(subset) in [set(k) for k in memo.keys()]:
        
        shuffled_permutations = itertools.permutations(subset)

        for perm in shuffled_permutations:
            if perm in my_dict:
                return memo[perm]
    
    overlaps = [(k, len(set(subset).intersection(set(k)))) for k in memo.keys()]
    
    if len(overlaps) == 0 or max(overlaps, key=lambda x: x[1])[1] == 0:
        result = pd.Series(1, index = data.index) if not subset else data[list(subset)].product(axis=1)
        memo[subset] = result
        return result
    else:
        max_subset = max(overlaps, key=lambda x: x[1])[0]
        max_subset_product = memo[max_subset]
        remaining_subset = tuple(set(subset) - set(max_subset))

        remaining_subset_product = _compute_subset_product_helper(remaining_subset, data)
        result = max_subset_product * remaining_subset_product

        memo[subset] = result

        return result
    
def compute_subset_product_naive(subset, data):
    return pd.Series(1, index = data.index) if not subset else data[list(subset)].product(axis=1)

def binary_map(arr):
    unique_values = list(set(arr))
    if 1 in unique_values and 0 in unique_values:
        return {1: 1, 0: 0}
    elif 1 in unique_values:
        unique_values.remove(1)
        return {np.round(unique_values[0], 3): 0, 1: 1}
    elif 0 in unique_values:
        unique_values.remove(0)
        return {0: 0, np.round(unique_values[0], 3): 1}
    else:
        return {np.round(unique_values[0], 3): 0, np.round(unique_values[1], 3): 1}
    
def bit_repr(column, K):
    
    unique_values = sorted(column.unique())
    
    assert len(unique_values) <= 2**K
    
    mapping = {val: i for i, val in enumerate(unique_values)}
    column_mapped = column.map(mapping)

    binary_representations = column_mapped.apply(lambda x: np.binary_repr(x, width=K)).apply(lambda x: pd.Series(list(x)))
    new_columns = [f'{column.name}_bit{i}' for i in range(K)]
    binary_representations.columns = new_columns

    return binary_representations, new_columns

def get_leaf_node_indices(tree, node_id=0):

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child == right_child:
        return [node_id]
    else:
        left_indices = get_leaf_node_indices(tree, left_child)
        right_indices = get_leaf_node_indices(tree, right_child)
        return left_indices + right_indices


