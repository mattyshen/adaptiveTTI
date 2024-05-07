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

def compute_subset_product(subset, data):
    return pd.Series(1, index = data.index) if not subset else data[list(subset)].product(axis=1)

def binary_map(arr):
    unique_values = list(set(arr))
    if 1 in unique_values and -1 in unique_values:
        return {1: 1, -1: -1}
    elif 1 in unique_values:
        unique_values.remove(1)
        return {unique_values[0]: -1, 1: 1}
    elif -1 in unique_values:
        unique_values.remove(-1)
        return {-1: -1, unique_values[0]: 1}
    else:
        return {unique_values[0]: -1, unique_values[1]: 1}
