import numpy as np
import pandas as pd

class TrieNode:
    def __init__(self, features = set(), feature = None):
        self.left = None
        self.right = None
        self.features = features
        self.is_end_of_feature = False
        if feature is not None:
            self.features.add(feature)

def build_trie(features, removals):
    root = TrieNode()
    stack = [(root, 0)]  # Initialize stack with root node and depth 0
    final_features = []
    while stack:
        node, depth = stack.pop()
        if depth == len(features) or any([len(set(node.features).intersection(s)) >= 2 for s in removals]):
            final_features.append(node.features.copy())
            node.is_end_of_feature = True
        else:
            node.left = TrieNode(node.features.copy())
            node.right = TrieNode(node.features.copy(), features[depth])
            stack.append((node.right, depth + 1))
            stack.append((node.left, depth + 1))
    return final_features, root
