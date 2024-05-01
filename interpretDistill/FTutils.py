import pandas as pd
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def compute_subset_product(subset, data):
    return pd.Series(1, index = data.index) if not subset else data[list(subset)].product(axis=1)