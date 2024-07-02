import pandas as pd

def is_continuous(feature):
    return (pd.api.types.is_float_dtype(feature) and len(feature.unique()) > 12) or (pd.api.types.is_integer_dtype(feature) and len(feature.unique()) > 12)