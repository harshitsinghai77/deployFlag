import json

import numpy as np
import pandas as pd

from deployflag.core.constants import (
    FREQ_DETECTED_QUANTILE_HIGH,
    FREQ_DETECTED_QUANTILE_LOW,
    FREQ_DETECTED_QUANTILE_MEDIUM_HIGH,
    FREQ_DETECTED_QUANTILE_MEDIUM_LOW,
)


def is_same(s):
    """Check whether all values are same or not."""
    a = s.to_numpy()
    return (a[0] == a[1:]).all()


def if_all_zero(s):
    """Check whether all values are zero or not."""
    a = s.to_numpy()
    return (a[0:] == 0).all()


def weighted_average(x):
    """Take numpy array and returns weighted average with higher weights on recent trends."""
    if len(x) > 0:
        weights = pd.Series(np.arange(1, len(x) + 1))
        return round(((x * weights).sum()) / (weights.sum()), 4)
    return 0


def load_json_as_dataframe(data_path):
    """Take in json path and returns pandas dataframe."""
    return pd.DataFrame(json.load(open(data_path, "r")))


def load_json(data_path):
    """Read and return the contents of the json file."""
    with open(data_path, "r") as json_file:
        return json.load(json_file)


def check_columns(df, list_of_cols):
    """Check the list_of_cols are present in dataframe, df or not."""
    cols = list(df.columns)
    return all(elem in cols for elem in list_of_cols)


def create_dummy_cols(df, list_of_cols):
    """Create dummy cols for each element of list_of_cols in df."""
    for col in list_of_cols:
        df = pd.get_dummies(df, prefix=col + "_", columns=[col])

    return df


def calculate_quantiles(series, value):
    """Categories series in different quantiles 25%, 50%, 75%."""
    if value >= series.quantile(0.75):
        return FREQ_DETECTED_QUANTILE_HIGH

    if series.quantile(0.5) <= value < series.quantile(0.75):
        return FREQ_DETECTED_QUANTILE_MEDIUM_HIGH

    if series.quantile(0.25) <= value < series.quantile(0.5):
        return FREQ_DETECTED_QUANTILE_MEDIUM_LOW

    return FREQ_DETECTED_QUANTILE_LOW


def movecol(df, cols_to_move, ref_col):
    """Reindexes the required column."""
    cols = df.columns.tolist()
    seg1 = cols[: list(cols).index(ref_col) + 1]
    seg2 = cols_to_move

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return df[seg1 + seg2 + seg3]


def get_group(g, key):
    """If key exists, return group else return empty dataframe."""
    if key in g.groups:
        return g.get_group(key)
    return pd.DataFrame()
