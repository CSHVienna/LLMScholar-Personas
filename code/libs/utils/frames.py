"""DataFrame helpers shared by all pipeline steps."""

import pandas as pd

from utils import constants as cons

# Ordered categories of every categorical column produced by the pipeline
CATEGORIES = {**cons.VARIABLE_VALUES,
              cons.COL_PARSE_METHOD: cons.PARSE_METHODS,
              'json_root': cons.JSON_ROOTS}

INTEGER_COLUMNS = ['prompt_tokens', 'completion_tokens', 'reasoning_tokens']


def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known columns to ordered categoricals / nullable ints (consistent order, smaller files)."""
    for column, values in CATEGORIES.items():
        if column in df:
            df[column] = pd.Categorical(df[column], categories=values, ordered=True)
    for column in INTEGER_COLUMNS:
        if column in df:
            df[column] = df[column].astype('Int64')
    return df
