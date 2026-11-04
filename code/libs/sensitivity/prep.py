import pandas as pd
from typing import List, Dict


def standardize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Z-score numeric columns so coefficients are comparable in magnitude."""
    out = df.copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            print(f"Standardizing numeric column: {c}")
            mu, sd = out[c].mean(), out[c].std()
            if sd > 0:
                out[c] = (out[c] - mu) / sd
    return out


def as_categorical(df: pd.DataFrame, 
                   cols: List[str],
                   refs: Dict[str, str] | None = None) -> pd.DataFrame:
    """Convert columns to categoricals with explicit reference level."""
    out = df.copy()
    refs = refs or {}
    for c in cols:
        if c in refs:
            print(f"Setting reference level for {c}: {refs[c]}")
            others = [x for x in out[c].unique() if x != refs[c]]
            out[c] = pd.Categorical(out[c], categories=[refs[c]] + others)
        else:
            print(f"No reference level set for {c}; using default alphabetical order.")
            out[c] = pd.Categorical(out[c])
    return out

def handle_nested_missingness(df: pd.DataFrame,
                              nested_pairs: dict[str, dict],
                              strategy: str = 'indicator',
                              available_metrics: set[str] | None = None,
                              verbose: bool = True) -> pd.DataFrame:
    """Handle structurally-nested missingness with one strategy for all pairs.

    nested_pairs format: {child_var: parent_var}. Only children in
    `available_metrics` are handled. A NaN in the child is "structural" when
    its parent is 0.

    Strategies:
        'zero':      Replace structural NaN with 0. Use when 0 = "no event
                     occurred" on the variable's scale.
        'mean':      Replace structural NaN with the variable's mean (among
                     defined rows). Neutral: contributes no slope information.
        'indicator': Add a binary `<child>_measurable` column (1 where the
                     child is defined) and fill EVERY NaN of the child with 1.

    Returns (df, new_vars), where new_vars lists the indicator columns added.
    """
    out = df.copy()
    new_vars = []
    for child, parent in nested_pairs.items():
        
        if child not in available_metrics:
            continue

        mask = (out[parent] == 0) & out[child].isna()
        n_struct = mask.sum()

        if strategy == 'zero':
            out.loc[mask, child] = 0.0
        elif strategy == 'mean':
            mean_val = out[child].mean()
            out.loc[mask, child] = mean_val
        elif strategy == 'indicator':
            new_var = f'{child}_measurable'
            out[new_var] = (~out[child].isna()).astype(int)
            # mean_val = out[child].mean()
            out.loc[out[child].isna(), child] = 1 # missing # mean_val
            new_vars.append(new_var)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if verbose:
            print(f'  {child}: {n_struct} structural NaN handled via "{strategy}"')

    return out, new_vars


def drop_remaining_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN in any of `cols` and reset index.
    Apply once before fitting any model so N is identical across specs."""
    n_before = len(df)
    out = df.dropna(subset=cols).reset_index(drop=True)
    print(f"Dropped {n_before - len(out)} rows ({(n_before-len(out))/n_before:.2%}); "
          f"N = {len(out)}")
    return out