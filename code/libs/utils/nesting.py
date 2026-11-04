import re
import pandas as pd

WITHIN_SUFFIX = "_within"

def within_index(df: pd.DataFrame, parent: str, child: str) -> pd.Series:
    """0-based index of each child level within its parent (sorted)."""
    return df.groupby(parent)[child].transform(lambda s: pd.factorize(s, sort=True)[0])

# def within_lookup(df: pd.DataFrame, parent: str, child: str) -> dict[tuple[str, int], str]:
#     """(parent_level, within_idx) -> child level; consistent with within_index."""
#     return {(p, i): c
#             for p, grp in df.groupby(parent)[child]
#             for i, c in enumerate(sorted(grp.dropna().unique()))}

def within_lookup(df: pd.DataFrame, parent: str, child: str) -> dict[tuple[str, int], str]:
    """(parent_level, within_idx) -> child level; derived from within_index for consistency."""
    pairs = pd.DataFrame({'p': df[parent].astype(str),
                          'i': within_index(df, parent, child),
                          'c': df[child].astype(str)}).drop_duplicates()
    return {(p, int(i)): c for p, i, c in pairs.itertuples(index=False)}

def nested_term_pattern(parent: str, child: str) -> re.Pattern:
    """Matches e.g. 'C(field_en)[Mathematics]:C(subfield_en_within)[T.1]'."""
    # return re.compile(
    #     rf"C\({re.escape(parent)}\)\[(?P<plevel>[^\]]+)\]:"
    #     rf"C\({re.escape(child)}{WITHIN_SUFFIX}\)\[T\.(?P<idx>\d+)\]"
    # )
    return re.compile(
        rf"C\({re.escape(parent)}\)\[(?P<plevel>[^\]]+)\]:{re.escape(child)}{WITHIN_SUFFIX}"
    )