"""
common.py — Small helpers shared by the factuality cascade scripts.

Every recommendation asks for three things, which the scripts compare against
the researchers' data:
  query    the recommended name, normalized exactly as for duplicates and
           consistency (libs/pipeline_common.author_key: "name lastname",
           lowercase, no punctuation): "J. Smith" and "J Smith" are one query
  Field    the field the prompt asked for, in English (same values as the
           parquet's Field column)
  bucket   the seniority the prompt asked for: "Junior" or "Senior"
"""

import pandas as pd

from libs.metrics.constants import FIELD_NORM_MAP, JUNIOR_MAX, LLM_TARGET_TO_BUCKET, SENIOR_MIN
from libs.pipeline_common import author_key

#se cargan los datos de los investigadores recomedados por el LLM
def add_request_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add query, Field and bucket to a recommendations table."""
    df["query"] = author_key(df).fillna("")   # the same key as duplicates and consistency
    df["Field"] = df["field"].replace(FIELD_NORM_MAP)
    df["bucket"] = df["target"].map(LLM_TARGET_TO_BUCKET)
    return df

#devuelve las combinaciones únicas que se recomendaron 
def load_requests(recommendations_path: str) -> pd.DataFrame:
    """The distinct (query, Field, bucket) the prompts asked for."""
    df = pd.read_csv(
        recommendations_path, usecols=["name", "lastname", "field", "target"]
    )
    df = add_request_columns(df)
    return df[["query", "Field", "bucket"]].drop_duplicates()

#se calcula el seniority
def seniority_bucket(career_age: pd.Series) -> pd.Series:
    bucket = pd.Series(None, index=career_age.index, dtype="object")
    bucket[career_age <= JUNIOR_MAX] = "Junior"
    bucket[career_age >= SENIOR_MIN] = "Senior"
    return bucket
