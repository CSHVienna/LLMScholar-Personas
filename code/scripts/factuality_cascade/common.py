"""
common.py — Small helpers shared by the factuality_cascade scripts.

Every recommendation asks for three things, which the scripts compare against
the researchers' data:
  query    the recommended name ("name lastname")
  Field    the field the prompt asked for, in English (same values as the
           parquet's Field column)
  bucket   the seniority the prompt asked for: "Junior" or "Senior"
"""

import pandas as pd

from libs.metrics.constants import FIELD_NORM_MAP, LLM_TARGET_TO_BUCKET

JUNIOR_MAX = 10  # career_age <= 10 → Junior
SENIOR_MIN = 20  # career_age >= 20 → Senior  (11–19 → neither)

#se cargan los datos de los investigadores recomedados por el LLM
def add_request_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add query, Field and bucket to a recommendations table."""
    df["query"] = (
        df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str)
    ).str.strip()
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
