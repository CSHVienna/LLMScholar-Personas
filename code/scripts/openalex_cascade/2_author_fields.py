"""
2_author_fields.py — Step 2: the field and subfields of each candidate author.

For every OpenAlex author found in step 1:
  field_topics    our field for the author's main OpenAlex field: the field
                  (of all 26) with most of their papers, read from
                  `authors.topics`. None when that field is not one of ours.
  field_concepts  our field for the author's main level-0 concept (inherited
                  from Microsoft Academic Graph), read from the legacy author
                  records. None when that concept is not one of ours.
  subfields       the author's 3 OpenAlex subfields with most papers (for the
                  tie-break on the prompt's subfield).
See fields.py for both classifications and how they were validated.

The concepts come from the legacy snapshot (58 GB of author records, ~15 min);
they are extracted once into author_concepts_level0.parquet, next to the output,
and reused.

Output: author_fields.parquet — oa_id, field_topics, field_concepts, subfields.

Usage (from code/, with PYTHONPATH=.):
  python scripts/openalex_cascade/2_author_fields.py \\
      --candidates ../results/summary/openalex_cascade/candidates.parquet \\
      --output     ../results/summary/openalex_cascade/author_fields.parquet
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fields import CONCEPT_MAP, study_field  # noqa: E402

from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

LEGACY_AUTHORS = "/code/openalex/oa_data/legacy-data/authors/updated_date=*/part_*.gz"

TOPICS = """
SELECT u.oa_id, t.field.display_name AS oa_field, t.subfield.display_name AS oa_subfield,
       sum(t.count) AS n
FROM candidates u
JOIN oa.authors a ON a.id = CAST(substr(u.oa_id, length('https://openalex.org/A') + 1) AS BIGINT),
     UNNEST(a.topics) AS x(t)
GROUP BY ALL
"""

CONCEPTS = """
COPY (
  SELECT r.id AS oa_id, c.display_name AS concept, c.score
  FROM read_json('{legacy}', format='newline_delimited', compression='gzip',
                 columns={{'id': 'VARCHAR',
                           'x_concepts': 'STRUCT(display_name VARCHAR, level INTEGER, score DOUBLE)[]'}}) r
  JOIN candidates u ON u.oa_id = r.id,
       UNNEST(r.x_concepts) AS x(c)
  WHERE c.level = 0
) TO '{out}' (FORMAT parquet)
"""


def run(candidates: str, duckdb_path: str, legacy: str, output: str) -> None:
    cand = pd.read_parquet(candidates, columns=["oa_id"]).drop_duplicates()
    con = duckdb.connect()
    con.register("candidates", cand)

    # Topics: papers per (author, OpenAlex field, subfield).
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    topics = con.execute(TOPICS).df()
    per_field = topics.groupby(["oa_id", "oa_field"])["n"].sum().reset_index()
    main_field = per_field.sort_values("n", ascending=False).drop_duplicates("oa_id")
    per_sub = topics.sort_values("n", ascending=False)
    # The main subfield inside the main field decides Sociology (a subfield of Social Sciences).
    main_sub = per_sub.merge(main_field[["oa_id", "oa_field"]], on=["oa_id", "oa_field"]).drop_duplicates("oa_id")
    main_field = main_field.merge(main_sub[["oa_id", "oa_subfield"]], on="oa_id")
    main_field["field_topics"] = [study_field(f, s) for f, s in zip(main_field.oa_field, main_field.oa_subfield)]
    top3 = (per_sub.groupby(["oa_id", "oa_subfield"])["n"].sum().reset_index()
            .sort_values("n", ascending=False).groupby("oa_id").head(3)
            .groupby("oa_id")["oa_subfield"].agg(list).rename("subfields"))

    # Concepts: the main level-0 concept, extracted once from the legacy records.
    concepts_file = Path(output).with_name("author_concepts_level0.parquet")
    if not concepts_file.exists():
        logger.info("Reading the legacy author records (~15 min)")
        con.execute("SET memory_limit='48GB'")
        con.execute(CONCEPTS.format(legacy=legacy, out=concepts_file))
    concepts = pd.read_parquet(concepts_file)
    main_concept = concepts.sort_values("score", ascending=False).drop_duplicates("oa_id")
    main_concept["field_concepts"] = main_concept["concept"].map(CONCEPT_MAP)
    con.close()

    out = (cand.merge(main_field[["oa_id", "field_topics"]], on="oa_id", how="left")
               .merge(main_concept[["oa_id", "field_concepts"]], on="oa_id", how="left")
               .merge(top3, on="oa_id", how="left"))
    out.to_parquet(output, index=False)
    logger.info("Wrote %s: %d authors | field from topics %.0f%%, from concepts %.0f%%, either %.0f%%",
                output, len(out), 100 * out.field_topics.notna().mean(),
                100 * out.field_concepts.notna().mean(),
                100 * (out.field_topics.notna() | out.field_concepts.notna()).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--legacy_authors", default=LEGACY_AUTHORS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.candidates, args.duckdb, args.legacy_authors, args.output)


if __name__ == "__main__":
    main()
