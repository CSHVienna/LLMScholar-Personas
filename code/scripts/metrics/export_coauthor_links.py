"""
export_coauthor_links.py — Materialise the coauthorship links behind
connectedness (paper Eqs. 6-7) as durable, inspectable files.

`aggregators.compute_connectedness` only ever sees the graph as an in-memory
sparse matrix cached in a joblib blob, so there is no way to look at *which*
recommended authors actually coauthored with whom. This script writes that out:
an undirected edge list, a per-author adjacency map, and a node table.

Pipeline
--------
1. Read the author-level factuality table (``factuality_full.csv``), keep the
   valid responses, apply the benchmark's factual rule — ``author_found`` =
   matched in Semantic Scholar OR in OpenAlex — and report how many factual
   authors carry an OpenAlex id. Only those can have a node in the graph at all,
   so this doubles as the openalex_id coverage check.
2. Take the unique OpenAlex ids as the graph population U.
3. Reuse ``<results>/.cache/coauthorship_graph.joblib`` when it was built for
   that exact population (the payload carries a population fingerprint);
   otherwise rebuild it from the OpenAlex snapshot — one chunked pass over
   ``oa.works`` exploding ``authorships`` and pairing co-listed authors, ~3 h.
   Two authors are linked when they appear in the authorship list of the same
   work; edges are undirected, deduplicated and carry no self-loops.
4. Export the links in three shapes and verify the JSON round-trips.

Restricting the graph to U is exact, not an approximation: every response's
author set U-hat_i is a subset of U and induced subgraphs compose, so
``G[U-hat_i] == (G[U])[U-hat_i]``. Coauthors *outside* U are deliberately not
edges here — Eq. 6 only ever reads the subgraph induced on the recommended
authors, so a link to a non-recommended coauthor cannot change any value.

Outputs (under ``<results>/coauthorship/``)
-------------------------------------------
``coauthor_edges.parquet`` / ``coauthor_edges.csv.gz``
    Undirected edge list, one row per pair, ``src`` < ``dst`` lexicographically.
``coauthor_adjacency.json.gz``
    ``{oa_id: [oa_id, ...]}``. Every node of U is a key, including the isolated
    ones (empty list) — the file is a complete record of the graph, so a missing
    key means "not in the population", never "no coauthors".
``coauthor_nodes.parquet`` / ``coauthor_nodes.csv``
    ``author_id``, ``display_name`` (from ``oa.authors``), ``degree``.
``manifest.json``
    Provenance and counts: source paths, population fingerprint, git commit.
``README.md``
    What each file is, regenerated on every run.

Keys are OpenAlex ids, not names, on purpose: display names are not unique
(homonyms collide and would silently merge two people's coauthors), and the
whole pipeline keys on ``author_id`` anyway. ``coauthor_nodes`` is the lookup
when a human-readable label is needed.

Usage (from code/, with PYTHONPATH=.)
-------------------------------------
  python scripts/metrics/export_coauthor_links.py
  python scripts/metrics/export_coauthor_links.py --results <dir> --out <dir>
  python scripts/metrics/export_coauthor_links.py --rebuild   # force graph rebuild

Defaults come from [data] in config.ini.
"""

import argparse
import gzip
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from libs.metrics.constants import VALID_FLAGS
from libs.metrics.io import (
    build_author_index,
    build_coauthorship_graph,
    validate_path,
)
from libs.utils.config import config_default, get_results_path
from libs.utils.logging import setup_logging

logger = setup_logging()


def _normalise_oa_id(value) -> str | None:
    """``oa_id`` -> bare ``A<digits>`` token, or None.

    Same normalisation as build_valid_calls.py, kept verbatim so the population
    here is byte-identical to the one the metric was computed over: oa_id may
    arrive as a URL, as a bare numeric id read back as float when the column has
    NaNs, or as a plain string.
    """
    if pd.isna(value) or value == "":
        return None
    token = (str(int(value)) if isinstance(value, float) else str(value))
    return token.split("/")[-1].strip() or None


def load_factual_authors(
    fact_path: Path, *, chunksize: int = 500_000
) -> tuple[list[str], dict]:
    """Unique OpenAlex ids of the factual authors, plus coverage counts.

    Streams ``factuality_full.csv`` (it is ~7 GB) reading only the three columns
    the factual rule needs. Returns ``(author_ids, stats)``; ``stats`` is what
    goes into the manifest and answers "does the openalex_id exist?".
    """
    cols = ["valid_flag", "author_status", "oa_id"]
    author_ids: set[str] = set()
    n_rows = n_valid = n_factual = n_with_oa = 0
    n_ss_only = 0

    logger.info("Reading %s", fact_path)
    for chunk in pd.read_csv(
        fact_path, usecols=cols, chunksize=chunksize, dtype=str, low_memory=False
    ):
        n_rows += len(chunk)
        valid = chunk[chunk["valid_flag"].isin(VALID_FLAGS)]
        n_valid += len(valid)

        oa = valid["oa_id"].map(_normalise_oa_id)
        found_ss = valid["author_status"] == "found"
        has_oa = oa.notna()
        factual = found_ss | has_oa

        n_factual += int(factual.sum())
        n_with_oa += int((factual & has_oa).sum())
        n_ss_only += int((factual & ~has_oa).sum())
        author_ids.update(oa[factual & has_oa].tolist())

    stats = {
        "rows_total": n_rows,
        "rows_valid_responses": n_valid,
        "recommendations_factual": n_factual,
        "recommendations_factual_with_openalex_id": n_with_oa,
        "recommendations_factual_without_openalex_id": n_ss_only,
        "pct_factual_without_openalex_id": (
            round(100 * n_ss_only / n_factual, 3) if n_factual else None
        ),
        "unique_authors_with_openalex_id": len(author_ids),
    }
    logger.info(
        "Factual recommendations: %d, of which %d (%.2f%%) have no OpenAlex id "
        "(matched only in Semantic Scholar) and therefore no node",
        n_factual,
        n_ss_only,
        100 * n_ss_only / n_factual if n_factual else 0.0,
    )
    logger.info("Graph population: %d unique OpenAlex ids", len(author_ids))
    return sorted(author_ids), stats


def load_or_build_graph(
    author_ids: list[str],
    *,
    cache_path: Path,
    oa_duckdb_path: str | None,
    rebuild: bool,
) -> tuple[sparse.csr_matrix, dict[str, int], bool]:
    """Return ``(adjacency, index, from_cache)``.

    Prefers the cache the metric run already produced. The population
    fingerprint is checked by ``build_coauthorship_graph`` itself, so passing
    the freshly derived ids is enough: a mismatch triggers a rebuild instead of
    silently returning a graph for other authors. Rebuilding needs the OpenAlex
    snapshot; without it, a mismatch is a hard error rather than a wrong export.
    """
    index = build_author_index(author_ids)

    if cache_path.exists() and not rebuild:
        import joblib

        payload = joblib.load(cache_path)
        cached_index = payload["index"]
        if set(cached_index) == set(index):
            logger.info(
                "Coauthorship graph: reusing %s (%d nodes, %d undirected edges)",
                cache_path,
                payload["adjacency"].shape[0],
                payload["adjacency"].nnz // 2,
            )
            return payload["adjacency"], cached_index, True
        logger.warning(
            "Cached graph covers a different population (%d vs %d authors) — "
            "it cannot be exported as-is",
            len(cached_index),
            len(index),
        )

    if not oa_duckdb_path:
        raise SystemExit(
            "The coauthorship graph has to be built and no OpenAlex snapshot is "
            "available: pass --oa_duckdb or set [data].oa_duckdb in config.ini. "
            "Budget ~3 h for the pass over oa.works."
        )

    adjacency, index = build_coauthorship_graph(
        author_ids,
        cache_path=cache_path,
        oa_duckdb_path=oa_duckdb_path,
        temp_dir=cache_path.parent / "duckdb_tmp",
        rebuild=rebuild,
        logger=logger,
    )
    return adjacency, index, False


def fetch_display_names(
    author_ids: list[str], *, oa_duckdb_path: str | None
) -> pd.Series:
    """``author_id -> display_name`` from ``oa.authors``, empty Series if absent.

    Joins on the full OpenAlex URL, which is what the snapshot stores. Cheap:
    one indexed lookup per id, seconds rather than the minutes the works pass
    takes. Names are a convenience label only — nothing downstream keys on them.
    """
    empty = pd.Series(dtype=str, name="display_name")
    if not oa_duckdb_path:
        logger.warning("No OpenAlex snapshot: display names will be empty")
        return empty

    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{oa_duckdb_path}' AS oa (READ_ONLY)")
        u_df = pd.DataFrame(
            {
                "author_url": [f"https://openalex.org/{a}" for a in author_ids],
                "author_id": author_ids,
            }
        )
        con.register("u_df", u_df)
        out = con.execute(
            """
            SELECT u.author_id AS author_id, a.display_name AS display_name
            FROM u_df u LEFT JOIN oa.authors a ON a.id_full = u.author_url
            """
        ).fetchdf()
    finally:
        con.close()

    names = out.set_index("author_id")["display_name"]
    logger.info(
        "Display names resolved for %d/%d authors", int(names.notna().sum()), len(names)
    )
    return names


def export_edges(
    adjacency: sparse.csr_matrix, ids: np.ndarray, out_dir: Path
) -> tuple[Path, Path, int]:
    """Write the undirected edge list as parquet + gzipped CSV.

    Only the upper triangle is taken, so each pair appears once with
    ``src`` < ``dst``. The adjacency is symmetric and diagonal-free by
    construction, which ``triu(k=1)`` relies on.
    """
    coo = sparse.triu(adjacency, k=1).tocoo()
    edges = pd.DataFrame({"src": ids[coo.row], "dst": ids[coo.col]})
    edges.sort_values(["src", "dst"], inplace=True, kind="stable")
    edges.reset_index(drop=True, inplace=True)

    parquet_path = out_dir / "coauthor_edges.parquet"
    csv_path = out_dir / "coauthor_edges.csv.gz"
    edges.to_parquet(parquet_path, index=False)
    edges.to_csv(csv_path, index=False, compression="gzip")
    logger.info("Edge list: %d pairs → %s, %s", len(edges), parquet_path, csv_path)
    return parquet_path, csv_path, len(edges)


def export_adjacency_json(
    adjacency: sparse.csr_matrix, ids: np.ndarray, out_dir: Path
) -> tuple[Path, int]:
    """Write ``{oa_id: [oa_id, ...]}`` gzipped, then verify it round-trips.

    Written incrementally rather than via one ``json.dump`` of a 300k-key dict:
    the payload is ~8 M ids and materialising it as a single string first is
    what would blow the memory budget. Each key and each value still goes
    through ``json.dumps``, so the output is valid JSON by construction — and
    the file is read back in full afterwards to prove it, since "is it
    serialisable?" is only answered by a round trip.

    CSR rows are already sorted by column index, so neighbour lists come out in
    a stable, id-sorted order for free.
    """
    path = out_dir / "coauthor_adjacency.json.gz"
    indptr, indices = adjacency.indptr, adjacency.indices

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
        fh.write("{\n")
        for row, author_id in enumerate(ids):
            neighbours = ids[indices[indptr[row] : indptr[row + 1]]].tolist()
            sep = ",\n" if row < len(ids) - 1 else "\n"
            fh.write(f"  {json.dumps(str(author_id))}: {json.dumps(neighbours)}{sep}")
        fh.write("}\n")

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reloaded = json.load(fh)

    if len(reloaded) != len(ids):
        raise AssertionError(
            f"adjacency round trip lost keys: {len(reloaded)} != {len(ids)}"
        )
    rng = random.Random(0)
    for row in rng.sample(range(len(ids)), k=min(1000, len(ids))):
        expected = ids[indices[indptr[row] : indptr[row + 1]]].tolist()
        got = reloaded[str(ids[row])]
        if got != expected:
            raise AssertionError(f"adjacency mismatch for {ids[row]}")
    logger.info(
        "Adjacency JSON: %d keys, round trip verified on 1000 sampled nodes → %s",
        len(reloaded),
        path,
    )
    return path, len(reloaded)


def export_nodes(
    ids: np.ndarray, degrees: np.ndarray, names: pd.Series, out_dir: Path
) -> tuple[Path, Path]:
    """Write the node table: id, display name, degree."""
    nodes = pd.DataFrame({"author_id": ids, "degree": degrees})
    nodes["display_name"] = (
        nodes["author_id"].map(names) if len(names) else pd.NA
    )
    nodes = nodes[["author_id", "display_name", "degree"]]

    parquet_path = out_dir / "coauthor_nodes.parquet"
    csv_path = out_dir / "coauthor_nodes.csv"
    nodes.to_parquet(parquet_path, index=False)
    nodes.to_csv(csv_path, index=False)
    logger.info("Node table: %d authors → %s, %s", len(nodes), parquet_path, csv_path)
    return parquet_path, csv_path


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


README_TEMPLATE = """# Coauthorship links (paper Eqs. 6-7)

Generated by `code/scripts/metrics/export_coauthor_links.py` on {generated_at}.
Do not edit by hand; rerun the script instead.

This is the graph `aggregators.compute_connectedness` consumes. Two authors are
linked when they appear in the authorship list of the same OpenAlex work.
Edges are undirected, deduplicated, and carry no self-loops.

The population is the {n_nodes} unique OpenAlex ids among the benchmark's
factual recommended authors. Coauthors outside that set are not edges here:
Eq. 6 only reads the subgraph induced on a response's own recommended authors,
so a link to a non-recommended coauthor cannot change any connectedness value.

| file | what it is |
|---|---|
| `coauthor_edges.parquet` / `.csv.gz` | undirected edge list, one row per pair, `src` < `dst`. {n_edges} rows. |
| `coauthor_adjacency.json.gz` | `{{oa_id: [oa_id, ...]}}`. All {n_nodes} nodes are keys, isolated ones with an empty list — a missing key means "not in the population", never "no coauthors". |
| `coauthor_nodes.parquet` / `.csv` | `author_id`, `display_name`, `degree`. `degree` counts coauthors **inside this population only** — it is not the author's total coauthor count in OpenAlex, which is larger. |
| `manifest.json` | provenance and counts for this run. |

Keys are OpenAlex ids rather than names because display names are not unique —
homonyms would silently merge two people's coauthors. Use `coauthor_nodes` as
the id → name lookup.

Graph shape for this run: {n_edges} edges over {n_nodes} nodes; degree mean
{mean_degree:.2f} / median {median_degree:.0f}; {pct_isolated:.1f}% of authors isolated,
i.e. with no coauthor anywhere in the population. See `manifest.json` for the
full counts, including how many factual recommendations carry no OpenAlex id
and therefore no node at all.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=None,
        help="Results directory (default: [data].results_dir in config.ini).",
    )
    parser.add_argument(
        "--factuality",
        default=None,
        help="Author-level factuality CSV "
        "(default: <results>/summary/factuality_full.csv).",
    )
    parser.add_argument(
        "--oa_duckdb",
        default=config_default("oa_duckdb"),
        help="OpenAlex DuckDB snapshot (default: [data].oa_duckdb). Needed to "
        "resolve display names, and to build the graph if no usable cache exists.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <results>/coauthorship).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the coauthorship graph from the snapshot instead of "
        "reusing the cache (~3 h).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk when streaming the factuality CSV.",
    )
    args = parser.parse_args()

    results = Path(args.results) if args.results else get_results_path()
    fact_path = (
        Path(args.factuality)
        if args.factuality
        else results / "summary" / "factuality_full.csv"
    )
    if not fact_path.exists():
        raise SystemExit(f"Factuality table not found: {fact_path}")

    out_dir = Path(args.out) if args.out else results / "coauthorship"
    validate_path(out_dir)
    cache_path = results / ".cache" / "coauthorship_graph.joblib"

    author_ids, coverage = load_factual_authors(fact_path, chunksize=args.chunksize)
    if not author_ids:
        raise SystemExit("No factual author carries an OpenAlex id — nothing to export")

    adjacency, index, from_cache = load_or_build_graph(
        author_ids,
        cache_path=cache_path,
        oa_duckdb_path=args.oa_duckdb,
        rebuild=args.rebuild,
    )

    # Row order of the matrix, not sorted(index): the adjacency's rows are what
    # the ids have to line up with.
    ids = np.empty(len(index), dtype=object)
    for author_id, row in index.items():
        ids[row] = author_id
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)

    names = fetch_display_names(list(ids), oa_duckdb_path=args.oa_duckdb)

    edges_parquet, edges_csv, n_edges = export_edges(adjacency, ids, out_dir)
    adjacency_json, n_keys = export_adjacency_json(adjacency, ids, out_dir)
    nodes_parquet, nodes_csv = export_nodes(ids, degrees, names, out_dir)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    graph_stats = {
        "n_nodes": int(len(ids)),
        "n_edges_undirected": int(n_edges),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(degrees.max()),
        "n_isolated": int((degrees == 0).sum()),
        "pct_isolated": round(float(100 * (degrees == 0).mean()), 2),
        "density": float(adjacency.nnz / (len(ids) * (len(ids) - 1))),
    }
    manifest = {
        "generated_at": generated_at,
        "generated_by": "code/scripts/metrics/export_coauthor_links.py",
        "git_commit": _git_commit(),
        "sources": {
            "factuality_table": str(fact_path),
            "coauthorship_graph_cache": str(cache_path),
            "graph_reused_from_cache": from_cache,
            "openalex_snapshot": args.oa_duckdb,
        },
        "openalex_id_coverage": coverage,
        "graph": graph_stats,
        "outputs": {
            "edges_parquet": edges_parquet.name,
            "edges_csv_gz": edges_csv.name,
            "adjacency_json_gz": adjacency_json.name,
            "adjacency_json_keys": n_keys,
            "nodes_parquet": nodes_parquet.name,
            "nodes_csv": nodes_csv.name,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(
            generated_at=generated_at,
            n_nodes=f"{graph_stats['n_nodes']:,}",
            n_edges=f"{graph_stats['n_edges_undirected']:,}",
            mean_degree=graph_stats["mean_degree"],
            median_degree=graph_stats["median_degree"],
            pct_isolated=graph_stats["pct_isolated"],
        ),
        encoding="utf-8",
    )

    logger.info(
        "Done: %d edges over %d nodes (mean degree %.2f, %.1f%% isolated) → %s",
        graph_stats["n_edges_undirected"],
        graph_stats["n_nodes"],
        graph_stats["mean_degree"],
        graph_stats["pct_isolated"],
        out_dir,
    )


if __name__ == "__main__":
    main()
