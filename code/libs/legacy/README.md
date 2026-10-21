# libs/legacy

Code that nothing in `scripts/` or the current `notebooks/` uses any more. It was
moved here so the active modules only contain what is in use.

Kept for reference only — do not import it in new code. Several functions use
constants that no longer exist and would fail if called.

| File | Came from |
|---|---|
| `aggregators.py` | `libs/metrics/aggregators.py` (old `aggregate_*` functions) |
| `io_helpers.py` | `libs/metrics/io.py` (generic file helpers; `libs/utils/ios.py` has the ones in use) |
| `vis.py` | `libs/visuals/vis.py` (old plotting functions) |
| `metrics_utils.py` | `libs/metrics/utils.py` |
| `utils_constants.py` | `libs/utils/constants.py` (was empty) |
| `discover_keys.py` | `libs/utils/discover_keys.py` (a standalone CLI) |
| `unused_functions.py` | single unused functions from `visuals/`, `prompt/`, `utils/`, `factuality/` |
| `structural_variants.py` | `libs/metrics/io.py` (similarity/connectedness per population variant, from `scripts/legacy/run_similarity.py`; only `notebooks/legacy/` and its tests use it) |
