"""clean_oa_cache_fuzzy.py — purga del cache de OpenAlex las entradas que
fueron resueltas por el Stage B (JW fuzzy) que ahora está deshabilitado.

Para cada entrada del pickle:
  - mantiene los exact matches  (oa_match_score == 1.0)
  - mantiene los not_found      (oa_status == 'not_found')
  - reemplaza por not_found     las que vinieron del fuzzy (oa_match_score < 1.0)

Idempotente: si la corrés dos veces, la segunda no hace nada (no quedan
entradas fuzzy).

Uso:
  python clean_oa_cache_fuzzy.py --cache /data/datasets/LLMScholar-Personas/results/summary/.oa_cache.pkl
  python clean_oa_cache_fuzzy.py --cache <path> --dry-run    # sólo cuenta, no escribe
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

from factuality_openalex import OA_COLS, STATUS_NOT_FOUND


def _empty_record() -> dict:
    return {col: (STATUS_NOT_FOUND if col == "oa_status" else None) for col in OA_COLS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="ruta al .oa_cache.pkl")
    ap.add_argument("--dry-run", action="store_true", help="no escribir, sólo reportar")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: no existe {cache_path}", file=sys.stderr)
        sys.exit(1)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Cache cargado: {len(cache):,} entradas")

    n_exact = n_fuzzy = n_not_found = n_other = 0
    fuzzy_keys = []
    for k, v in cache.items():
        if not isinstance(v, dict):
            n_other += 1
            continue
        status = v.get("oa_status")
        score  = v.get("oa_match_score")
        if status == "found" and score == 1.0:
            n_exact += 1
        elif status == "found" and isinstance(score, (int, float)) and score < 1.0:
            n_fuzzy += 1
            fuzzy_keys.append(k)
        elif status == STATUS_NOT_FOUND:
            n_not_found += 1
        else:
            n_other += 1

    print(f"  exact   (mantener): {n_exact:>8,}")
    print(f"  fuzzy   (purgar):   {n_fuzzy:>8,}")
    print(f"  not_found:          {n_not_found:>8,}")
    print(f"  otros:              {n_other:>8,}")

    if args.dry_run:
        print("\n--dry-run: no se escribe nada.")
        if fuzzy_keys[:10]:
            print("Primeras 10 keys fuzzy:")
            for k in fuzzy_keys[:10]:
                print(f"  {k!r}  (score={cache[k]['oa_match_score']:.3f}, "
                      f"oa_id={cache[k].get('oa_id')})")
        return

    if n_fuzzy == 0:
        print("\nNo hay entradas fuzzy — nada que hacer.")
        return

    # Backup antes de tocar nada
    backup = cache_path.with_suffix(cache_path.suffix + ".bak_pre_fuzzy_purge")
    shutil.copy2(cache_path, backup)
    print(f"\nBackup creado: {backup}")

    # Reemplazar fuzzy entries por _empty_record()
    for k in fuzzy_keys:
        cache[k] = _empty_record()

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache actualizado: {len(cache):,} entradas (purgaron {n_fuzzy:,})")


if __name__ == "__main__":
    main()
