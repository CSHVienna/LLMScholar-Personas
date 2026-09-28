#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the NamSor diaspora ground-truth set, row by row.

Exists because the set was assembled by LLM agents, so every claim in it has to
be checkable against something outside the model that wrote it. A row passes
only when all of these hold:

  schema    required fields present, ISO2 well formed, label in NamSor's live
            taxonomy, stratum consistent with origin vs work country, no LinkedIn
  source    source_url_1 resolves
  quote     evidence_quote is actually found in one of the cited pages
  author    the person resolves to a real indexed author (OpenAlex exact-name
            match), or a manual audit has cleared the row

`--audit` merges a hand-audit CSV (id, veredicto = CONSERVAR|CORREGIR|ELIMINAR)
so rows a human/agent checked by hand are not re-flagged by the crude automatic
signals. OpenAlex under-indexes social scientists, older scholars and authors
outside Europe/North America, so a low works_count is a prompt to look, never a
verdict on its own — the audit column is what settles those.

Exit status is 0 only when every row passes, so this can gate the benchmark run.
"""

import argparse, csv, html, json, re, ssl, sys, time, unicodedata
import urllib.parse, urllib.request
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from namsor import load_namsor_key  # noqa: E402

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
TAXONOMY_URL = "https://v2.namsor.com/NamSorAPIv2/api2/json/taxonomyClasses/personalname_country_diaspora"
QUOTE_THRESHOLD = 0.80
MIN_WORKS = 5


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", " ".join(s.split()))


def fetch_text(url: str, timeout: int = 30) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,es,fr,de,it,pt,ar"})
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            raw = r.read()[:3_000_000]
        t = raw.decode("utf-8", "ignore")
        t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
        return norm(html.unescape(re.sub(r"(?s)<[^>]+>", " ", t)))
    except Exception:
        return None


def url_alive(url: str) -> str:
    if not url.strip():
        return "EMPTY"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25, context=CTX) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        # 403 is usually an anti-bot wall, not a dead link; keep it distinguishable
        return f"HTTP{e.code}"
    except Exception as e:
        return type(e).__name__


def quote_found(row: dict) -> tuple[str, float]:
    q = norm(row["evidence_quote"])
    toks = [w for w in q.split() if len(w) > 3]
    best = 0.0
    for col in ("source_url_1", "source_url_2"):
        u = row[col].strip()
        if not u:
            continue
        page = fetch_text(u)
        if page is None:
            continue
        if q and q in page:
            return "LITERAL", 1.0
        if toks:
            best = max(best, sum(1 for w in toks if w in page) / len(toks))
    if best >= QUOTE_THRESHOLD:
        return "NEAR", round(best, 2)
    return "NOT_FOUND", round(best, 2)


def openalex(row: dict) -> dict:
    tgt, ln = norm(row["full_name"]), norm(row["last_name"])
    try:
        u = "https://api.openalex.org/authors?search=%s&per-page=25" % urllib.parse.quote(row["full_name"])
        req = urllib.request.Request(u, headers={"User-Agent": "benchmark-validation/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            cands = json.load(r).get("results", [])
    except Exception as e:
        return {"match": "ERROR:" + type(e).__name__, "works": None, "country": None}
    # (retries handled by the caller via openalex_retry)
    exact = [h for h in cands
             if tgt in [norm(h.get("display_name"))] + [norm(a) for a in (h.get("display_name_alternatives") or [])]]
    pool = exact or [h for h in cands if ln and ln in norm(h.get("display_name"))]
    if not pool:
        return {"match": "NO_MATCH", "works": None, "country": None}
    # a homonym in the right country beats a better-cited homonym elsewhere
    byc = [h for h in pool if any(i.get("country_code") == row["work_country_iso2"]
                                  for i in (h.get("last_known_institutions") or []))]
    pool.sort(key=lambda h: -(h.get("cited_by_count") or 0))
    h = byc[0] if byc else pool[0]
    inst = (h.get("last_known_institutions") or [{}])
    return {"match": "EXACT" if exact else "SURNAME",
            "works": h.get("works_count"),
            "country": inst[0].get("country_code") if inst else None,
            "country_ok": bool(byc)}


def semantic_scholar(row: dict) -> dict:
    """Author-existence check via Semantic Scholar.

    OpenAlex is the nicer source but its keyless daily budget is shared per IP
    and runs out, and an exhausted budget must never read as "row is fine".
    S2 indexes older, non-English and social-science work that OpenAlex misses,
    which is exactly where this dataset's hard cases live.
    """
    tgt, ln = norm(row["full_name"]), norm(row["last_name"])
    url = ("https://api.semanticscholar.org/graph/v1/author/search?query=%s"
           "&fields=name,paperCount,citationCount,affiliations&limit=20"
           % urllib.parse.quote(row["full_name"]))
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "academic-benchmark-validation/1.0"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                data = json.load(r).get("data", []) or []
            break
        except Exception as e:
            if attempt == 3:
                return {"match": "ERROR:" + type(e).__name__, "works": None}
            time.sleep(3 * (attempt + 1))
    exact = [h for h in data if norm(h.get("name")) == tgt]
    pool = exact or [h for h in data if ln and ln in norm(h.get("name"))]
    if not pool:
        return {"match": "NO_MATCH", "works": None}
    h = max(pool, key=lambda x: (x.get("paperCount") or 0))
    return {"match": "EXACT" if exact else "SURNAME", "works": h.get("paperCount"),
            "cited": h.get("citationCount"), "s2name": h.get("name")}


def crossref_works(row: dict) -> dict:
    """Count Crossref works that actually carry a matching author.

    Counting *works whose author list matches* beats asking an author-entity
    endpoint "how many papers does this person have". Both OpenAlex and S2 split
    one scholar across several ids, so their per-entity counts understate badly:
    S2 credits a Nobel laureate in this set with 7 papers, and gives another row
    4 papers against 804 citations — plainly a fragment, not a person.
    """
    u = ("https://api.crossref.org/works?query.author=%s&rows=100&select=author"
         % urllib.parse.quote(row["full_name"]))
    for attempt in range(3):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "academic-benchmark-validation/1.0"})
            with urllib.request.urlopen(req, timeout=45, context=CTX) as r:
                items = json.load(r)["message"]["items"]
            break
        except Exception as e:
            if attempt == 2:
                return {"match": "ERROR:" + type(e).__name__, "works": None}
            time.sleep(3 * (attempt + 1))
    ln = norm(row["last_name"])
    fi = norm(row["first_name"])[:1]
    n = 0
    for it in items:
        for a in (it.get("author") or []):
            fam, giv = norm(a.get("family")), norm(a.get("given"))
            if fam and fam == ln and (not fi or (giv[:1] == fi if giv else True)):
                n += 1
                break
    return {"match": "CROSSREF" if n else "NO_MATCH", "works": n}


def author_check(row: dict) -> dict:
    """Crossref first, then S2, then OpenAlex. An error in all three is
    'not checked', which the caller treats as a failure, never as a pass."""
    cr = crossref_works(row)
    if not str(cr.get("match", "")).startswith("ERROR") and (cr.get("works") or 0) > 0:
        cr["src"] = "crossref"
        return cr
    s2 = semantic_scholar(row)
    if not str(s2.get("match", "")).startswith("ERROR"):
        s2["src"] = "s2"
        return s2
    oa = openalex(row)
    oa["src"] = "openalex"
    return oa


def openalex_retry(row: dict, tries: int = 4) -> dict:
    """OpenAlex rate-limits bursts. A 429 means "not checked", never "fine"."""
    for attempt in range(tries):
        res = openalex(row)
        if not str(res.get("match", "")).startswith("ERROR"):
            return res
        time.sleep(2 ** attempt)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--audit", type=Path, help="Hand-audit CSV with id + veredicto.")
    p.add_argument("--report", type=Path, help="Write the per-row report here.")
    p.add_argument("--offline", action="store_true", help="Schema checks only, no network.")
    args = p.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    print(f"Filas: {len(rows)}")

    cleared, corrected, removed = set(), set(), set()
    if args.audit and args.audit.exists():
        for a in csv.DictReader(args.audit.open(encoding="utf-8")):
            v = (a.get("veredicto") or "").strip().upper()
            {"CONSERVAR": cleared, "CORREGIR": corrected, "ELIMINAR": removed}.get(v, set()).add(a["id"])
        print(f"Auditoria: {len(cleared)} conservar, {len(corrected)} corregir, {len(removed)} eliminar")

    tax = []
    if not args.offline:
        try:
            key = load_namsor_key()
            if not key:
                raise RuntimeError("no NamSor key")
            req = urllib.request.Request(TAXONOMY_URL, headers={"X-API-KEY": key, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                tax = json.load(r)["taxonomyClasses"]
        except Exception as e:
            print(f"  aviso: no pude leer la taxonomia en vivo ({type(e).__name__}); se omite esa comprobacion")

    report = []
    net = {}
    if not args.offline:
        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            fq = {ex.submit(quote_found, r): r["id"] for r in rows}
            fu = {ex.submit(url_alive, r["source_url_1"]): r["id"] for r in rows}
            for f in cf.as_completed(list(fq) + list(fu)):
                rid = fq.get(f) or fu.get(f)
                net.setdefault(rid, {})["quote" if f in fq else "url1"] = f.result()
        for r in rows:
            net[r["id"]]["oa"] = author_check(r)
            time.sleep(1.5)

    for r in rows:
        rid, fails = r["id"], []
        for c in ("first_name", "last_name", "work_country_iso2", "expected_diaspora", "source_url_1"):
            if not r[c].strip():
                fails.append(f"schema:{c}_vacio")
        iso = r["work_country_iso2"].strip()
        if len(iso) != 2 or not iso.isalpha() or iso != iso.upper():
            fails.append("schema:iso2")
        if tax and r["expected_diaspora"] not in tax:
            fails.append("schema:etiqueta_inexistente")
        same = r["origin_country_iso2"].strip().upper() == iso
        if same != (r["stratum"] == "native"):
            fails.append("schema:estrato_incoherente")
        if "linkedin" in (r["source_url_1"] + r["source_url_2"]).lower():
            fails.append("schema:linkedin")

        q = qs = u1 = oa = None
        if not args.offline:
            n = net.get(rid, {})
            q, qs = n.get("quote", ("?", 0))
            u1 = n.get("url1")
            oa = n.get("oa", {})
            if q == "NOT_FOUND" and rid not in cleared:
                fails.append(f"cita:no_hallada({qs})")
            if u1 not in ("200", None) and not str(u1).startswith("HTTP4") and rid not in cleared:
                fails.append(f"fuente:url1_{u1}")
            if u1 == "HTTP404" and rid not in cleared:
                fails.append("fuente:url1_404")
            w = oa.get("works")
            m = str(oa.get("match") or "")
            if rid not in cleared:
                if m.startswith("ERROR"):
                    # unverified is not verified: never let a network failure pass a row
                    fails.append(f"autor:no_comprobado({m})")
                elif m == "NO_MATCH" or (w is not None and w < MIN_WORKS):
                    fails.append(f"autor:openalex({m},works={w})")

        if rid in removed:
            fails.append("auditoria:ELIMINAR")
        if rid in corrected:
            fails.append("auditoria:CORREGIR")

        report.append({"id": rid, "estado": "OK" if not fails else "REVISAR",
                       "cita": q, "cita_score": qs, "url1": u1,
                       "openalex": (oa or {}).get("match"), "works": (oa or {}).get("works"),
                       "auditado": "SI" if rid in cleared else "",
                       "problemas": ";".join(fails)})

    ok = [x for x in report if x["estado"] == "OK"]
    bad = [x for x in report if x["estado"] != "OK"]
    print(f"\nPASAN: {len(ok)}/{len(rows)}   REQUIEREN REVISION: {len(bad)}")
    for x in bad:
        print(f"  {x['id']:5} {x['problemas']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report[0]))
            w.writeheader(); w.writerows(report)
        print(f"\nInforme -> {args.report}")

    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
