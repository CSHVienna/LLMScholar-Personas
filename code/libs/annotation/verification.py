"""
Verification tasks: the annotator checks an output of the pipeline against OpenAlex / the web (the output is shown,
unlike the blind tag tasks of tasks.py). Labels come in GROUPS: one label per group (none = not answered).

matching   one recommended name (distinct name x requested field x seniority) and the OA person(s) behind the
           pipeline's decision (B.6):
               linked     the person picked by the pipeline (author / field / seniority level)
               name only  name matches without a pick: up to 3 of the people matching by the pipeline's rule
               none       no match: up to 3 closest OA names (fuzzy candidates below the thresholds), or none
           group 'match': correct_match / false_match (something shown) | missed_match / not_in_oa (nothing linked)
           group 'intended' (optional): likely_intended / other_intended
           strata: <match_rule>|linked, <match_rule>|name, none
authors    one GT membership (author x field, A.4): the scholar's OA profile, the field, seniority and country
           the pipeline assigned. Groups: field, seniority, country (correct / wrong), profile (merged).
           strata: field|seniority

Rates (evaluate): per item and group the majority label of the sure annotators (ties left out); per stratum the share
of every label with a Wilson CI; overall, the stratum shares weighted by the stratum sizes in the population (the
sample over-represents rare strata on purpose).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from annotation.tasks import ITEM_ID, Section, stratified_sample
from groundtruth.db import sql_path
from utils import constants as cons

MATCHING, AUTHORS = cons.ANNOTATION_TASK_MATCHING, cons.ANNOTATION_TASK_AUTHORS
VERIFICATION_TASKS = [MATCHING, AUTHORS]
STRATUM, STRATUM_SIZE = 'stratum', 'stratum_size'
MAX_SHOWN = 3
OA_URL = 'https://openalex.org/A{}'
OA_SEARCH_URL = 'https://api.openalex.org/authors?search={}'
REC_ID_COLUMNS = cons.RESPONSE_ID_COLUMNS + [cons.COL_REC_INDEX]
PROFILE_COLUMNS = ['author_id', 'display_name', 'orcid', 'first_pub_year', 'last_pub_year', 'n_works', 'n_citations',
                   'h_index', 'n_works_topic', 'country', 'last_known_country']


# -----------------------------------------------------------------------------
# Codebooks
# -----------------------------------------------------------------------------
def _label(label_id: str, group: str, short: str, definition: str, scope: Optional[str] = None) -> dict:
    """scope: the labels a rate is computed among (default: the whole group), e.g. precision = correct_match among
    the items with a person shown (correct_match, false_match), not among all items."""
    return {'id': label_id, 'group': group, 'short': short, 'definition': definition, 'scope': scope or group}


def codebook(task: str) -> dict:
    if task == MATCHING:
        return {
            'task': task, 'unit': 'recommended name', 'verification': True,
            'instructions': 'A recommended name (with the field and seniority the prompt asked for) and the OpenAlex '
                            'person(s) behind the pipeline\'s decision. Open the links (OpenAlex profile, ORCID, the '
                            'search link) and judge the NAME match. One label per group.',
            'labels': [
                _label('correct_match', 'match', 'shown OA person has this name',
                       'A person is shown and carries the recommended name (spelling variants, accents, initials, '
                       'name order and middle names allowed). With several people shown: they all do.', scope='shown'),
                _label('false_match', 'match', 'shown OA person has another name',
                       'A person is shown but the name is a different name (e.g. Nimalan vs Nirmala, another surname). '
                       'Paste the right OA id if you find one.', scope='shown'),
                _label('missed_match', 'match', 'not linked, but in OpenAlex',
                       'Nothing is linked (or only near names are listed), but you find an OA person with the '
                       'recommended name: paste the OA id (A...).', scope='not_linked'),
                _label('not_in_oa', 'match', 'not linked, and not in OpenAlex',
                       'Nothing is linked and you find no OA person with the recommended name.', scope='not_linked'),
                _label('likely_intended', 'intended', 'probably the scholar the LLM meant',
                       'Optional. The person shown fits the request (field, career stage) and is likely the one meant '
                       '(e.g. a well-known scholar of the field with this name).'),
                _label('other_intended', 'intended', 'the LLM probably meant someone else',
                       'Optional. Another OA person with the same name fits the request better (a homonym was picked).'),
            ],
            'rules': ['correct_match / false_match only when a person is shown; missed_match / not_in_oa only when '
                      'nothing is linked.',
                      'Judge the name, not the field or seniority (they are checked by the pipeline from OA data).',
                      'Leave "intended" empty if you cannot tell.'],
            'fields': [{'id': 'openalex_id', 'label': 'OpenAlex id you found (A...), for missed / false matches'}],
        }
    if task == AUTHORS:
        return {
            'task': task, 'unit': 'scholar in the ground truth', 'verification': True,
            'instructions': 'An OpenAlex profile with the field, seniority and country the pipeline derived from it. '
                            'Check them with the profile, ORCID, a homepage or a web search. One label per group.',
            'labels': [
                _label('field_correct', 'field', 'works in the field shown',
                       'The field shown is the scholar\'s main area or a substantial part of their work.'),
                _label('field_wrong', 'field', 'does not work in this field',
                       'The field shown is not a real area of the scholar (e.g. a few stray works).'),
                _label('seniority_correct', 'seniority', 'career stage fits',
                       'The career stage shown (from the first publication year) fits what you find (first paper, '
                       'PhD year, position).'),
                _label('seniority_wrong', 'seniority', 'career stage is wrong',
                       'The real career start differs enough to change the stage: give the year you find.'),
                _label('country_correct', 'country', 'country fits',
                       'The country shown is that of the current (last known) affiliation.'),
                _label('country_wrong', 'country', 'country is wrong or missing',
                       'Another country, or none shown although you find one: give the country.'),
                _label('profile_merged', 'profile', 'profile mixes several people',
                       'The OA profile holds works of different people (topics, periods or places that cannot be one '
                       'career). Leave empty otherwise.'),
            ],
            'rules': ['One label per group; leave a group empty if you cannot tell.',
                      'Seniority classes: junior <= junior_max years since the first publication, senior >= senior_min '
                      '(see the analysis config); in between: mid.'],
            'fields': [{'id': 'correct_field', 'label': 'Field you would assign (if field_wrong)'},
                       {'id': 'first_year', 'type': 'number', 'label': 'Career start year you found (if seniority_wrong)'},
                       {'id': 'correct_country', 'label': 'Country you found (if country_wrong)'}],
        }
    raise ValueError(f"Unknown verification task {task!r} (expected one of {VERIFICATION_TASKS})")


def groups(task: str) -> Dict[str, List[str]]:
    """Group -> its labels, in codebook order."""
    out: Dict[str, List[str]] = {}
    for label in codebook(task)['labels']:
        out.setdefault(label['group'], []).append(label['id'])
    return out


# -----------------------------------------------------------------------------
# Profiles (what the annotator sees of an OA person)
# -----------------------------------------------------------------------------
FIELD_COLUMNS = [c for c in ['mathematics', 'computer_science', 'physics', 'biology', 'sociology', 'psychology']]


def load_profiles(con: duckdb.DuckDBPyConnection, authors_dir: Path | str, author_ids: Sequence[int]) -> pd.DataFrame:
    """A.1 rows (bibliometrics, field work counts, country) of the given authors."""
    ids = sorted({int(a) for a in author_ids if a is not None and not (isinstance(a, float) and math.isnan(a))})
    if not ids:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    con.register('wanted_ids', pd.DataFrame({'author_id': ids}))
    return con.execute(f"""
        SELECT a.* FROM read_parquet({sql_path(Path(authors_dir) / 'bucket-*.parquet')}, union_by_name = true) a
        JOIN wanted_ids USING (author_id)""").df()


def profile_text(p: pd.Series, reference_year: int = cons.GT_REFERENCE_YEAR) -> str:
    """Readable profile of one OA person (links are made clickable by the tool); p: an A.1 row (author_id as a
    column or as the row name)."""
    if 'author_id' not in p.index:
        p = pd.concat([pd.Series({'author_id': p.name}), p])
    def num(v):
        return '–' if v is None or (isinstance(v, float) and math.isnan(v)) else f"{int(v):,}"
    lines = [f"{p.get('display_name') or '(no display name)'}", f"OpenAlex: {OA_URL.format(int(p['author_id']))}"]
    if isinstance(p.get('orcid'), str) and p['orcid']:
        orcid = p['orcid'] if p['orcid'].startswith('http') else f"https://orcid.org/{p['orcid']}"
        lines.append(f"ORCID: {orcid}")
    first, last = p.get('first_pub_year'), p.get('last_pub_year')
    if first is not None and not (isinstance(first, float) and math.isnan(first)):
        last_text = '–' if last is None or (isinstance(last, float) and math.isnan(last)) else str(int(last))
        lines.append(f"Publications: {int(first)}–{last_text} (academic age {reference_year - int(first)} in {reference_year})")
    lines.append(f"Works: {num(p.get('n_works'))} · citations: {num(p.get('n_citations'))} · h-index: {num(p.get('h_index'))}")
    total = p.get('n_works_topic')
    shares = []
    for f in FIELD_COLUMNS:
        n = p.get(f"works_{f}")
        if total and n and not (isinstance(n, float) and math.isnan(n)) and n > 0:
            shares.append((n / total, f.replace('_', ' ').title(), int(n)))
    if shares:
        lines.append('Works in the study fields: ' + ', '.join(f"{name} {int(n)} ({s:.0%})" for s, name, n in sorted(shares, reverse=True)))
    country, last_known = p.get('country'), p.get('last_known_country')
    lines.append(f"Country: {country or '–'} (last known affiliation: {last_known or '–'})")
    return '\n'.join(lines)


def search_link(first: Optional[str], last: Optional[str]) -> str:
    from urllib.parse import quote
    return OA_SEARCH_URL.format(quote(f"{first or ''} {last or ''}".strip()))


# -----------------------------------------------------------------------------
# matching: sample and items
# -----------------------------------------------------------------------------
def match_stratum(df: pd.DataFrame) -> pd.Series:
    linked = df[['author_id', 'author_id_field', 'author_id_seniority']].notna().any(axis=1)
    rule = df['match_rule'].fillna('none').astype(str)
    return np.where(rule == 'none', 'none', rule + np.where(linked, '|linked', '|name'))


def linked_author(row: pd.Series) -> Optional[int]:
    for column in ('author_id', 'author_id_field', 'author_id_seniority'):
        if pd.notna(row.get(column)):
            return int(row[column])
    return None


def rule_condition(rule: str, params: Dict) -> str:
    if rule == cons.MATCH_FUZZY:
        parts = [f"coalesce(jw_full >= {float(params.get('jw_full', 0))}, false)"]
        for key in ('jw_first', 'jw_last'):
            if params.get(key) is not None:
                parts.append(f"coalesce({key} >= {float(params[key])}, false)")
        return ' AND '.join(parts)
    return f"coalesce({rule}, false)"


def sample_matching(matching_dir: Path | str, candidates_dir: Path | str, authors_dir: Path | str,
                    per_stratum: int, max_items: int, seed: int, models: Optional[List[str]] = None,
                    recommendations_metadata_dir: Optional[Path | str] = None,
                    valid_rec_tags: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per sampled distinct (rec_name, field, seniority) among the VALID recommendations (tags all in
    valid_rec_tags, as in the metrics; all if no metadata folder is given): identifiers of one occurrence, stratum,
    stratum size (distinct names in the population), and the sections shown."""
    from utils import ios
    params = json.loads((Path(matching_dir) / '_params.json').read_text()) if (Path(matching_dir) / '_params.json').exists() else {}
    columns = REC_ID_COLUMNS + ['first_name', 'last_name', 'rec_name', 'match_rule', 'author_id', 'author_id_field',
                                'author_id_seniority', 'matched_name']
    recs = ios.load_parquet_folder(matching_dir, columns, models)
    recs = recs[recs['rec_name'].notna()].copy()
    if recommendations_metadata_dir is not None:
        from responses.tag_stats import is_valid
        tags = ios.load_parquet_folder(recommendations_metadata_dir, REC_ID_COLUMNS + [cons.COL_REC_TAGS], models)
        valid = tags[tags[cons.COL_REC_TAGS].map(lambda t: is_valid(t, list(valid_rec_tags or cons.DEFAULT_VALID_REC_TAGS)))]
        as_text = lambda df: df[REC_ID_COLUMNS].astype(str).agg('\x1f'.join, axis=1)
        recs = recs[as_text(recs).isin(set(as_text(valid)))].copy()
    recs[STRATUM] = match_stratum(recs)
    distinct = recs.drop_duplicates(['rec_name', cons.VAR_FIELD, cons.VAR_SENIORITY]).reset_index(drop=True)
    distinct[STRATUM_SIZE] = distinct[STRATUM].map(distinct[STRATUM].value_counts())
    sample = stratified_sample(distinct, STRATUM, per_stratum, max_items, seed)

    con = duckdb.connect()
    con.register('sampled', sample[['rec_name']].drop_duplicates())
    candidates = con.execute(f"""
        SELECT c.* FROM read_parquet({sql_path(Path(candidates_dir) / 'bucket-*.parquet')}, union_by_name = true) c
        JOIN sampled USING (rec_name)""").df()
    shown_ids: List[List[int]] = []
    for _, row in sample.iterrows():
        linked = linked_author(row)
        own = candidates[candidates['rec_name'] == row['rec_name']]
        if linked is not None:
            ids = [linked]
        elif row['stratum'] != 'none':                      # name matches without a pick: people of the rule
            rule = str(row['match_rule'])
            con.register('own', own)
            chosen = con.execute(f"SELECT author_id FROM own WHERE {rule_condition(rule, params)}").df() \
                if len(own) else pd.DataFrame({'author_id': []})
            ids = sorted(chosen['author_id'].astype('int64'),
                         key=lambda a: hashlib.md5(f"{seed}|{row['rec_name']}|{int(a)}".encode()).hexdigest())[:MAX_SHOWN]
        else:                                               # closest names below the thresholds
            ids = list(own.sort_values('jw_full', ascending=False)['author_id'].astype('int64').head(MAX_SHOWN)) \
                if len(own) and own['jw_full'].notna().any() else []
        shown_ids.append(ids)
    sample['shown_author_ids'] = shown_ids
    profiles = load_profiles(con, authors_dir, [a for ids in shown_ids for a in ids]).set_index('author_id')
    reference_year = int(params.get('reference_year', cons.GT_REFERENCE_YEAR))
    sample['sections'] = [matching_sections(row, profiles, reference_year) for _, row in sample.iterrows()]
    return sample


def matching_sections(row: pd.Series, profiles: pd.DataFrame, reference_year: int) -> List[Section]:
    request = (f"First name: {row['first_name'] or '–'}\nLast name: {row['last_name'] or '–'}\n"
               f"Requested: {row[cons.VAR_FIELD]}" + (f" ({row[cons.VAR_SUBFIELD]})" if row.get(cons.VAR_SUBFIELD) else '')
               + f", {row[cons.VAR_SENIORITY]} scholars\nSearch OpenAlex: {search_link(row['first_name'], row['last_name'])}")
    sections = [Section('Recommended person', request, preformatted=True)]
    ids = row['shown_author_ids']
    texts = [profile_text(profiles.loc[a], reference_year) if a in profiles.index else f"OpenAlex: {OA_URL.format(a)}"
             for a in ids]
    if linked_author(row) is not None:
        sections.append(Section('OpenAlex person linked by the pipeline', texts[0], preformatted=True))
    elif row[STRATUM] != 'none':
        sections.append(Section(f"OpenAlex people with a matching name (not linked: field / seniority undecided; "
                                f"{len(ids)} shown)", '\n\n'.join(texts), preformatted=True))
    else:
        sections.append(Section('Nothing linked. Closest OpenAlex names (below the thresholds)',
                                '\n\n'.join(texts) if texts else '(no OpenAlex name close to it)', preformatted=True))
    return sections


# -----------------------------------------------------------------------------
# authors: sample and items
# -----------------------------------------------------------------------------
def sample_authors(gt_dir: Path | str, authors_dir: Path | str, per_stratum: int, max_items: int, seed: int,
                   reference_year: int = cons.GT_REFERENCE_YEAR) -> pd.DataFrame:
    members = pd.read_parquet(Path(gt_dir) / 'gt_members.parquet',
                              columns=['author_id', 'field', 'field_share', 'seniority', 'country'])
    members[STRATUM] = members['field'].astype(str) + '|' + members['seniority'].astype(str)
    members[STRATUM_SIZE] = members[STRATUM].map(members[STRATUM].value_counts())
    sample = stratified_sample(members, STRATUM, per_stratum, max_items, seed)
    profiles = load_profiles(duckdb.connect(), authors_dir, sample['author_id']).set_index('author_id')
    sample['sections'] = [authors_sections(row, profiles, reference_year) for _, row in sample.iterrows()]
    return sample


def authors_sections(row: pd.Series, profiles: pd.DataFrame, reference_year: int) -> List[Section]:
    a = int(row['author_id'])
    profile = profiles.loc[a] if a in profiles.index else pd.Series({'author_id': a})
    assigned = (f"Field: {row['field']} ({row['field_share']:.0%} of the works with a study-field topic)\n"
                f"Seniority: {row['seniority']}\nCountry: {row['country'] or '–'}")
    return [Section('OpenAlex profile', profile_text(profile, reference_year), preformatted=True),
            Section('Assigned by the pipeline (to check)', assigned, preformatted=True)]


# -----------------------------------------------------------------------------
# Items file and key
# -----------------------------------------------------------------------------
def items_payload(task: str, sample: pd.DataFrame) -> dict:
    items = [{ITEM_ID: row[ITEM_ID], 'sections': [s.to_dict() for s in row['sections']]} for _, row in sample.iterrows()]
    return {'codebook': codebook(task), 'items': items, 'tag_separator': cons.TAG_SEPARATOR}


def key_table(task: str, sample: pd.DataFrame) -> pd.DataFrame:
    if task == MATCHING:
        columns = [ITEM_ID] + REC_ID_COLUMNS + ['first_name', 'last_name', 'rec_name', 'match_rule', 'author_id',
                                                'author_id_field', 'author_id_seniority', STRATUM, STRATUM_SIZE]
        key = sample[[c for c in columns if c in sample.columns]].copy()
        key['shown_author_ids'] = sample['shown_author_ids'].map(lambda ids: ' '.join(str(a) for a in ids))
        return key
    return sample[[ITEM_ID, 'author_id', 'field', 'field_share', 'seniority', 'country', STRATUM, STRATUM_SIZE]].copy()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def group_answers(annotations: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per (item, annotator, group) with the chosen label ('' = none; > 1 label in a group = '')."""
    return group_answers_of(annotations, groups(task))


def group_answers_of(annotations: pd.DataFrame, label_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """group_answers for any group -> labels mapping (also used by the demographics task)."""
    rows = []
    for _, a in annotations[~annotations['unsure']].iterrows():
        chosen = set(t for t in str(a['tags']).split(cons.TAG_SEPARATOR) if t)
        for group, labels in label_groups.items():
            picked = [l for l in labels if l in chosen]
            rows.append({ITEM_ID: a[ITEM_ID], 'annotator': a['annotator'], 'group': group,
                         'label': picked[0] if len(picked) == 1 else ''})
    return pd.DataFrame(rows, columns=[ITEM_ID, 'annotator', 'group', 'label'])


def adjudicate_groups(answers: pd.DataFrame) -> pd.DataFrame:
    """Per item and group: the label chosen by more than half of the annotators who answered the group
    (is_tie when no label has a majority)."""
    answered = answers[answers['label'] != '']
    rows = []
    for (item, group), g in answered.groupby([ITEM_ID, 'group']):
        counts = g['label'].value_counts()
        top = counts.index[0]
        majority = counts.iloc[0] / len(g) > 0.5
        rows.append({ITEM_ID: item, 'group': group, 'label': top if majority else '', 'is_tie': not majority,
                     'n_annotators': len(g)})
    return pd.DataFrame(rows, columns=[ITEM_ID, 'group', 'label', 'is_tie', 'n_annotators'])


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def scopes(task: str) -> Dict[str, List[str]]:
    """Label -> the labels of its scope (a rate's denominator)."""
    labels = codebook(task)['labels']
    return {l['id']: [m['id'] for m in labels if m['scope'] == l['scope']] for l in labels}


def label_rates(key: pd.DataFrame, gold: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per group, stratum and label: n (items whose gold label is in the label's scope), k, share, Wilson CI; plus
    'overall (weighted)' rows: the stratum shares weighted by the stratum sizes, over the strata where the scope
    occurs (CI from the weighted per-stratum variances). E.g. correct_match = precision among the items with a person
    shown; missed_match = share among the items with nothing linked."""
    merged = gold[~gold['is_tie']].merge(key[[ITEM_ID, STRATUM, STRATUM_SIZE]], on=ITEM_ID)
    scope_of = scopes(task)
    rows = []
    for group, labels in groups(task).items():
        g_all = merged[merged['group'] == group]
        for label in labels:
            g = g_all[g_all['label'].isin(scope_of[label])]
            if g.empty:
                continue
            sizes = g.groupby(STRATUM)[STRATUM_SIZE].first()
            weighted, variance, weight = 0.0, 0.0, 0.0
            for stratum, s in g.groupby(STRATUM):
                n, k = len(s), int((s['label'] == label).sum())
                low, high = wilson(k, n)
                rows.append({'group': group, 'stratum': stratum, 'label': label, 'n': n, 'k': k, 'share': k / n,
                             'ci_low': low, 'ci_high': high, 'stratum_size': int(sizes[stratum])})
                w = float(sizes[stratum])
                weighted += w * k / n
                variance += w * w * (k / n) * (1 - k / n) / n
                weight += w
            share = weighted / weight
            half = 1.959964 * math.sqrt(variance) / weight
            rows.append({'group': group, 'stratum': 'overall (weighted)', 'label': label, 'n': len(g),
                         'k': int((g['label'] == label).sum()), 'share': share, 'ci_low': max(0.0, share - half),
                         'ci_high': min(1.0, share + half), 'stratum_size': int(sizes.sum())})
    return pd.DataFrame(rows)


def group_agreement(answers: pd.DataFrame) -> pd.DataFrame:
    """Per group: Krippendorff's alpha (nominal) over items rated by >= 2 annotators (empty if one annotator)."""
    import krippendorff
    if answers['annotator'].nunique() < 2:
        return pd.DataFrame(columns=['group', 'krippendorff_alpha', 'n_items_multiple_annotators'])
    rows = []
    for group, g in answers[answers['label'] != ''].groupby('group'):
        codes = {l: i for i, l in enumerate(sorted(g['label'].unique()))}
        table = g.assign(code=g['label'].map(codes)).pivot_table(index='annotator', columns=ITEM_ID, values='code',
                                                                  aggfunc='first')
        rated_twice = int((table.notna().sum(axis=0) >= 2).sum())
        data = table.to_numpy(dtype=float)
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='nominal') \
            if rated_twice and np.nanstd(data) > 0 else np.nan
        rows.append({'group': group, 'krippendorff_alpha': alpha, 'n_items_multiple_annotators': rated_twice})
    return pd.DataFrame(rows)
