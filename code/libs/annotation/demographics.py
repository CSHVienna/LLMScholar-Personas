"""
Demographics annotation (blind): the annotator sees ONLY a name and assigns the PERCEIVED gender and
the PERCEIVED ethnicity (never identity; 'unknown' = cannot tell from the name). The inferred labels (nomquamgender,
raceBERT race and ethnicity models; A.3-A.4) are never shown; they stay in the key.

Unit: one OpenAlex person, shown by the name the classifiers saw: the A.1 display name, on one line as written. Two
sides, so that the misclassification can be checked where it is used:
    gt           GT scholars (gt_authors)
    recommended  people recommended by the LLMs (B.6 matched author_id), also GT scholars; a person on both sides is
                 kept once, as recommended (the rarer side)
Strata: side x inferred gender x inferred race (rare combinations are all checked; at most per_stratum each); the
key keeps the stratum sizes, so the evaluation re-weights the sample to each side's population.

Ethnicity is annotated in origin-based classes (european, hispanic, portuguese_brazilian, mena, east_asian,
south_asian, black, aian, unknown) and compared with both inferred labels (evaluation group):
    race       the annotated class mapped onto the race classes (cons.ANNOTATION_RACE_MAPPINGS: 'main' = the US census
               convention the race model was trained on, the others robustness checks, never chosen by fit)
    ethnicity  the inferred 13 classes mapped onto the annotated ones (cons.ANNOTATION_FROM_ETHNICITY; the annotated
               classes the model cannot separate merged, cons.ANNOTATION_ETHNICITY_MERGE), mapping 'main'
'fine' = no mapping for either, to see where each group lands.

Evaluation (per mapping, side and group gender / race / ethnicity): Krippendorff's alpha between annotators; majority
label per item (ties left out); weighted confusion matrix annotated x inferred with P(inferred | annotated) and
P(annotated | inferred) (inputs of the Barron / Marshall correction, D.4) and the weighted accuracy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

from annotation.tasks import ITEM_ID, Section
from demographics import ethnicity as eth
from demographics import race
from groundtruth.db import sql_path
from utils import constants as cons

DEMOGRAPHICS = cons.ANNOTATION_TASK_DEMOGRAPHICS
INFERRED = ['gender', 'race', 'ethnicity']                    # labels of A.4 kept in the key
PROBABILITIES = ['p_female', *race.PROB_COLUMNS, *eth.PROB_COLUMNS]
SIDE_GT, SIDE_RECOMMENDED = 'gt', 'recommended'
STRATUM, STRATUM_SIZE = 'stratum', 'stratum_size'
GROUPS = {'gender': list(cons.ANNOTATION_GENDER_CLASSES), 'ethnicity': list(cons.ANNOTATION_ETHNICITY_CLASSES)}
GENDER_DEFINITIONS = {
    cons.GENDER_MALE: 'The name reads as a man\'s name.',
    cons.GENDER_FEMALE: 'The name reads as a woman\'s name.',
    cons.GENDER_NEUTRAL: 'The name is commonly used for both (e.g. Alex, Kim, Andrea across languages).',
    cons.GENDER_UNKNOWN: 'You cannot tell from the name (initials only, a name you do not know, not a person\'s name).',
}
ETHNICITY_DEFINITIONS = {   # by the language / region a name comes from; never by appearance
    cons.ANN_EUROPEAN: 'European names, incl. Anglo / North American of European origin, Slavic, Greek, Nordic '
                       '(not Spanish or Portuguese: see below).',
    cons.ANN_HISPANIC: 'Spanish-language names (Spain, Latin America).',
    cons.ANN_PORTUGUESE_BRAZILIAN: 'Portuguese-language names (Portugal, Brazil).',
    cons.ANN_MENA: 'Middle East / North Africa: Arabic, Persian, Turkish names.',
    cons.ANN_EAST_ASIAN: 'East / Southeast Asian names: Chinese, Japanese, Korean, Vietnamese, Thai, Indonesian, Filipino.',
    cons.ANN_SOUTH_ASIAN: 'South Asian names: Indian, Pakistani, Bangladeshi, Sri Lankan, Nepali.',
    cons.ANN_BLACK: 'African names (Sub-Saharan Africa) or names typical of the African diaspora.',
    cons.ANN_AIAN: 'Indigenous American names.',
    cons.ETHNICITY_UNKNOWN: 'You cannot tell from the name, or it fits none of the groups.',
}


# -----------------------------------------------------------------------------
# Codebook
# -----------------------------------------------------------------------------
def label_id(group: str, cls: str) -> str:
    """Tool label id, unique across groups (both groups have 'unknown'): e.g. gender_male, ethnicity_unknown."""
    return f"{group}_{cls}"


def class_of(label: str) -> Tuple[str, str]:
    group, _, cls = label.partition('_')
    return group, cls


def codebook(task: str = DEMOGRAPHICS) -> dict:
    definitions = {'gender': GENDER_DEFINITIONS, 'ethnicity': ETHNICITY_DEFINITIONS}
    labels = [{'id': label_id(group, cls), 'group': group, 'short': cls, 'definition': definitions[group][cls],
               'scope': group}
              for group, classes in GROUPS.items() for cls in classes]
    return {'task': task, 'unit': 'name', 'verification': False,
            'instructions': 'Only a name is shown. Pick the PERCEIVED gender and the PERCEIVED '
                            'ethnicity the name suggests to you: one label per group. This is about how the name '
                            'reads, not who the person is: do not look the person up. Use "unknown" when you cannot '
                            'tell.',
            'labels': labels,
            'rules': ['One label per group (gender, ethnicity); answer both.',
                      'Ethnicity = where the NAME comes from (language / region), e.g. a German name of a US scholar '
                      'is european; a mixed name (e.g. a Spanish given name and a Chinese family name): the family name.',
                      'Judge the name only: no search, no guessing from other knowledge of the person.',
                      'unknown = you cannot tell from the name; the "unsure" box = you are unsure about the item.'],
            'fields': []}


def groups(task: str = DEMOGRAPHICS) -> Dict[str, List[str]]:
    return {group: [label_id(group, c) for c in classes] for group, classes in GROUPS.items()}


# -----------------------------------------------------------------------------
# Sampling (DuckDB: gt_authors has tens of millions of rows)
# -----------------------------------------------------------------------------
def quota_for(sizes: pd.Series, per_stratum: int, max_items: int) -> int:
    """Per-stratum quota, lowered until the sample fits max_items (as tasks.stratified_sample)."""
    quota = per_stratum
    while quota > 1 and sizes.clip(upper=quota).sum() > max_items:
        quota -= 1
    return quota


def sample_demographics(gt_dir: Path | str, matching_dir: Optional[Path | str], per_stratum: int, max_items: int,
                        seed: int, threads: int = cons.GT_STATS_DUCKDB_THREADS,
                        memory_limit: str = cons.GT_STATS_DUCKDB_MEMORY) -> pd.DataFrame:
    """Stratified sample of people (side x inferred gender x inferred race) with their names and inferred
    labels / probabilities. Without B.6 outputs (matching_dir missing), only the GT side is sampled."""
    con = duckdb.connect(':memory:')
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    authors = f"read_parquet({sql_path(Path(gt_dir) / 'gt_authors.parquet')})"
    rec_files = sorted(Path(matching_dir).glob('*.parquet')) if matching_dir and Path(matching_dir).exists() else []
    if rec_files:
        con.execute(f"""CREATE TEMP TABLE rec AS SELECT DISTINCT author_id
                        FROM read_parquet({sql_path(Path(matching_dir) / '*.parquet')}) WHERE author_id IS NOT NULL""")
        rec_side = f"""UNION ALL SELECT a.author_id, '{SIDE_RECOMMENDED}' AS side, a.gender, a.race
                       FROM {authors} a SEMI JOIN rec USING (author_id)"""
    else:
        rec_side = ''
    con.execute(f"""CREATE TEMP VIEW pool AS
                    SELECT author_id, '{SIDE_GT}' AS side, gender, race FROM {authors} {rec_side}""")
    sizes = con.execute("SELECT side, gender, race, count(*) AS n FROM pool GROUP BY ALL").df()
    quota = quota_for(sizes['n'], per_stratum, max_items)
    drawn = con.execute(f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY side, gender, race ORDER BY hash(author_id + {int(seed)}))
                      AS r
            FROM pool)
        SELECT r.author_id, r.side, r.gender, r.race, a.ethnicity, a.display_name,
               {', '.join(f'a.{c}' for c in PROBABILITIES)}
        FROM ranked r JOIN {authors} a USING (author_id)
        WHERE r.r <= {quota}""").df()
    con.close()
    # a person drawn on both sides is kept once, as recommended
    drawn = drawn.sort_values('side', key=lambda s: s.ne(SIDE_RECOMMENDED), kind='stable') \
                 .drop_duplicates('author_id').reset_index(drop=True)
    drawn[STRATUM] = drawn['side'] + '|' + drawn['gender'].astype(str) + '|' + drawn['race'].astype(str)
    sizes[STRATUM] = sizes['side'] + '|' + sizes['gender'].astype(str) + '|' + sizes['race'].astype(str)
    drawn[STRATUM_SIZE] = drawn[STRATUM].map(sizes.set_index(STRATUM)['n']).astype('int64')
    drawn['sections'] = [[Section('Name', str(name).strip() or '–', preformatted=True)] for name in drawn['display_name']]
    return drawn.sample(frac=1, random_state=seed).reset_index(drop=True)


def items_payload(task: str, sample: pd.DataFrame) -> dict:
    items = [{ITEM_ID: row[ITEM_ID], 'sections': [s.to_dict() for s in row['sections']]} for _, row in sample.iterrows()]
    return {'codebook': codebook(task), 'items': items, 'tag_separator': cons.TAG_SEPARATOR}


def key_table(task: str, sample: pd.DataFrame) -> pd.DataFrame:
    columns = [ITEM_ID, 'author_id', 'side', 'display_name', *INFERRED, *PROBABILITIES, STRATUM, STRATUM_SIZE]
    return sample[[c for c in columns if c in sample.columns]].copy()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def gold_classes(gold: pd.DataFrame) -> pd.DataFrame:
    """Adjudicated labels (verification.adjudicate_groups) -> item, group, annotated class (ties left out)."""
    g = gold[~gold['is_tie'] & (gold['label'] != '')].copy()
    g['annotated'] = g['label'].map(lambda l: class_of(l)[1])
    return g[[ITEM_ID, 'group', 'annotated', 'n_annotators']]


def map_classes(items: pd.DataFrame, mapping: str) -> pd.DataFrame:
    """Rows of the mapping (module docstring): race rows with the annotated class mapped onto the race classes,
    ethnicity rows (main) with the inferred class mapped onto the annotated ones; 'fine' keeps every class as it is.
    Gender only under 'main'; ethnicity under 'main' and 'fine'; race under every mapping."""
    out = items.copy()
    if mapping == cons.ANNOTATION_MAPPING_FINE:
        return out[out['group'] != 'gender']
    is_race, is_eth = out['group'] == 'race', out['group'] == 'ethnicity'
    table = cons.ANNOTATION_RACE_MAPPINGS[mapping]
    out.loc[is_race, 'annotated'] = out.loc[is_race, 'annotated'].map(lambda c: table.get(c, c))
    if mapping != cons.ANNOTATION_MAPPING_MAIN:
        return out[is_race]
    out.loc[is_eth, 'inferred'] = out.loc[is_eth, 'inferred'].map(lambda c: cons.ANNOTATION_FROM_ETHNICITY.get(c, c))
    out.loc[is_eth, 'annotated'] = out.loc[is_eth, 'annotated'].map(lambda c: cons.ANNOTATION_ETHNICITY_MERGE.get(c, c))
    return out


def mappings() -> List[str]:
    return [*cons.ANNOTATION_RACE_MAPPINGS, cons.ANNOTATION_MAPPING_FINE]


def evaluate(key: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """summary and confusion for every mapping (column 'mapping'; see map_classes for the groups of each)."""
    items = weighted_items(key, gold)
    out = {'summary': [], 'confusion': []}
    for mapping in mappings():
        mapped = map_classes(items, mapping)
        out['summary'].append(summary(mapped).assign(mapping=mapping))
        out['confusion'].append(confusion(mapped).assign(mapping=mapping))
    return {name: pd.concat(parts, ignore_index=True).pipe(lambda d: d[['mapping', *d.columns.drop('mapping')]])
            for name, parts in out.items()}


def weighted_items(key: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, evaluation group) with annotated and inferred class (unmapped) and the item weight =
    stratum size / items of the stratum answered for that annotation group (re-weights the stratified sample to each
    side's population). The annotated ethnicity gives two rows: group race (vs the inferred race) and group
    ethnicity (vs the inferred ethnicity, when the key has it)."""
    inferred = [c for c in INFERRED if c in key]
    g = gold_classes(gold).merge(key[[ITEM_ID, 'side', *inferred, STRATUM, STRATUM_SIZE]], on=ITEM_ID)
    answered = g.groupby(['group', STRATUM])[ITEM_ID].transform('count')
    g['weight'] = g[STRATUM_SIZE] / answered
    targets = {'gender': ['gender'], 'ethnicity': [c for c in ('race', 'ethnicity') if c in inferred]}
    g = pd.concat([g[g['group'] == annotated].assign(group=target, inferred=lambda d, t=target: d[t])
                   for annotated, ts in targets.items() for target in ts], ignore_index=True)
    return g[[ITEM_ID, 'side', 'group', STRATUM, 'annotated', 'inferred', 'weight']]


def confusion(items: pd.DataFrame) -> pd.DataFrame:
    """Per side and group, annotated x inferred: n (items), weighted n, P(inferred | annotated) (rows; the
    misclassification matrix of Barron / Marshall) and P(annotated | inferred) (columns; how often a label is right)."""
    out = items.groupby(['side', 'group', 'annotated', 'inferred']).agg(n=(ITEM_ID, 'count'),
                                                                       weighted_n=('weight', 'sum')).reset_index()
    out['p_inferred_given_annotated'] = out['weighted_n'] / out.groupby(['side', 'group', 'annotated'])['weighted_n'].transform('sum')
    out['p_annotated_given_inferred'] = out['weighted_n'] / out.groupby(['side', 'group', 'inferred'])['weighted_n'].transform('sum')
    return out


def summary(items: pd.DataFrame) -> pd.DataFrame:
    """Per side and group: items, weighted accuracy (annotated == inferred), weighted share annotated as unknown."""
    rows = []
    for (side, group), g in items.groupby(['side', 'group']):
        w = g['weight'].sum()
        rows.append({'side': side, 'group': group, 'n_items': len(g),
                     'accuracy_weighted': float((g['weight'] * (g['annotated'] == g['inferred'])).sum() / w),
                     'accuracy_unweighted': float((g['annotated'] == g['inferred']).mean()),
                     'annotated_unknown_weighted': float((g['weight'] * (g['annotated'] == cons.GENDER_UNKNOWN)).sum() / w)})
    return pd.DataFrame(rows)
