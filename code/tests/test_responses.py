"""
Unit tests for response parsing, response tags and recommendation tags (synthetic fixtures only).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

from utils import constants as cons                                     # noqa: E402
from responses import tags as tag_utils                                  # noqa: E402
from responses.content_parser import ContentParser                       # noqa: E402
from responses.context import PromptContext                              # noqa: E402
from responses.persons import missing_fields                             # noqa: E402
from responses.recommendations import NameTagger, normalize_tokens, tag_recommendations  # noqa: E402
from utils.text import clean_text                                         # noqa: E402
from responses.sources import RawResponse, get_adapter                   # noqa: E402
from responses.taxonomy import classify                                  # noqa: E402

# A complete recommendation (all requested fields populated); its reason contains a refusal word
PERSON = {"name": "Ada", "lastname": "Lovelace",
          "current_affiliations": [{"position": "Professor", "affiliation": "University of London"}],
          "areas_of_research_or_work": ["Mathematics"], "reason": "She cannot be ignored.", "source": "N/A"}
C = cons  # short alias for readability below


def tags_of(text, **raw_fields):
    """Response tags of a text as if it came from any source, plus the parsed content."""
    raw = RawResponse(text=text, **raw_fields)
    parsed = ContentParser().parse(text) if text else None
    return classify(raw, parsed), parsed


class TestResponseTags(unittest.TestCase):

    def test_correct(self):
        self.assertEqual(tags_of(json.dumps([PERSON]))[0], [C.RESPONSE_CORRECT])
        # cosmetic cleaning is not a repair
        text = "<think>reasoning</think>\n```json\n" + json.dumps([PERSON]) + "\n```"
        self.assertEqual(tags_of(text)[0], [C.RESPONSE_CORRECT])

    def test_wrapped_is_not_correct(self):
        for value in ({"candidates": [PERSON]}, {"1": PERSON, "2": PERSON}, PERSON):
            tags, parsed = tags_of(json.dumps(value))
            self.assertEqual((tags, len(parsed.persons) >= 1), ([C.RESPONSE_WRAPPED], True), value)

    def test_completeness_is_not_a_response_tag(self):
        no_source = {k: v for k, v in PERSON.items() if k != 'source'}
        self.assertEqual(tags_of(json.dumps([PERSON, no_source]))[0], [C.RESPONSE_CORRECT])
        self.assertEqual(tags_of(json.dumps({"candidates": [{**PERSON, "reason": "N/A"}]}))[0], [C.RESPONSE_WRAPPED])

    def test_python_literal(self):
        self.assertEqual(tags_of(str([PERSON]))[0], [C.RESPONSE_PYTHON_LITERAL])          # single quotes

    def test_embedded(self):
        self.assertEqual(tags_of("Here you go: " + json.dumps([PERSON]))[0], [C.RESPONSE_EMBEDDED])

    def test_truncated_only_for_truncation(self):
        for cut in (30, 3):  # cut inside a string / right after a comma
            tags, parsed = tags_of(json.dumps([PERSON, PERSON])[:-cut])
            self.assertEqual((tags, parsed.method, len(parsed.persons)), ([C.RESPONSE_TRUNCATED], C.PARSE_TRUNCATED, 1))
        wrapped = json.dumps({"candidates": [PERSON, PERSON]})[:-40]
        self.assertEqual(tags_of(wrapped)[0], [C.RESPONSE_TRUNCATED, C.RESPONSE_WRAPPED])
        # broken in the middle (missing comma) is not truncation
        broken = json.dumps([PERSON, PERSON]).replace('}, {', '} {', 1)
        self.assertEqual(tags_of(broken)[0], [C.RESPONSE_INVALID])

    def test_toy(self):
        text = "I can generate a plausible example based on the criteria:\n" + json.dumps([PERSON])
        self.assertEqual(tags_of(text)[0], [C.RESPONSE_EMBEDDED, C.RESPONSE_TOY])
        self.assertEqual(tags_of(json.dumps([{**PERSON, "reason": "Perfil ficticio."}]))[0], [C.RESPONSE_TOY])

    def test_empty(self):
        for text in (None, "", "   ", "<think>only reasoning</think>"):
            self.assertEqual(tags_of(text)[0], [C.RESPONSE_EMPTY], text)

    def test_empty_list(self):
        for text in ('{"candidates": []}', '{"candidates": [], "professors": []}'):
            self.assertEqual(tags_of(text)[0], [C.RESPONSE_EMPTY_LIST], text)

    def test_refused(self):
        self.assertEqual(tags_of('{"error": "A reliable JSON array cannot be generated."}')[0], [C.RESPONSE_REFUSED])
        self.assertEqual(tags_of("Lo siento, no puedo ayudar con eso.")[0], [C.RESPONSE_REFUSED])
        self.assertEqual(tags_of(json.dumps([PERSON]), api_refusal="blocked")[0], [C.RESPONSE_REFUSED])

    def test_invalid(self):
        for text in ('{" } } ] ]} [ ] } ] ]', '{"names": ["Ada Lovelace"]}', "[]", "{}",
                     '{"name": "", "lastname": "N/A"}', '{"name": "First Name", "lastname": "Last Name"}',
                     '[{"name": "Candidate 1"}, {"name": "Another Candidate"}]'):
            self.assertEqual(tags_of(text)[0], [C.RESPONSE_INVALID], text)
        self.assertEqual(tags_of(None, api_error="500")[0], [C.RESPONSE_INVALID])


class TestRequiredFields(unittest.TestCase):

    def test_complete(self):
        self.assertEqual(missing_fields(PERSON), [])

    def test_only_absent_non_name_keys_in_prompt_order(self):
        person = {"name": "", "lastname": "Candidate 1", " Reason ": "", "source": "N/A"}
        self.assertEqual(missing_fields(person), ['current_affiliations', 'areas_of_research_or_work'])
        self.assertEqual(missing_fields({**PERSON, "reason": None, "source": [], "current_affiliations": ""}), [])


class TestTagEncoding(unittest.TestCase):

    def test_columns(self):
        columns = tag_utils.tag_columns([C.REC_REPETITION_REMOVED, C.REC_TITLE_REMOVED], C.REC_TAGS, C.REC_TAG_PREFIX)
        self.assertEqual(columns['rec_tags'], 'title_removed|repetition_removed')
        self.assertTrue(columns['rec_repetition_removed'] and columns['rec_title_removed'] and not columns['rec_correct'])


class TestNameTagger(unittest.TestCase):

    def assertTag(self, first, last, expected_first, expected_last, tags):
        tagged = NameTagger().tag(first, last)
        self.assertEqual((tagged.first_name, tagged.last_name, tagged.tags),
                         (expected_first, expected_last, tags), (first, last))

    def test_correct(self):
        self.assertTag("Ada", "Lovelace", "Ada", "Lovelace", [C.REC_CORRECT])

    def test_title_removed(self):
        self.assertTag("Dr. Ada", "Lovelace", "Ada", "Lovelace", [C.REC_TITLE_REMOVED])
        self.assertTag("Ada", "Lovelace, PhD", "Ada", "Lovelace", [C.REC_TITLE_REMOVED])

    def test_repetition_removed(self):
        self.assertTag("John Smith", "Smith", "John", "Smith", [C.REC_REPETITION_REMOVED])
        self.assertTag("Yamamoto Hiroshi", "Yamamoto", "Hiroshi", "Yamamoto", [C.REC_REPETITION_REMOVED])  # family name first
        self.assertTag("John", "John Smith", "John", "Smith", [C.REC_REPETITION_REMOVED])
        self.assertTag("Juan de la Cruz", "de la Cruz", "Juan", "de la Cruz", [C.REC_REPETITION_REMOVED])

    def test_repetition_whole_words_only(self):
        """A hyphenated / apostrophe word is one word: it is removed entirely or not at all, never emptying a part."""
        self.assertTag("Kevin O'Connell", "O'Connell", "Kevin", "O'Connell", [C.REC_REPETITION_REMOVED])
        self.assertTag("Dror Bar-Natan", "Bar-Natan", "Dror", "Bar-Natan", [C.REC_REPETITION_REMOVED])
        self.assertTag("Wei-Chen", "Chen", "Wei-Chen", "Chen", [C.REC_CORRECT])       # given name Wei-Chen
        self.assertTag("Fei", "Fei-Fei", "Fei", "Fei-Fei", [C.REC_CORRECT])
        self.assertTag("Fei", "Fei-Fei Li", "Fei", "Fei-Fei Li", [C.REC_CORRECT])
        self.assertTag("Kaito", "Ito", "Kaito", "Ito", [C.REC_CORRECT])              # never inside a word
        self.assertTag("Juan Carlos Suárez Zeira", "Suárez-Zeira", "Juan Carlos", "Suárez-Zeira",
                       [C.REC_REPETITION_REMOVED])                                       # tokens compared, cut between words
        self.assertTag("Willem*J.*Rousseau", "Rousseau", "Willem*J", "Rousseau", [C.REC_REPETITION_REMOVED])
        self.assertTag("Friedhelm Gödde", "Gödde", "Friedhelm", "Gödde", [C.REC_REPETITION_REMOVED])  # decomposed ö

    def test_double_lastname_cut_in_half(self):
        self.assertTag("Marie Skłodowska-Curie", "Curie", "Marie", "Skłodowska-Curie", [C.REC_REPETITION_REMOVED])
        self.assertTag("José María Torres-Pérez", "Pérez", "José María", "Torres-Pérez", [C.REC_REPETITION_REMOVED])
        self.assertEqual(NameTagger().tag("Pumla Gobodo-Madikizela", "Madikizela").fix, 'double_lastname_in_name')
        self.assertTag("Paul-Eckert", "Eckert", "Paul-Eckert", "Eckert", [C.REC_CORRECT])          # no word before it
        self.assertTag("Wolf-Christian Butterbach", "Wolf", "Wolf-Christian Butterbach", "Wolf",   # not the last word
                       [C.REC_CORRECT])

    def test_text_fixes(self):
        """Escapes written as text are decoded; invisible characters are removed (not turned into word breaks)."""
        self.assertTag("Carlos", "Garc\\u00EDa", "Carlos", "García", [C.REC_CORRECT])
        self.assertTag("Elisabeth", "​J​u​n​g", "Elisabeth", "Jung", [C.REC_CORRECT])  # not initials
        self.assertTag("Luis", "de Leó​n", "Luis", "de León", [C.REC_CORRECT])
        self.assertTag("Kazuo Miyashita", "Miya\xadshita", "Kazuo", "Miyashita", [C.REC_REPETITION_REMOVED])
        self.assertTag("Niko\xa0B.\xa0Ganz", "Ganz", "Niko B", "Ganz", [C.REC_REPETITION_REMOVED])
        self.assertEqual(clean_text("\\ud83d\\ude00 \\ud800 a\tb"), "\U0001F600 \\ud800 a b")   # pair decoded, lone kept
        self.assertEqual(clean_text("Claudia Y. Wilson-GarcÃ\xada"), "Claudia Y. Wilson-García")   # mojibake (ftfy)
        self.assertEqual(clean_text("RodrÃƒÂ\xadguez"), "Rodríguez")                             # doubly encoded
        self.assertEqual(clean_text("Tarcí\xadsio"), "Tarcísio")                                 # soft hyphen
        for name in ["Müller", "Zoë Çelik", "Łukasz Żółć", "김민수", "Wei Zhang", "Jean-Pierre O'Neil"]:
            self.assertEqual(clean_text(name), name)                                            # correct names unchanged
        self.assertTag("Claudia", "Wilson-GarcÃ\xada", "Claudia", "Wilson-García", [C.REC_CORRECT])
        self.assertEqual(normalize_tokens("Garc\\u00EDa​-López"), ['garcia', 'lopez'])

    def test_split(self):
        self.assertTag("Ludwig van Beethoven", None, "Ludwig", "van Beethoven", [C.REC_SPLIT])
        self.assertTag(None, "Juan Carlos de la Cruz", "Juan Carlos", "de la Cruz", [C.REC_SPLIT])
        self.assertTag("Smith, John", "", "John", "Smith", [C.REC_SPLIT])
        self.assertTag("Ada Lovelace", "Ada Lovelace", "Ada", "Lovelace", [C.REC_SPLIT])
        self.assertTag("Van Morrison", None, "Van", "Morrison", [C.REC_SPLIT])       # capitalised: not a particle

    def test_all_tags_are_kept(self):
        self.assertTag("Prof. Dr. Jörg Müller", "Muller", "Jörg", "Muller", [C.REC_TITLE_REMOVED, C.REC_REPETITION_REMOVED])
        self.assertTag("Dr. Ada Lovelace", None, "Ada", "Lovelace", [C.REC_TITLE_REMOVED, C.REC_SPLIT])

    def test_initials(self):
        self.assertTag("M.", "Smith", "M", "Smith", [C.REC_INITIALS_NAME])          # initials are never correct
        self.assertTag("Michael", "G.", "Michael", "G", [C.REC_INITIALS_LASTNAME])
        self.assertTag("S. M.", "D.", "S M", "D", [C.REC_INITIALS])
        self.assertTag("Professor", "M. T.", "M", "T", [C.REC_TITLE_REMOVED, C.REC_SPLIT, C.REC_INITIALS])

    def test_placeholder(self):
        self.assertTag("Candidate Name", "Lastname", None, None, [C.REC_PLACEHOLDER])
        self.assertTag("Candidato 1", "Candidato 1", None, None, [C.REC_PLACEHOLDER])
        self.assertTag("Ada", "Last Name", None, None, [C.REC_PLACEHOLDER])

    def test_na_nan_are_names_but_NA_is_missing(self):
        self.assertTag("Li", "Na", "Li", "Na", [C.REC_CORRECT])                     # Li Na
        self.assertTag("Jinkyung", "Na", "Jinkyung", "Na", [C.REC_CORRECT])         # Korean surname Na
        self.assertTag("Nan", "Zhang", "Nan", "Zhang", [C.REC_CORRECT])
        self.assertTag("Matthew Farrer", "NA", "Matthew", "Farrer", [C.REC_SPLIT])  # 'NA' = not available
        self.assertTag("Matthew Farrer", "N/A", "Matthew", "Farrer", [C.REC_SPLIT])

    def test_invalid(self):
        self.assertTag("Smith", None, None, "Smith", [C.REC_INVALID])
        self.assertTag("Professor", "Smith", None, "Smith", [C.REC_TITLE_REMOVED, C.REC_INVALID])
        self.assertTag("", "N/A", None, None, [C.REC_INVALID])

    def test_name_key_for_duplicates(self):
        persons = [{"name": "Ada", "lastname": "Lovelace"}, {"name": "Lovelace Ada", "lastname": ""},
                   {"name": "Dr. Ada", "lastname": "Lovelace"}, {"name": "Alan", "lastname": "Turing"}]
        rows = tag_recommendations(persons)
        self.assertEqual([row[C.COL_REC_TAGS] for row in rows], ['correct', 'split', 'title_removed', 'correct'])
        self.assertEqual([row['name_key'] for row in rows],
                         ['ada lovelace', 'ada lovelace', 'ada lovelace', 'alan turing'])  # duplicates found later

    def test_full_record_is_kept(self):
        person = {**PERSON, "email": "ada@example.org"}
        row = tag_recommendations([person])[0]
        self.assertEqual((row['extra_fields'], json.loads(row['record_json']), row['is_complete']),
                         ("email", person, True))


class TestModelMeanShares(unittest.TestCase):

    def setUp(self):
        import pandas as pd
        from responses import tag_stats
        self.tag_stats = tag_stats
        # model a: 100% correct; model b: 50% correct / 50% invalid (2 configurations x 2 runs each)
        rows = [('a', 'english', p, 'correct') for p in (1, 1, 2, 2)]
        rows += [('b', 'english', p, tag) for p, tag in ((1, 'correct'), (1, 'invalid'), (2, 'correct'), (2, 'invalid'))]
        self.df = pd.DataFrame(rows, columns=[C.COL_MODEL, C.VAR_LANGUAGE, C.COL_PROMPT_ID, C.COL_RESPONSE_TAGS])
        self.categories = tag_stats.CombinationCategories.from_combinations(self.df[C.COL_RESPONSE_TAGS], ['correct'])

    def test_mean_of_model_shares_with_ci(self):
        stats = self.tag_stats.model_mean_shares(self.df, C.COL_RESPONSE_TAGS, self.categories, n_boot=200, seed=1)
        correct = stats.set_index('category').loc['correct']
        self.assertAlmostEqual(correct['mean'], 75.0)                    # mean of 100% and 50%
        self.assertTrue(correct['ci_low'] <= correct['mean'] <= correct['ci_high'])
        self.assertEqual(int(correct['n_models']), 2)

    def test_reproducible_with_seed(self):
        first = self.tag_stats.model_mean_shares(self.df, C.COL_RESPONSE_TAGS, self.categories, n_boot=50, seed=7)
        second = self.tag_stats.model_mean_shares(self.df, C.COL_RESPONSE_TAGS, self.categories, n_boot=50, seed=7)
        self.assertTrue(first.equals(second))


class TestCaptions(unittest.TestCase):

    def test_response_and_recommendation_definitions_do_not_clash(self):
        import pandas as pd
        from responses import tag_stats
        from viz.captions import tags_figure_caption
        tags = pd.Series(['correct', 'wrapped', 'invalid'])
        categories = tag_stats.CombinationCategories.from_combinations(tags, C.DEFAULT_VALID_RESPONSE_TAGS)
        caption = tags_figure_caption('responses', categories, C.DEFAULT_VALID_RESPONSE_TAGS, 43, 1000, 1000)
        self.assertIn(C.RESPONSE_TAG_DEFINITIONS[C.RESPONSE_CORRECT], caption)
        self.assertNotIn(C.REC_TAG_DEFINITIONS[C.REC_CORRECT], caption)
        self.assertIn('$N_r$ = 1,000 responses from $N_m$ = 43 models', caption)


class TestAnnotationEvaluation(unittest.TestCase):

    def annotations(self, rows):
        import pandas as pd
        return pd.DataFrame(rows, columns=['item_id', 'annotator', 'tags', 'unsure'])

    def test_majority_tie_and_unsure(self):
        from annotation import evaluation
        rows = [('i1', 'a', 'wrapped', False), ('i1', 'b', 'wrapped', False), ('i1', 'c', 'invalid', False),  # majority
                ('i2', 'a', 'wrapped', False), ('i2', 'b', 'invalid', False),                                # tie
                ('i3', 'a', 'correct', True)]                                                                # only unsure
        gold = evaluation.adjudicate(self.annotations(rows), C.ANNOTATION_TASK_RESPONSES).set_index('item_id')
        self.assertEqual(gold.loc['i1', 'gold_tags'], 'wrapped')
        self.assertTrue(gold.loc['i2', 'is_tie'])
        self.assertNotIn('i3', gold.index)

    def test_answer_fields_need_a_majority(self):
        import pandas as pd
        from annotation import evaluation
        rows = pd.DataFrame([('i1', 'a', 'wrapped', False, '5'), ('i1', 'b', 'wrapped', False, '5'),
                             ('i2', 'a', 'wrapped', False, '3'), ('i2', 'b', 'wrapped', False, '4')],
                            columns=['item_id', 'annotator', 'tags', 'unsure', 'n_recommendations'])
        gold = evaluation.adjudicate(rows, C.ANNOTATION_TASK_RESPONSES).set_index('item_id')
        self.assertEqual((gold.loc['i1', 'gold_n_recommendations'], gold.loc['i2', 'gold_n_recommendations']), ('5', ''))

    def test_renamed_tags_in_saved_annotations(self):
        import tempfile
        from pathlib import Path
        from annotation import evaluation
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'annotation_responses_x.csv').write_text(
                'item_id,annotator,task,tags,unsure\ni1,x,responses,fixed|wrapped,0\n', encoding='utf-8')
            self.assertEqual(evaluation.load_annotations(folder, C.ANNOTATION_TASK_RESPONSES)['tags'][0], 'truncated|wrapped')

    def test_tag_metrics(self):
        import pandas as pd
        from annotation import evaluation
        compared = pd.DataFrame({'inferred_tags': ['wrapped', 'wrapped|toy', 'correct'],
                                 'gold_tags': ['wrapped', 'wrapped', 'correct']})
        metrics = evaluation.tag_metrics(compared, C.ANNOTATION_TASK_RESPONSES).set_index('tag')
        self.assertEqual((metrics.loc['wrapped', 'precision'], metrics.loc['wrapped', 'recall']), (1.0, 1.0))
        self.assertEqual(metrics.loc['toy', 'fp'], 1)


class TestParquetWriting(unittest.TestCase):

    def test_lone_surrogates_are_replaced(self):
        import tempfile
        import pandas as pd
        from utils import ios
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.parquet'
            ios.to_parquet(pd.DataFrame({'text': ['bad \udcaa x', None]}), path)
            self.assertEqual(pd.read_parquet(path)['text'].tolist(), ['bad \ufffd x', None])


class TestAdapters(unittest.TestCase):

    def test_gpt(self):
        record = {"response": {"status_code": 200, "body": {
            "created": 1766051415, "choices": [{"message": {"content": "[]", "refusal": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 0}}}},
            "error": None}
        raw = get_adapter(C.SOURCE_GPT).extract(record)
        self.assertEqual((raw.text, raw.created_at[:10], raw.completion_tokens), ("[]", "2025-12-18", 2))

    def test_gpt_refusal_field_with_content(self):
        record = {"response": {"body": {"choices": [{"message": {"content": None, "refusal": "[]"}}]}}}
        raw = get_adapter(C.SOURCE_GPT).extract(record)
        self.assertEqual((raw.text, raw.api_refusal), ("[]", None))

    def test_gemini(self):
        record = {"response": {"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}], "usageMetadata": {}}}
        raw = get_adapter(C.SOURCE_GEMINI).extract(record)
        self.assertEqual((raw.text, raw.finish_reason, raw.created_at), (None, "MAX_TOKENS", None))

    def test_ollama(self):
        record = {"created_at": "2025-11-07T18:21:08Z", "done_reason": "stop", "total_duration": 2e9,
                  "message": {"content": "{}"}}
        raw = get_adapter(C.SOURCE_OLLAMA).extract(record)
        self.assertEqual((raw.text, raw.duration_s), ("{}", 2.0))

    def test_malformed_record_never_raises(self):
        raw = get_adapter(C.SOURCE_GPT).extract({"response": {"body": {"choices": "oops"}}})
        self.assertIn("malformed", raw.api_error)


class TestPromptContext(unittest.TestCase):

    def test_translated_values_map_to_english(self):
        context = PromptContext(PROJECT_ROOT / 'data' / 'context')
        parameters = {"persona_context": {"role": "Director(a)/Reclutador(a)", "location": "Sudáfrica"},
                      "user_request": {"k": 1, "target": "Profesor(a) Júnior", "field": "Matemáticas",
                                       "subfield": "Teoría de números"}}
        self.assertEqual(context.canonical(C.LANG_ES, parameters),
                         {"language": "spanish", "role": "Director/Recruiter", "location": "South Africa", "k": 1,
                          "seniority": "junior", "field": "Mathematics", "subfield": "Number theory"})


if __name__ == '__main__':
    unittest.main()
