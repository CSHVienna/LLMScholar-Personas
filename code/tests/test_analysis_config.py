"""
Unit tests for the analysis config (libs/utils/analysis_config.py) and its wizard (scripts/configure_analysis.py).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'scripts'))

from utils import analysis_config as ac                                  # noqa: E402
from utils import constants as cons                                       # noqa: E402
import configure_analysis as wiz                                         # noqa: E402

N_SETTINGS = sum(len(section.settings) for section in ac.SCHEMA)


def scripted(answers):
    """ask() replacement answering from a list; the wizard's output is collected in `said`."""
    answers, said = list(answers), []
    return (lambda prompt: answers.pop(0)), said.append, said


class TestValues(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(ac.validate(ac.defaults()), [])

    def test_parse_value(self):
        s = ac.SETTINGS[('gender', 'p_female_edges')]
        self.assertEqual(ac.parse_value(s, '0.1, 0.4 0.6,0.9'), [0.1, 0.4, 0.6, 0.9])
        tags = ac.SETTINGS[('validity', 'valid_rec_tags')]
        self.assertEqual(ac.parse_value(tags, '1 title_removed'), [tags.choices[0], 'title_removed'])
        with self.assertRaises(ValueError):
            ac.parse_value(tags, 'nonsense')
        with self.assertRaises(ValueError):
            ac.parse_value(ac.SETTINGS[('field', 'min_field_share')], '1.5')
        labels = ac.SETTINGS[('gender', 'labels')]
        self.assertEqual(ac.parse_value(labels, 'male, mostly male, neutral'), ['male', 'mostly male', 'neutral'])

    def test_cross_checks(self):
        values = ac.defaults()
        values['gender']['labels'] = ['male', 'female']                  # 2 cut points need 3 labels
        values['seniority']['senior_min'] = 5                            # below junior_max
        ok = ac.defaults()
        ok['seniority']['single_cutoff'] = ok['seniority']['junior_max']   # single is independent of three
        self.assertEqual(ac.validate(ok), [])
        problems = ac.validate(values)
        self.assertTrue(any(p.startswith('gender:') for p in problems) and any(p.startswith('seniority:') for p in problems))

    def test_save_load_and_changed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.yaml'
            values = ac.defaults()
            values['gender'] = {'p_female_edges': [0.1, 0.4, 0.6, 0.9],
                                'labels': ['male', 'mostly male', 'neutral', 'mostly female', 'female']}
            ac.save(path, values)
            self.assertEqual(ac.load(path), values)
            self.assertIn('# [late] cut points on p_female', path.read_text())   # comments kept for the reader
            steps = ac.changed_steps(ac.defaults(), values)
            self.assertEqual(list(steps), [ac.STEP_POPULATION])
        with self.assertRaises(FileNotFoundError):
            ac.load(Path(tmp) / 'missing.yaml')

    def test_config_sets_defaults_command_line_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.yaml'
            values = ac.defaults()
            values['matching']['jw_first'] = 0.97
            values['matching']['seed'] = 7
            ac.save(path, values)
            parser = argparse.ArgumentParser()
            parser.add_argument('--jw_first', type=float, default=0.5)
            parser.add_argument('--seed', type=int, default=1)
            ac.add_config_argument(parser)
            args = ac.parse_with_config(parser, ['--config', str(path), '--seed', '3'])
            self.assertEqual((args.jw_first, args.seed), (0.97, 3))
            self.assertEqual(ac.parse_with_config(parser, []).jw_first, 0.97)   # defaults stay set on this parser


class TestWizard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'config' / 'analysis.yaml'

    def tearDown(self):
        self.tmp.cleanup()

    def test_enter_everywhere_saves_the_defaults(self):
        ask, say, said = scripted([''] * N_SETTINGS + ['s'])
        self.assertEqual(wiz.wizard(self.path, ask, say), ac.defaults())
        self.assertEqual(ac.load(self.path), ac.defaults())

    def test_invalid_answer_is_asked_again_and_help(self):
        # first setting: '?', then an unknown tag, then a valid answer; the rest Enter
        ask, say, said = scripted(['?', 'bogus', 'correct wrapped'] + [''] * (N_SETTINGS - 1) + ['s'])
        values = wiz.wizard(self.path, ask, say)
        self.assertEqual(values['validity']['valid_response_tags'], ['correct', 'wrapped'])
        self.assertTrue(any('✗' in line for line in said))

    def test_back_and_quit_writes_nothing(self):
        ask, say, said = scripted(['', 'b', '', '', 'q'])
        self.assertIsNone(wiz.wizard(self.path, ask, say))
        self.assertFalse(self.path.exists())

    def test_cross_check_repeats_the_section(self):
        n_before = sum(len(s.settings) for s in ac.SCHEMA[:3])                     # validity, seniority, field
        gender = ['0.1 0.9', 'male female']                                         # 2 labels for 3 intervals: rejected
        fixed = ['', 'male neutral female']
        rest = [''] * (N_SETTINGS - n_before - 2)
        ask, say, said = scripted([''] * n_before + gender + fixed + rest + ['s'])
        values = wiz.wizard(self.path, ask, say)
        self.assertEqual(values['gender']['labels'], ['male', 'neutral', 'female'])
        self.assertTrue(any('revise this section' in line for line in said))

    def test_editing_lists_the_steps_to_rerun(self):
        ac.save(self.path, ac.defaults())
        i = [s.key for s in ac.SCHEMA].index('matching')
        n_before = sum(len(s.settings) for s in ac.SCHEMA[:i])                      # everything before 'matching'
        n_after = sum(len(s.settings) for s in ac.SCHEMA[i:]) - 1                   # jw_first is matching's first
        ask, say, said = scripted([''] * n_before + ['0.95'] + [''] * n_after + ['s'])
        values = wiz.wizard(self.path, ask, say)
        self.assertEqual(values['matching']['jw_first'], 0.95)
        self.assertTrue(any(ac.STEP_AUTHORS in line for line in said))



class TestNewSettings(unittest.TestCase):
    def test_old_file_missing_settings(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.yaml'
            old = ac.defaults()
            old['matching']['seed'] = 7
            del old['social']                                            # written before the social section existed
            path.write_text(yaml.safe_dump(old))
            with self.assertRaises(ValueError) as e:
                ac.load(path)
            self.assertIn('social.include_unknown', str(e.exception))
            values, missing = ac.load_partial(path)
            self.assertEqual(values['matching']['seed'], 7)               # answers kept
            self.assertIn('social.parity_target', missing)
            ask, say, said = scripted([''] * N_SETTINGS + ['s'])
            saved = wiz.wizard(path, ask, say)                            # the wizard adds the new settings
            self.assertEqual((saved['matching']['seed'], saved['social']['include_unknown']), (7, False))
            self.assertEqual(ac.load(path)['social']['parity_target'], 'field_x_seniority')
            self.assertTrue(any('New settings' in line for line in said))


class TestPhases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'analysis.yaml'

    def tearDown(self):
        self.tmp.cleanup()

    def n(self, phase):
        return sum(len(s.settings) for s in ac.phase_settings(phase))

    def test_split(self):
        self.assertEqual(self.n('early') + self.n('late'), N_SETTINGS)
        late = set(ac.late_keys())
        self.assertTrue({'gender.p_female_edges', 'matching.jw_first', 'validity.valid_rec_tags'} <= late)
        self.assertTrue({'seniority.junior_max', 'prominence.bin_labels', 'matching.candidates_floor'}.isdisjoint(late))

    def test_early_then_late(self):
        ask, say, said = scripted([''] * self.n('early') + ['s'])
        wiz.wizard(self.path, ask, say, phase='early')
        self.assertFalse(ac.late_reviewed(self.path))
        self.assertEqual(ac.load(self.path), ac.defaults())                # complete: the pipeline runs
        with self.assertRaises(ValueError) as e:
            ac.require_late_reviewed(self.path)
        self.assertIn('--phase late', str(e.exception))
        self.assertIn('PROVISIONAL', self.path.read_text())
        # late: first late setting is valid_response_tags; change it, Enter for the rest
        ask, say, said = scripted(['correct wrapped'] + [''] * (self.n('late') - 1) + ['s'])
        values = wiz.wizard(self.path, ask, say, phase='late')
        self.assertEqual(values['validity']['valid_response_tags'], ['correct', 'wrapped'])
        ac.require_late_reviewed(self.path)
        self.assertTrue(any(ac.STEP_METRICS in line for line in said))     # steps to re-run are listed
        # editing early settings again keeps the review
        ask, say, said = scripted([''] * self.n('early') + ['s'])
        wiz.wizard(self.path, ask, say, phase='early')
        self.assertTrue(ac.late_reviewed(self.path))

    def test_late_needs_a_file_and_old_files_count_as_reviewed(self):
        with self.assertRaises(SystemExit):
            wiz.wizard(self.path, *scripted([])[:2], phase='late')
        import yaml
        self.path.write_text(yaml.safe_dump(ac.defaults()))                # written before the phases existed
        ac.require_late_reviewed(self.path)

    def test_defaults_are_provisional(self):
        ac.save(self.path, ac.defaults(), reviewed=False)
        self.assertFalse(ac.late_reviewed(self.path))


class TestVariants(unittest.TestCase):
    """Variants: config/variants/<name>.yaml (here in a temporary folder), reuse of the base stages, the header."""

    def setUp(self):
        import configure_variant as cv
        self.cv = cv
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.base = root / 'analysis.yaml'
        ac.save(self.base, ac.defaults())
        self.saved_dir, ac.VARIANTS_DIR = ac.VARIANTS_DIR, root / 'variants'
        self.results = root / 'results'

    def tearDown(self):
        ac.VARIANTS_DIR = self.saved_dir
        self.tmp.cleanup()

    def create(self, name, *settings, overwrite=False):
        self.cv.create(argparse.Namespace(name=name, set=list(settings), base=str(self.base), overwrite=overwrite))
        return ac.variant_info(name, self.results)

    def test_codes(self):
        self.assertEqual(set(ac.CODES), set(ac.SETTINGS))
        self.assertEqual(len(set(ac.CODES.values())), len(ac.CODES))

    def test_stages_reused_or_own(self):
        tags = self.create('tags', 'validity.valid_rec_tags=correct')
        self.assertEqual((tags['population'], tags['authors']), ('main', 'main'))       # only the metrics change
        self.assertEqual(tags['differs'], 'ctags=correct')
        self.assertEqual(tags['gt_dir'], str(self.results / 'groundtruth' / 'population' / 'main'))
        self.assertEqual(tags['metrics_dir'], str(self.results / 'metrics' / 'tags'))
        gender = self.create('pf02', 'pf=0.2,0.8')                                        # code instead of section.key
        self.assertEqual((gender['population'], gender['authors']), ('pf02', 'main'))
        self.assertEqual(gender['gt_dir'], str(self.results / 'groundtruth' / 'population' / 'pf02'))
        jw = self.create('jw097', 'matching.jw_full=0.97', 'dup=yes')
        self.assertEqual((jw['population'], jw['authors']), ('main', 'jw097'))
        self.assertEqual(jw['differs'], 'jwfull=0.97, dup=yes')
        senior = self.create('single', 'seniority.matching_definition=single')
        self.assertEqual(senior['authors'], 'single')
        self.assertEqual(sorted(ac.list_variants()), ['jw097', 'main', 'pf02', 'single', 'tags'])

    def test_refusals(self):
        with self.assertRaises(SystemExit):                                               # shared candidate step
            self.create('floor', 'matching.candidates_floor=0.85')
        with self.assertRaises(SystemExit):                                               # nothing differs
            self.create('same', 'matching.seed=42')
        with self.assertRaises(SystemExit):
            self.create('main', 'dup=yes')
        with self.assertRaises(SystemExit):
            self.create('bad', 'nosuch=1')
        with self.assertRaises(ValueError):
            ac.check_variant_name('../x')
        self.create('v', 'dup=yes')
        self.create('v', 'dup=yes')                                                       # same settings: kept
        with self.assertRaises(SystemExit):                                               # same name, other settings
            self.create('v', 'order=yes')
        info = self.create('v', 'order=yes', overwrite=True)
        self.assertEqual(info['differs'], 'order=yes')

    def test_header_survives_edits_and_is_needed(self):
        self.create('pf', 'pf=0.3,0.7')
        path = ac.variant_file('pf')
        values = ac.load(path)
        values['ethnicity']['min_prob'] = 0.6                                             # e.g. the wizard edits it
        ac.save(path, values)
        info = ac.variant_info('pf', self.results)
        self.assertEqual(info['differs'], 'pf=0.3-0.7, eprob=0.6')
        self.assertIn('variant: pf', path.read_text())
        bare = ac.VARIANTS_DIR / 'bare.yaml'
        ac.save(bare, ac.defaults())                                                     # no header: refused
        with self.assertRaises(ValueError):
            ac.variant_info('bare', self.results)

    def test_paths_for_the_runners(self):
        import contextlib
        import io
        self.create('tags', 'ctags=correct')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.cv.paths(argparse.Namespace(name='tags', results_dir=str(self.results), config=None))
        lines = dict(line.split('=', 1) for line in out.getvalue().split())
        self.assertEqual((lines['VARIANT_POPULATION'], lines['VARIANT_AUTHORS'], lines['VARIANT_METRICS']),
                         ('main', 'main', 'tags'))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.cv.paths(argparse.Namespace(name='nosuch', results_dir=str(self.results), config=None))

    def test_shared_gt_dir(self):
        from groundtruth.db import shared_gt_dir
        self.assertEqual(shared_gt_dir('/r/groundtruth/population/main'), Path('/r/groundtruth'))
        self.assertEqual(shared_gt_dir('/r/groundtruth'), Path('/r/groundtruth'))

    def test_plots_dir(self):
        path = ac.DEFAULT_CONFIG_FILE
        self.assertEqual(ac.variant_info('main', self.results, path)['plots_dir'], str(self.results / 'plots' / 'main'))
        self.assertEqual(ac.variant_info('main', config=path)['plots_dir'], str(cons.PROJECT_ROOT / 'plots' / 'main'))

    def test_project_configs_are_complete(self):
        for path in (ac.DEFAULT_CONFIG_FILE, ac.SMOKE_CONFIG_FILE):
            ac.load(path)
            self.assertEqual(ac.variant_info('main', self.results, path)['population'], 'main')


if __name__ == '__main__':
    unittest.main()
