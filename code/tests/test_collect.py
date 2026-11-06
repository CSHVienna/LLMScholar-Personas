"""
Unit tests for the inputs and the collection of raw responses: model names (llm/models.py), prompt
records (prompt/generation.py), the response store (llm/store.py), cloud batch bookkeeping (llm/cloud.py),
Ollama workers against a fake local server (llm/ollama.py), and API keys from the environment (utils/config.py).
No real API is called.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

from llm import cloud                                                    # noqa: E402
from llm import models as lm                                             # noqa: E402
from llm import ollama as ol                                             # noqa: E402
from llm.store import ResponseStore, load_prompts                        # noqa: E402
from prompt import generation as gen                                     # noqa: E402
from utils import constants as cons                                      # noqa: E402
from utils import ios                                                    # noqa: E402
from utils.config import read_api_key                                    # noqa: E402

GPT = lm.ModelSpec(slug='gpt-4_1-nano-2025-04-14', api='gpt', api_name='gpt-4.1-nano-2025-04-14')
GEMINI = lm.ModelSpec(slug='gemini-2_5-flash', api='gemini', api_name='gemini-2.5-flash')
OLLAMA = lm.ModelSpec(slug='gemma3-4b-it-q4_K_M', api='ollama', api_name='gemma3:4b-it-q4_K_M')

INSTRUCTIONS = [{"role": "PhD student", "task": "seeking an advisor", "targets": ["Junior Professor", "Senior Professor"]}]
LOCATIONS = ["Germany", "Japan"]
INPUTS = {"k": [1, 5], "fields": [{"field": "Physics", "subfields": ["Education"]}]}


def prompts(n=4):
    return [{'id': i, 'parameters': {'persona_context': {}, 'user_request': {'k': i}}, 'prompt': f"prompt {i}"}
            for i in range(n)]


def gpt_line(prompt_id, text='[]'):
    return {'id': f'batch_req_{prompt_id}', 'custom_id': str(prompt_id),
            'response': {'status_code': 200, 'body': {'choices': [{'message': {'content': text}}]}}, 'error': None}


class TestModels(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(lm.model_slug('llama3.2:3b-instruct-q4_K_M'), 'llama3_2-3b-instruct-q4_K_M')
        self.assertEqual(lm.model_slug('gpt-4.1-nano-2025-04-14'), 'gpt-4_1-nano-2025-04-14')

    def test_validate_and_spec(self):
        meta = {GPT.slug: {'api': 'gpt', 'api_name': GPT.api_name},
                'bad-slug': {'api': 'ollama', 'api_name': 'other:1b'},
                'no-api': {'api_name': 'no-api'}}
        problems = lm.validate_metadata(meta)
        self.assertEqual(len(problems), 2)
        self.assertEqual(lm.spec(meta, GPT.slug), GPT)
        with self.assertRaises(ValueError):
            lm.spec(meta, 'bad-slug')
        with self.assertRaises(KeyError):
            lm.spec(meta, 'unknown')
        self.assertEqual(lm.slugs(meta, ['gpt']), [GPT.slug])

    def test_project_metadata(self):
        """The real metadata: complete, and its keys are models.txt."""
        meta = lm.load_metadata(cons.DEFAULT_MODELS_METADATA_FILE)
        self.assertEqual(lm.validate_metadata(meta), [])
        models = [m for m in Path(cons.DEFAULT_MODELS_FILE).read_text().splitlines() if m.strip()]
        self.assertEqual(sorted(models), lm.slugs(meta))


class TestPrompts(unittest.TestCase):
    def test_records(self):
        records = gen.build_prompt_records(INSTRUCTIONS, LOCATIONS, INPUTS, cons.LANG_EN)
        self.assertEqual(len(records), 1 * 2 * 2 * 1 * 2)           # role x locations x k x subfields x targets
        self.assertEqual([r['id'] for r in records], list(range(len(records))))
        first = records[0]
        self.assertEqual(set(first), {'id', 'parameters', 'prompt'})
        self.assertEqual(first['parameters']['persona_context']['location'], 'Germany')
        self.assertEqual(first['parameters']['user_request'],
                         {'k': 1, 'target': 'Junior Professor', 'field': 'Physics', 'subfield': 'Education'})
        self.assertTrue(first['prompt'].startswith("You are a PhD student seeking an advisor in Germany.\n\nIdentify 1 "))
        self.assertIn("5 Junior Professors", records[2]['prompt'])

    def test_read_from_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'english'
            ios.save_json(INSTRUCTIONS, folder / 'instructions.json')
            ios.save_json({'locations': LOCATIONS}, folder / 'locations.json')
            ios.save_json(INPUTS, folder / 'input.json')
            records = gen.read_prompt_records(folder, cons.LANG_EN)
            self.assertEqual(records, gen.build_prompt_records(INSTRUCTIONS, LOCATIONS, INPUTS))
            ios.save_json(records, folder / cons.PROMPTS_FILE)
            self.assertEqual(load_prompts(tmp, 'english', limit=3), records[:3])


class TestStore(unittest.TestCase):
    def test_paths_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GPT, 'english')
            self.assertEqual(store.path, Path(tmp) / 'results_gpt_english' / 'gpt_english_gpt-4_1-nano-2025-04-14.json')
            store.add_prompts(prompts(3))
            store.add_response(1, {'a': 1})
            store.save()
            again = ResponseStore(tmp, GPT, 'english').load()
            self.assertEqual(again.records['1']['responses'], [{'a': 1}])
            self.assertEqual(again.records['0']['model'], GPT.api_name)
            self.assertEqual(list(again.records['0']), ['id', 'parameters', 'prompt', 'language', 'model', 'responses'])
            self.assertEqual([p['id'] for p in again.missing(prompts(3), 1)], [0, 2])
            self.assertEqual(again.missing(prompts(3), 2), prompts(3))
            with self.assertRaises(KeyError):
                again.add_response(99, {})

    def test_prompt_change_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GPT, 'english')
            store.add_prompts(prompts(2))
            changed = prompts(2)
            changed[1]['prompt'] = 'another prompt'
            with self.assertRaises(ValueError):
                store.add_prompts(changed)

    def test_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = ResponseStore(tmp, OLLAMA, 'german')             # workers never load the records
            for i in [2, 0, 2]:
                worker.write_temp(i, {'answer': i})
            store = ResponseStore(tmp, OLLAMA, 'german')
            store.add_prompts(prompts(3))
            self.assertEqual(store.merge_temp(), 3)
            self.assertEqual(store.n_runs(2), 2)
            self.assertEqual(list(store.temp_dir.glob('*.json')), [])
            self.assertEqual(ResponseStore(tmp, OLLAMA, 'german').load().n_runs(0), 1)   # saved
            ResponseStore(tmp, OLLAMA, 'english').write_temp(0, {})  # other language: own folder
            self.assertEqual(store.merge_temp(), 0)


class TestCloud(unittest.TestCase):
    def test_prepare_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GPT, 'english')
            written = cloud.prepare_batches(store, prompts(4), repetitions=3)
            self.assertEqual(written, {1: 4, 2: 4, 3: 4})
            open_ = cloud.open_batches(store)
            self.assertEqual([f.run for f in open_], [1, 2, 3])
            line = json.loads(open_[0].requests.read_text().splitlines()[0])
            self.assertEqual(line['custom_id'], '0')
            self.assertEqual(line['body']['model'], GPT.api_name)
            self.assertEqual(line['body']['response_format'], {'type': 'json_object'})
            self.assertEqual(cloud.prepare_batches(store, prompts(4), repetitions=3), {})       # all queued

            # run 2 finishes first (prompt 3 failed in it), run 1 still open
            cloud.merge_output(store, open_[1], '\n'.join(json.dumps(gpt_line(i)) for i in range(3)) + '\n')
            self.assertTrue((store.folder / '_batches' / 'done' / (open_[1].name + '.responses.jsonl')).exists())
            self.assertEqual([store.n_runs(i) for i in range(4)], [1, 1, 1, 0])
            # only prompt 3 is short: 0 answers + 2 open batches (runs 1, 3) < 3
            written = cloud.prepare_batches(store, prompts(4), repetitions=3)
            self.assertEqual(written, {2: 1})
            self.assertEqual(json.loads(cloud.batch_files(store, 2).requests.read_text())['custom_id'], '3')

    def test_exact_repetitions_any_order(self):
        """Batches finish in random order and lose random answers: prepare + merge until nothing is open ->
        every prompt has exactly `repetitions` answers."""
        import random
        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GPT, 'english')
            for _ in range(50):
                cloud.prepare_batches(store, prompts(6), repetitions=4)
                open_ = cloud.open_batches(store)
                if not open_:
                    break
                files = rng.choice(open_)
                ids = [json.loads(l)['custom_id'] for l in files.requests.read_text().splitlines()]
                kept = [i for i in ids if rng.random() < 0.7]                 # failed requests are not in the output
                cloud.merge_output(store, files, ''.join(json.dumps(gpt_line(i)) + '\n' for i in kept))
            self.assertEqual(cloud.open_batches(store), [])
            self.assertEqual([store.n_runs(i) for i in range(6)], [4] * 6)

    def test_overwrite_only_unsubmitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GEMINI, 'english')
            cloud.prepare_batches(store, prompts(2), repetitions=2)
            ios.save_json({'job_id': 'batches/1'}, cloud.batch_files(store, 1).submitted)
            written = cloud.prepare_batches(store, prompts(3), repetitions=2, overwrite=True)
            self.assertEqual(written, {2: 3})                         # run 1 is submitted: kept
            line = json.loads(cloud.batch_files(store, 2).requests.read_text().splitlines()[2])
            self.assertEqual(line['key'], '2')
            self.assertEqual(line['request']['generation_config'], {'responseMimeType': 'application/json'})

    def test_archive_and_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResponseStore(tmp, GEMINI, 'english')
            for _ in range(2):
                cloud.prepare_batches(store, prompts(1), repetitions=1)
                files = cloud.open_batches(store)[0]
                target = cloud.archive(files, cons.BATCHES_FAILED_DIR)
            self.assertTrue(target.name.endswith('__run01_2'))
            self.assertEqual(target.run, 1)
            self.assertEqual(cloud.open_batches(store), [])
            answers = cloud.parse_output('gemini', '{"key": "0", "response": {}}\n\n{"key": "1", "response": {}}')
            self.assertEqual([a[0] for a in answers], ['0', '1'])
            with self.assertRaises(ValueError):
                cloud.provider('ollama')


class _FakeOllama(BaseHTTPRequestHandler):
    """Answers /api/version and /api/chat like an Ollama server (prompt 'fail' -> HTTP 500)."""

    def do_GET(self):
        self._send(200, {'version': '0.0.0'})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        prompt = body['messages'][0]['content']
        if 'fail' in prompt:
            self._send(500, {'error': 'boom'})
            return
        self._send(200, {'model': body['model'], 'created_at': '2026-01-01T00:00:00Z', 'done': True,
                         'done_reason': 'stop', 'eval_count': 3,
                         'message': {'role': 'assistant', 'content': f'["{prompt}"]'}})

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestOllama(unittest.TestCase):
    def test_options(self):
        self.assertEqual(ol.chat_options(), {})
        self.assertEqual(ol.chat_options(max_output_tokens=10, context_window=4096), {'num_predict': 10, 'num_ctx': 4096})

    def test_pool_with_fake_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), _FakeOllama)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertTrue(ol.is_port_alive('127.0.0.1', port))
            self.assertFalse(ol.is_port_alive('127.0.0.1', port + 1) if port + 1 < 65536 else False)
            items = prompts(3) + [{'id': 3, 'parameters': {}, 'prompt': 'please fail'}]
            with tempfile.TemporaryDirectory() as tmp:
                settings = {'responses_dir': tmp, 'model': OLLAMA, 'language': 'english', 'host': '127.0.0.1',
                            'first_port': port, 'options': {}, 'timeout': 10, 'log_file': None}
                failed = ol.run_pool(items, settings, n_servers=1, max_processes=2)
                self.assertEqual(list(failed), ['3'])
                store = ResponseStore(tmp, OLLAMA, 'english')
                store.add_prompts(items)
                self.assertEqual(store.merge_temp(), 3)
                answer = store.records['1']['responses'][0]
                self.assertEqual(answer['message']['content'], '["prompt 1"]')
                self.assertEqual(answer['done_reason'], 'stop')
        finally:
            server.shutdown()
            server.server_close()


class TestConfig(unittest.TestCase):
    def test_key_from_environment(self):
        saved = os.environ.pop('GEMINI_API_KEY', None)
        try:
            with self.assertRaises(RuntimeError):
                read_api_key('gemini')
            os.environ['GEMINI_API_KEY'] = ' env-key\n'
            self.assertEqual(read_api_key('gemini'), 'env-key')
        finally:
            os.environ.pop('GEMINI_API_KEY', None)
            if saved is not None:
                os.environ['GEMINI_API_KEY'] = saved

if __name__ == '__main__':
    unittest.main()
