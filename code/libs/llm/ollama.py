"""
Ollama (batch_collect_local.py): one or more local servers on consecutive ports
(``first_port``, ``first_port + 1``, ...), called directly; answers arrive right away.

A pool of worker processes sends the prompts; each prompt goes to the alive server with the lowest load.
A watcher thread re-checks the servers every few seconds, so a server that went down is avoided and one that
came back is used again. Each answer is written as its own temp file (``ResponseStore.write_temp``) and merged
into the response file by the caller. A failing prompt never stops the others: workers return the error.
"""

import http.client
import multiprocessing
import os
import random
import socket
import threading
from typing import Dict, List, Optional, Tuple

from ollama import Client
from ollama._types import ResponseError

from llm.store import ResponseStore
from utils import ios
from utils import logs

WATCH_INTERVAL_S = 5
MAX_TASKS_PER_CHILD = 50


# -----------------------------------------------------------------------------
# Servers
# -----------------------------------------------------------------------------
def server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def is_port_alive(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if an Ollama server answers ``GET /api/version`` on the port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/api/version")
        status = conn.getresponse().status
        conn.close()
        return 200 <= status < 300
    except Exception:
        return False


def alive_ports(host: str, first_port: int, n_servers: int) -> List[int]:
    return [first_port + i for i in range(n_servers) if is_port_alive(host, first_port + i)]


def pull_missing(host: str, port: int, api_names: List[str]) -> List[str]:
    """Pull the models the server does not have yet (servers on one machine share the model folder).
    Returns the models pulled."""
    client, pulled = Client(host=server_url(host, port)), []
    for name in api_names:
        try:
            client.show(name)
        except ResponseError:
            ios.printf(f"Model {name} not on the server (port {port}): pulling ...")
            client.pull(name)
            pulled.append(name)
            ios.printf(f"Pulling {name} done.")
    return pulled


def chat_options(max_output_tokens: Optional[int] = None, num_thread: Optional[int] = None,
                 context_window: Optional[int] = None) -> Dict[str, int]:
    """Ollama options (only those given: the server defaults apply otherwise)."""
    options = {'num_predict': max_output_tokens, 'num_thread': num_thread, 'num_ctx': context_window}
    return {key: value for key, value in options.items() if value is not None}


def send_request(client: Client, api_name: str, prompt: str, options: dict) -> dict:
    """One chat request in JSON mode (as in the original collection); the whole answer as a dict."""
    response = client.chat(model=api_name, messages=[{"role": "user", "content": prompt.strip()}],
                           format="json", options=options)
    return response.model_dump(mode='json')


def port_watcher(port_status, port_load, host: str, first_port: int, lock, stop_event: threading.Event,
                 interval: float = WATCH_INTERVAL_S) -> None:
    """Thread: keep ``port_status`` up to date until ``stop_event`` is set."""
    while not stop_event.is_set():
        for i in range(len(port_status)):
            alive = is_port_alive(host, first_port + i)
            with lock:
                if port_status[i] != alive:
                    ios.printf(f"[watcher] port {first_port + i} -> {'alive' if alive else 'down'}")
                    port_status[i] = alive
        if random.random() < 0.2:
            ios.printf(f"[watcher] alive: {[first_port + i for i, s in enumerate(port_status) if s]} "
                       f"| load: {list(port_load)}")
        stop_event.wait(interval)


# -----------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------
_WORKER: dict = {}


def init_worker(port_status, port_load, lock, settings: dict) -> None:
    """Pool initializer: shared server state and the settings of this model x language."""
    logs.setup_worker_logging(settings.get('log_file'))
    _WORKER.update(port_status=port_status, port_load=port_load, lock=lock, settings=settings,
                   store=ResponseStore(settings['responses_dir'], settings['model'], settings['language']))


def process_prompt(item: dict) -> Tuple[str, Optional[str]]:
    """Send one prompt to the least-loaded alive server and store the answer. Returns (prompt id, error)."""
    port_status, port_load, lock = _WORKER['port_status'], _WORKER['port_load'], _WORKER['lock']
    settings, store = _WORKER['settings'], _WORKER['store']
    prompt_id = str(item['id'])

    with lock:
        alive = [(i, load) for i, load in enumerate(port_load) if port_status[i]]
        if not alive:
            return prompt_id, 'no server alive'
        idx = min(alive, key=lambda x: x[1])[0]
        port_load[idx] += 1
    port = settings['first_port'] + idx

    try:
        client = Client(host=server_url(settings['host'], port), timeout=settings['timeout'])
        response = send_request(client, settings['model'].api_name, item['prompt'], settings['options'])
        store.write_temp(prompt_id, response)
        ios.printf(f"[{settings['model'].slug}|{settings['language']}] prompt {prompt_id} done (port {port})")
        return prompt_id, None
    except Exception as error:
        logs.get_logger().error(f"[worker {os.getpid()}] port {port}, prompt {prompt_id}: {error!r}")
        with lock:
            port_status[idx] = False              # the watcher brings it back once it answers again
        return prompt_id, repr(error)
    finally:
        with lock:
            port_load[idx] -= 1


def run_pool(items: List[dict], settings: dict, n_servers: int, max_processes: int) -> Dict[str, str]:
    """Send ``items`` with a pool of ``max_processes`` workers over ``n_servers`` servers.

    ``settings``: responses_dir, model (ModelSpec), language, host, first_port, options, timeout, log_file.
    Returns {prompt id: error} of the prompts that failed (empty = all answered).
    """
    ctx = multiprocessing.get_context('spawn')
    failed = {}
    with ctx.Manager() as manager:
        port_status = manager.list([is_port_alive(settings['host'], settings['first_port'] + i)
                                    for i in range(n_servers)])
        port_load = manager.list([0] * n_servers)
        lock = manager.Lock()
        stop_event = threading.Event()
        watcher = threading.Thread(target=port_watcher, daemon=True,
                                   args=(port_status, port_load, settings['host'], settings['first_port'],
                                         lock, stop_event))
        watcher.start()
        try:
            with ctx.Pool(processes=max_processes, initializer=init_worker, maxtasksperchild=MAX_TASKS_PER_CHILD,
                          initargs=(port_status, port_load, lock, settings)) as pool:
                for prompt_id, error in pool.imap_unordered(process_prompt, items):
                    if error:
                        failed[prompt_id] = error
        finally:
            stop_event.set()
            watcher.join()
    return failed
