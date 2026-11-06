#!/usr/bin/env python3
"""
Serve the annotation tool from the server, for annotating remotely in a local browser.

Serves code/tools/annotation_tool.html and the items files of <annotations_dir>, and saves the
annotations posted by the tool as <annotations_dir>/annotation_<task>_<annotator>.csv (written after
every item, so nothing is lost). Binds to 127.0.0.1 only: reach it through a tunnel, e.g.
    VS Code: the port is forwarded automatically (Ports panel) -> open http://localhost:<port>
    ssh:     ssh -L 8765:localhost:8765 <server>                 -> open http://localhost:8765

Usage (from any folder):
    python code/scripts/serve_annotation_tool.py              # port 8765
    python code/scripts/serve_annotation_tool.py --port 9000
Stop with Ctrl+C.
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from utils import constants as cons
from utils import ios
from utils import logs

NAME = re.compile(r'^[A-Za-z0-9_-]{1,64}$')   # allowed task / annotator ids (also prevents path traversal)
MAX_BODY_BYTES = 20 * 1024 * 1024


class AnnotationHandler(BaseHTTPRequestHandler):
    """GET /               -> the tool
       GET /tasks          -> tasks with an items file (JSON list)
       GET /items/<task>   -> annotation_<task>_items.json
       POST /save?task=&annotator=  (body: CSV) -> annotation_<task>_<annotator>.csv"""

    def __init__(self, *args, annotations_dir: Path, tool_file: Path, **kwargs):
        self.annotations_dir, self.tool_file = annotations_dir, tool_file
        super().__init__(*args, **kwargs)

    # -- GET ---------------------------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            return self._send(200, self.tool_file.read_bytes(), 'text/html; charset=utf-8')
        if path == '/tasks':
            tasks = [t for t in cons.ANNOTATION_TASKS if self._items_file(t).exists()]
            return self._send(200, ('[' + ','.join(f'"{t}"' for t in tasks) + ']').encode(), 'application/json')
        if path.startswith('/items/'):
            task = path.removeprefix('/items/')
            if task in cons.ANNOTATION_TASKS and self._items_file(task).exists():
                return self._send(200, self._items_file(task).read_bytes(), 'application/json; charset=utf-8')
        if path == '/annotations':      # the annotator's saved CSV (to resume in another browser)
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            task, annotator = query.get('task', ''), query.get('annotator', '')
            if task in cons.ANNOTATION_TASKS and NAME.match(annotator):
                target = self.annotations_dir / f"{cons.PREFIX_ANNOTATION}_{task}_{annotator}.csv"
                body = target.read_bytes() if target.exists() else b''
                return self._send(200, body, 'text/csv; charset=utf-8')
            return self._send(400, b'bad request', 'text/plain')
        return self._send(404, b'not found', 'text/plain')

    # -- POST --------------------------------------------------------------------------------
    def do_POST(self) -> None:
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        task, annotator = query.get('task', ''), query.get('annotator', '')
        length = int(self.headers.get('Content-Length', 0))
        if url.path != '/save' or task not in cons.ANNOTATION_TASKS or not NAME.match(annotator) \
                or not 0 < length <= MAX_BODY_BYTES:
            return self._send(400, b'bad request', 'text/plain')
        target = self.annotations_dir / f"{cons.PREFIX_ANNOTATION}_{task}_{annotator}.csv"
        temporary = target.with_suffix('.csv.tmp')
        temporary.write_bytes(self.rfile.read(length))
        temporary.replace(target)            # atomic: a crash never leaves a half-written file
        return self._send(200, f'saved {target.name}'.encode(), 'text/plain')

    # -- helpers -----------------------------------------------------------------------------
    def _items_file(self, task: str) -> Path:
        return self.annotations_dir / f"{cons.PREFIX_ANNOTATION}_{task}_items.json"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:   # route request logs through the pipeline logger
        ios.printf(f"{self.address_string()} {fmt % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='Interface to bind (keep 127.0.0.1 and use a tunnel)')
    parser.add_argument('--annotations_dir', default=cons.DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument('--tool_file', default=cons.ANNOTATION_TOOL_FILE)
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs.setup_logging('annotation_server', args, params=f"port{args.port}")
    annotations_dir = Path(args.annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(AnnotationHandler, annotations_dir=annotations_dir, tool_file=Path(args.tool_file))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    ios.printf(f"Serving the annotation tool on http://{args.host}:{args.port}  (annotations -> {annotations_dir})")
    ios.printf(f"Open http://localhost:{args.port} in your local browser (VS Code forwards the port; or ssh -L).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ios.printf("Stopped.")


if __name__ == '__main__':
    main()
