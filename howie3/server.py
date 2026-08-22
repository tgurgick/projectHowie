"""The draft-night cockpit server: stdlib HTTP, localhost only.

Serves the single-page UI and the JSON API over the service layer. All state
mutations go through one lock and the draft event log. After every state
change a background thread refines the deterministic ranking with Monte
Carlo; the UI polls and picks it up when ready.
"""

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import service
from .config import Settings
from .state import DraftState

UI_PATH = Path(__file__).parent / "ui" / "index.html"

_lock = threading.Lock()
_mc_cache: dict = {"gen": "", "data": None, "running": ""}
_det_cache: dict = {"gen": "", "pick": None}  # deterministic payload per generation
MC_SIMS = 150


def _generation(state: DraftState) -> str:
    # identity + length: a reset draft with the same event count must never
    # reuse a cached ranking that references the previous draft's pool
    return f"{state.created}:{state.seed}:{len(state.events)}"


def _kick_mc(settings: Settings, gen: int) -> None:
    with _lock:
        if _mc_cache["running"] == gen or _mc_cache["gen"] == gen:
            return  # already computed or in flight for this generation
        _mc_cache["running"] = gen

    def run() -> None:
        try:
            state = DraftState.load(settings)
            if _generation(state) != gen:
                return
            data = service.pick_payload(settings, state, sims=MC_SIMS, top_n=10)
            with _lock:
                if _generation(DraftState.load(settings)) == gen:
                    _mc_cache["gen"] = gen
                    _mc_cache["data"] = data
        except Exception:
            traceback.print_exc()
        finally:
            with _lock:
                if _mc_cache["running"] == gen:
                    _mc_cache["running"] = ""

    threading.Thread(target=run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    settings: Settings = None  # injected by serve()

    # ------------------------------------------------ plumbing

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------ routes

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        s = self.settings
        try:
            if url.path in ("/", "/index.html"):
                body = UI_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/state":
                self._json(service.state_payload(s, DraftState.load(s)))
            elif url.path == "/api/pick":
                state = DraftState.load(s)
                gen = _generation(state)
                top = int(q.get("top", 10))
                with _lock:
                    cached = _det_cache["pick"] if _det_cache["gen"] == gen else None
                if cached is None or len(cached["rows"]) < top:
                    cached = service.pick_payload(s, state, sims=0, top_n=max(top, 25))
                    with _lock:
                        _det_cache["gen"] = gen
                        _det_cache["pick"] = cached
                payload = dict(cached)
                payload["rows"] = cached["rows"][:top] if top < len(cached["rows"]) else cached["rows"]
                with _lock:
                    if _mc_cache["gen"] == gen:
                        payload["mc"] = _mc_cache["data"]
                self._json(payload)
            elif url.path == "/api/positions":
                self._json(service.positions_payload(s, DraftState.load(s)))
            elif url.path == "/api/search":
                self._json(service.search_payload(s, q.get("q", ""), limit=8))
            elif url.path == "/api/card":
                self._json(service.card_payload(s, q.get("uid", "")))
            elif url.path == "/api/strategy":
                self._json(service.strategy_payload(DraftState.load(s)))
            elif url.path == "/api/anchors":
                self._json(service.anchors_payload(s, DraftState.load(s)))
            else:
                self._error("not found", 404)
        except ValueError as e:
            self._error(str(e))
        except Exception as e:
            traceback.print_exc()
            self._error(f"{e.__class__.__name__}: {e}", 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        body = self._body()
        s = self.settings
        try:
            # service mutations take the cross-process file lock themselves
            if url.path == "/api/mark":
                result = service.mark_pick(s, str(body.get("uid", "")),
                                           mine=bool(body.get("mine")), source="ui")
            elif url.path == "/api/undo":
                result = {"undone": service.undo_pick(s)}
            elif url.path == "/api/mock/start":
                result = service.start_mock(s)
            elif url.path == "/api/reset":
                result = service.reset_draft(s, str(body.get("mode", "live")))
            elif url.path == "/api/strategy":
                result = service.update_strategy(s, rules=body.get("rules"),
                                                 notes=body.get("notes"))
            else:
                self._error("not found", 404)
                return
            _kick_mc(s, _generation(DraftState.load(s)))
            self._json(result)
        except ValueError as e:
            self._error(str(e))
        except Exception as e:
            traceback.print_exc()
            self._error(f"{e.__class__.__name__}: {e}", 500)


def serve(settings: Optional[Settings] = None, port: int = 8787) -> ThreadingHTTPServer:
    settings = settings or Settings()
    Handler.settings = settings
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    _kick_mc(settings, _generation(DraftState.load(settings)))
    return server


def main(port: int = 8787) -> None:
    server = serve(port=port)
    print(f"Howie cockpit: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
