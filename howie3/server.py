"""The draft-night cockpit server: stdlib HTTP, localhost only.

Serves the single-page UI and the JSON API over the service layer. All state
mutations go through one lock and the draft event log. After every state
change a background thread refines the deterministic ranking with Monte
Carlo; the UI polls and picks it up when ready.
"""

import hashlib
import json
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import service
from .config import Settings
from .state import DraftState, DraftStateError

UI_DIR = Path(__file__).parent / "ui"
UI_PATH = UI_DIR / "index.html"
# Static assets the server will hand out — an allowlist, not a directory walk
STATIC = {
    "/ui/app.js": "application/javascript; charset=utf-8",
    "/ui/lib.js": "application/javascript; charset=utf-8",
    "/ui/style.css": "text/css; charset=utf-8",
}
CSP = ("default-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; "
       "script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'")
MAX_BODY = 1_000_000  # bytes; the largest legitimate body is a pasted mock draft

_lock = threading.Lock()
_mc_cache: dict = {"gen": "", "data": None, "running": "", "error": None}
_det_cache: dict = {"gen": "", "pick": None}  # deterministic payload per generation
MC_SIMS = 150


def _generation(settings: Settings, state: DraftState) -> str:
    """Cache key for a recommendation: EVERYTHING that changes the ranking.

    Draft identity (created + seed) and length, the active strategy rules
    (targets / waits / bans re-rank the board), and the league config file
    (slot, scoring, anchor). Notes are deliberately excluded — free text
    never touches the engine, and invalidating the Monte Carlo on every
    keystroke would keep it perpetually "running"."""
    rules = [r.text.strip().upper() for r in state.rules if r.on]
    cfg = settings.data_dir / "league_config.json"
    cfg_sig = hashlib.sha1(cfg.read_bytes()).hexdigest()[:8] if cfg.exists() else "default"
    rule_sig = hashlib.sha1(json.dumps(sorted(rules)).encode()).hexdigest()[:8]
    # data identity: the db file (refresh, research import, roster status) and
    # the mock-lab store (its availability rates feed p_available)
    def sig(path: Path) -> str:
        try:
            st = path.stat()
            return f"{st.st_mtime_ns}-{st.st_size}"
        except FileNotFoundError:
            return "none"
    data_sig = hashlib.sha1(
        (sig(settings.db_path) + sig(settings.data_dir / "mock_sims.json")).encode()).hexdigest()[:8]
    return f"{state.created}:{state.seed}:{len(state.events)}:{rule_sig}:{cfg_sig}:{data_sig}"


def _kick_mc(settings: Settings, gen: str) -> None:
    with _lock:
        if _mc_cache["running"] == gen or _mc_cache["gen"] == gen:
            return  # already computed or in flight for this generation
        _mc_cache["running"] = gen

    def run() -> None:
        try:
            state = DraftState.load(settings)
            if _generation(settings, state) != gen:
                return
            data = service.pick_payload(settings, state, sims=MC_SIMS, top_n=10)
            with _lock:
                _mc_cache["error"] = None
                if _generation(settings, DraftState.load(settings)) == gen:
                    _mc_cache["gen"] = gen
                    _mc_cache["data"] = data
        except Exception as e:
            traceback.print_exc()
            with _lock:
                _mc_cache["error"] = f"{e.__class__.__name__}: {e}"
        finally:
            with _lock:
                if _mc_cache["running"] == gen:
                    _mc_cache["running"] = ""

    threading.Thread(target=run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    settings: Settings = None  # type: ignore[assignment]  # injected by serve() before any request is handled
    # Per-process session token: every mutating request must carry it in
    # X-Howie-Token. The page receives it in a <meta> tag, so a hostile
    # site that can reach 127.0.0.1 cannot drive the draft or spend API
    # budget from the user's browser (CSRF) — localhost is not an auth boundary.
    token: str = ""

    # ------------------------------------------------ plumbing

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page navigated away mid-response; nothing to report

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise RequestTooLarge(length)
        if not length:
            return {}
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}
        return body if isinstance(body, dict) else {}

    def _authorized(self) -> bool:
        return bool(self.token) and secrets.compare_digest(
            self.headers.get("X-Howie-Token", ""), self.token)

    # ------------------------------------------------ routes

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        s = self.settings
        try:
            if url.path in ("/", "/index.html"):
                body = UI_PATH.read_text().replace("__HOWIE_TOKEN__", self.token).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", CSP)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            elif url.path in STATIC:
                body = (UI_DIR / url.path.rsplit("/", 1)[1]).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", STATIC[url.path])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/state":
                self._json(service.state_payload(s, DraftState.load(s)))
            elif url.path == "/api/pick":
                state = DraftState.load(s)
                gen = _generation(s, state)
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
                    payload["mc_status"] = ("ready" if _mc_cache["gen"] == gen
                                            else "error" if _mc_cache["error"] else
                                            "running" if _mc_cache["running"] == gen else "pending")
                    payload["mc_error"] = _mc_cache["error"]
                if payload["mc_status"] in ("pending", "error"):
                    _kick_mc(s, gen)  # a failed or never-started worker is retried on demand
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
            elif url.path == "/api/data/games":
                self._json(service.games_distribution(
                    s, q.get("pos", "RB"), q.get("stat", "pts"), q.get("tier", "starter")))
            elif url.path == "/api/data/sim":
                self._json(service.sim_payload(s, q.get("uid", "")))
            elif url.path == "/api/data/roster_sim":
                self._json(service.roster_sim_payload(s, DraftState.load(s)))
            elif url.path == "/api/data/query":
                self._json(service.query_payload(s, q.get("q", "")))
            elif url.path == "/api/data/build":
                self._json(service.build_query(
                    s, entity=q.get("entity", "player"), pos=q.get("pos", "ALL"),
                    season=q.get("season", "2025"), measure=q.get("measure", "total"),
                    stat=q.get("stat", "pts"), thr=float(q.get("thr", 100) or 100),
                    min_games=int(q.get("min_games", 1) or 1), order=q.get("order", "desc"),
                    limit=int(q.get("limit", 20) or 20)))
            elif url.path == "/api/sim/mock/status":
                from . import mocksim
                self._json(dict(mocksim.STATUS))
            elif url.path == "/api/sim/mock/results":
                from . import mocksim
                self._json(mocksim.aggregates(s))
            elif url.path == "/api/config":
                self._json(service.config_payload(s))
            elif url.path == "/api/autodraft/events":
                from .autodraft import recent_events
                self._json({"events": recent_events(s, int(q.get("n", 30)))})
            elif url.path == "/api/coach/status":
                from . import coach
                self._json({**coach.STATUS, "sessions": coach.load_sessions(s)["sessions"][-5:]})
            elif url.path == "/api/sequence":
                state = DraftState.load(s)
                with _lock:
                    mc = _mc_cache["data"] if _mc_cache["gen"] == _generation(s, state) else None
                now_uid = mc["rows"][0]["uid"] if mc and mc.get("rows") else None
                self._json(service.sequence_payload(s, state, now_uid=now_uid))
            elif url.path == "/api/lookahead":
                self._json(service.lookahead_payload(s, DraftState.load(s), int(q.get("n", 3))))
            elif url.path == "/api/plan":
                self._json(service.plan_payload(s, DraftState.load(s)))
            elif url.path == "/api/season_grid":
                self._json(service.season_grid_payload(s, DraftState.load(s)))
            elif url.path == "/api/team":
                self._json(service.team_payload(s, DraftState.load(s), q.get("team", "PHI")))
            elif url.path == "/api/risk":
                self._json(service.roster_risk(s, DraftState.load(s)))
            elif url.path == "/api/research/status":
                from . import insights
                self._json(insights.research_status(s))
            elif url.path == "/api/research/facts":
                from . import insights
                self._json(insights.facts_for(s, q.get("q", "")))
            else:
                self._error("not found", 404)
        except DraftStateError as e:
            self._error(f"draft log problem: {e}", 409)
        except ValueError as e:
            self._error(str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(f"{e.__class__.__name__}: {e}", 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if not self._authorized():
            self._error("missing or invalid session token", 403)
            return
        try:
            body = self._body()
        except RequestTooLarge as e:
            # drain (bounded) so the client sees the 413 instead of a broken pipe
            remaining = min(int(str(e)), 16 * MAX_BODY)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            self._error(f"request body too large ({e} bytes, limit {MAX_BODY})", 413)
            return
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
            elif url.path == "/api/sim/mock/run":
                from . import mocksim
                started = mocksim.run_in_background(
                    s, int(body.get("n", 25)), str(body.get("policy", "adp")))
                self._json({"started": started, "status": dict(mocksim.STATUS)})
                return
            elif url.path == "/api/config":
                self._json(service.update_config(s, body))
                return
            elif url.path == "/api/ask":
                self._json(service.ask_howie(s, str(body.get("question", ""))))
                return
            elif url.path == "/api/lab/insights":
                from . import insights
                self._json(insights.generate_insights(s, str(body.get("kind", "mock")), body))
                return
            elif url.path == "/api/research/run":
                from . import insights
                if body.get("team"):
                    self._json(insights.research_team(s, str(body["team"])))
                else:
                    self._json(insights.research_player(s, str(body.get("player", "")), body.get("team_hint")))
                return
            elif url.path == "/api/coach/run":
                from . import coach
                started = coach.run_in_background(
                    s, iterations=int(body.get("iterations", 3)), n_drafts=int(body.get("drafts", 12)),
                    reps=int(body.get("reps", 6)), seed=int(body.get("seed", 101)))
                self._json({"started": started, "status": dict(coach.STATUS)})
                return
            elif url.path == "/api/sim/mock/import":
                from . import mocksim
                self._json(mocksim.import_external(s, str(body.get("text", "")),
                                                   str(body.get("source", "external"))))
                return
            else:
                self._error("not found", 404)
                return
            _kick_mc(s, _generation(s, DraftState.load(s)))
            self._json(result)
        except DraftStateError as e:
            self._error(f"draft log problem: {e}", 409)
        except ValueError as e:
            self._error(str(e))
        except Exception as e:
            traceback.print_exc()
            self._error(f"{e.__class__.__name__}: {e}", 500)


class RequestTooLarge(Exception):
    pass


def _warm_imports() -> None:
    """Import the simulation stack on the main thread BEFORE any worker runs:
    a background thread importing lazily while the first request imports the
    same circular pair (roster <-> distributions) can observe a partially
    initialized module."""
    from . import mocksim as _m, service as _service, status as _st  # noqa: F401
    from .value import distributions as _d, policy as _p, roster as _r, simulate as _s  # noqa: F401


def serve(settings: Optional[Settings] = None, port: int = 8787) -> ThreadingHTTPServer:
    settings = settings or Settings()
    _warm_imports()
    Handler.settings = settings
    Handler.token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    _kick_mc(settings, _generation(settings, DraftState.load(settings)))
    return server


def main(port: int = 8787) -> None:
    server = serve(port=port)
    print(f"Howie cockpit: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
