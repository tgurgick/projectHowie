"""Autodraft bridge: a Playwright process owns the ESPN (or Sleeper) draft
tab, streams the room into the cockpit, and lets the engine pick.

Loop (about once a second):
  1. read the room's pick history (text), diff against what we've seen,
     `service.sync_picks` the new ones — the cockpit board updates live;
  2. if the room says you are on the clock: ask the engine
     (`pick_payload`, deterministic, ~1 s), type the name into the room's
     player search, click that row's DRAFT button, then verify on the next
     read. Before your pick comes up, the top two candidates are placed in
     the room's queue so its own timer fallback is Howie's choice;
  3. write every event to data/autodraft.jsonl (the cockpit's Howie panel
     and `/api/autodraft/events` read it; Claude analyzes from there).

Browser: a persistent Chromium profile under data/browser_profile. Run
`howie autodraft signin` once: it opens that browser on espn.com and the
USER signs in by hand; the session cookie persists in the profile. No
credentials pass through this code.

Gate: clicks happen only with --autopilot, and only when the page title
says Mock/Practice unless --real is also passed.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import Settings

PICK_RE = re.compile(r"^(?P<name>.+?) / (?P<team>[A-Z]{2,3}) (?P<pos>QB|RB|WR|TE|K|D/ST)$")
ROUND_RE = re.compile(r"^R(?P<round>\d+), P(?P<pick>\d+) - (?P<owner>.+)$")
POLL_SECONDS = 1.0


def profile_dir(settings: Settings) -> Path:
    return settings.data_dir / "browser_profile"


def log_path(settings: Settings) -> Path:
    return settings.data_dir / "autodraft.jsonl"


def log_event(settings: Settings, kind: str, **fields) -> dict:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "kind": kind, **fields}
    with open(log_path(settings), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def recent_events(settings: Settings, n: int = 40) -> List[dict]:
    p = log_path(settings)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------- room parsing

def parse_picks(text: str) -> List[dict]:
    """ESPN's pick panel prints 'Name / TEAM POS' then 'Rn, Pm - Owner'."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    picks = []
    for i, ln in enumerate(lines[:-1]):
        m = PICK_RE.match(ln)
        r = ROUND_RE.match(lines[i + 1]) if m else None
        if m and r:
            picks.append({"name": m["name"], "team": m["team"], "pos": m["pos"],
                          "round": int(r["round"]), "pick": int(r["pick"]), "owner": r["owner"]})
    picks.sort(key=lambda p: (p["round"], p["pick"]))   # ESPN numbers picks within the round
    return picks


def on_clock(text: str) -> bool:
    return "You are on the clock" in text or "You're on the clock!" in text


# ---------------------------------------------------------------- the bridge

class AutoDrafter:
    def __init__(self, settings: Settings, url: str, autopilot: bool = False, real: bool = False,
                 headless: bool = False):
        self.settings = settings
        self.url = url
        self.autopilot = autopilot
        self.real = real
        self.headless = headless
        self.seen: set = set()
        self.queued: set = set()

    # -- browser
    def _launch(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        prof = profile_dir(self.settings)
        prof.mkdir(parents=True, exist_ok=True)
        self.ctx = self._pw.chromium.launch_persistent_context(
            str(prof), headless=self.headless, viewport={"width": 1400, "height": 1000})
        self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
        self.page.goto(self.url, wait_until="domcontentloaded")

    def close(self) -> None:
        try:
            self.ctx.close()
        finally:
            self._pw.stop()

    # -- reads
    def room_text(self) -> str:
        return self.page.inner_text("body")

    def picks_text(self) -> str:
        """The pick-history panel; the whole body if no panel is recognized."""
        for sel in ("[class*='pick-list']", "[class*='picks']", "aside"):
            try:
                loc = self.page.locator(sel).first
                if loc.count():
                    txt = loc.inner_text(timeout=500)
                    if ", P" in txt:
                        return txt
            except Exception:
                continue
        return self.room_text()

    # -- actions (player search box, then the row's button — never coordinates)
    def _search(self, name: str) -> None:
        box = self.page.get_by_placeholder("Player Name").first
        box.fill("")
        box.fill(name)
        self.page.wait_for_timeout(350)

    def _row_button(self, name: str, label: str):
        row = self.page.locator("tr", has_text=name).first
        if not row.count():
            row = self.page.locator("[class*='row']", has_text=name).first
        return row.get_by_role("button", name=re.compile(label, re.I)).first

    def click_draft(self, name: str) -> bool:
        self._search(name)
        btn = self._row_button(name, "DRAFT")
        if not btn.count():
            return False
        btn.click(timeout=2000)
        self.page.wait_for_timeout(400)
        confirm = self.page.get_by_role("button", name=re.compile("^(draft|confirm|yes)$", re.I))
        if confirm.count():
            try:
                confirm.first.click(timeout=1000)
            except Exception:
                pass
        return True

    def queue(self, name: str) -> bool:
        if name in self.queued:
            return True
        self._search(name)
        btn = self._row_button(name, "QUEUE")
        if not btn.count():
            return False
        btn.click(timeout=2000)
        self.queued.add(name)
        return True

    # -- the loop
    def run(self, max_minutes: float = 180.0) -> None:
        from . import service
        from .state import DraftState, snake_team_for_pick

        self._launch()
        title = self.page.title()
        log_event(self.settings, "start", url=self.url, title=title, autopilot=self.autopilot)
        if self.autopilot and not self.real and not re.search(r"mock|practice", title, re.I):
            log_event(self.settings, "refused", reason="page is not a mock/practice room (pass --real to override)")
            self.autopilot = False
        league = self.settings.league
        total = league.num_teams * league.roster_size
        t_end = time.time() + max_minutes * 60
        last_clock_pick = None
        while time.time() < t_end:
            try:
                picks = parse_picks(self.picks_text())
                new = [p for p in picks if (p["round"], p["pick"]) not in self.seen]
                if new:
                    r = service.sync_picks(self.settings, [p["name"] for p in new], source="autodraft")
                    for p in new:
                        self.seen.add((p["round"], p["pick"]))
                    log_event(self.settings, "sync",
                              picks=[f"{p['round']}.{p['pick']} {p['name']} ({p['owner']})" for p in new],
                              unresolved=r["unresolved"], next_pick=r["next_pick"])
                state = DraftState.load(self.settings)
                if state.next_pick_no() > total:
                    log_event(self.settings, "complete", picks=len(state.events))
                    break
                body = self.room_text()
                if on_clock(body) and state.next_pick_no() != last_clock_pick:
                    last_clock_pick = state.next_pick_no()
                    pk = service.pick_payload(self.settings, state, sims=0, top_n=3)
                    best = pk["rows"][0] if pk["rows"] else None
                    log_event(self.settings, "on_clock", pick=pk["current_pick"],
                              best=[{"name": r["name"], "pos": r["pos"], "delta": r["delta"],
                                     "avail_next": r["avail_next"]} for r in pk["rows"][:3]])
                    if best and self.autopilot:
                        ok = self.click_draft(best["name"])
                        if not ok and len(pk["rows"]) > 1:
                            best = pk["rows"][1]
                            ok = self.click_draft(best["name"])
                        log_event(self.settings, "draft_click", name=best["name"], ok=ok)
                elif not on_clock(body) and self.autopilot:
                    # pre-queue Howie's top two when our pick is within three
                    nxt = state.next_pick_no()
                    mine = [k for k in range(nxt, min(nxt + 4, total + 1))
                            if snake_team_for_pick(league, k) == league.draft_position]
                    if mine:
                        pk = service.pick_payload(self.settings, state, sims=0, top_n=2)
                        for r in pk["rows"][:2]:
                            if r["name"] not in self.queued and self.queue(r["name"]):
                                log_event(self.settings, "queued", name=r["name"], for_pick=mine[0])
            except Exception as e:  # keep the loop alive; the log shows what broke
                log_event(self.settings, "error", error=f"{e.__class__.__name__}: {e}"[:300])
                time.sleep(2)
            time.sleep(POLL_SECONDS)
        self.close()


def signin(settings: Settings, url: str = "https://www.espn.com/") -> None:
    """Open the bridge's own browser profile so the user can sign in by hand;
    the window stays open until they close it."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        prof = profile_dir(settings)
        prof.mkdir(parents=True, exist_ok=True)
        ctx = pw.chromium.launch_persistent_context(str(prof), headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url)
        print("Sign in to ESPN in that window, then close it. The session stays in", prof)
        try:
            page.wait_for_event("close", timeout=0)
        except Exception:
            pass
        ctx.close()
