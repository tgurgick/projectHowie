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
from typing import Any, Dict, List

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


SLOT_RE = re.compile(r"Your first pick: Round 1, Pick (\d+)")
TEAMS_RE = re.compile(r"(\d+)-Team")
ROSTER_RE = re.compile(r"Roster Limits\s*\d+/(\d+) Players")


def room_config(text: str, title: str) -> dict:
    """League shape the room itself states: teams, your slot, scoring, roster
    size. Whatever it doesn't state is left alone."""
    out: dict = {}
    m = TEAMS_RE.search(title) or TEAMS_RE.search(text)
    if m:
        out["num_teams"] = int(m.group(1))
    m = SLOT_RE.search(text)
    if m:
        out["draft_position"] = int(m.group(1))
    m = ROSTER_RE.search(text)
    if m:
        out["roster_size"] = int(m.group(1))
    low = title.lower()
    if "half" in low:
        out["scoring_type"] = "half_ppr"
    elif "ppr" in low:
        out["scoring_type"] = "ppr"
    elif "standard" in low:
        out["scoring_type"] = "standard"
    return out


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
    def _follow_room(self) -> None:
        """ESPN opens the draft room in a new window from the lobby: follow
        whichever page is the room."""
        for pg in self.ctx.pages:
            if "/football/draft" in pg.url and pg is not self.page:
                self.page = pg
                log_event(self.settings, "room", url=pg.url, title=pg.title())
                break

    def room_text(self) -> str:
        self._follow_room()
        return self.page.inner_text("body")

    def is_on_clock(self) -> bool:
        """Cheap check: the clock banner, not the whole page."""
        self._follow_room()
        try:
            return self.page.get_by_text(re.compile(r"You(?:'re| are) on the clock!"), exact=False).first.is_visible(timeout=200)
        except Exception:
            return False

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
        box.fill(name, timeout=1500)
        self.page.wait_for_timeout(250)

    def _row_button(self, name: str, label: str):
        """After the search box filters the list to this player, his row is
        the only one with a visible button labelled `label`."""
        btns = self.page.locator("button").filter(has_text=re.compile(label, re.I))
        for i in range(min(btns.count(), 4)):
            b = btns.nth(i)
            try:
                if b.is_visible():
                    return b
            except Exception:
                continue
        return None

    def _click(self, btn) -> bool:
        try:
            btn.scroll_into_view_if_needed(timeout=1000)
        except Exception:
            pass
        try:
            btn.click(timeout=3000)
            return True
        except Exception:
            try:
                btn.click(force=True, timeout=2000)
                return True
            except Exception:
                return False

    def click_draft(self, name: str) -> bool:
        self._search(name)
        btn = self._row_button(name, "^draft$")
        if btn is None or not self._click(btn):
            return False
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
        btn = self._row_button(name, "^queue$")
        if btn is None or not self._click(btn):
            return False
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
        configured = False
        t_end = time.time() + max_minutes * 60
        last_clock_pick = None
        cached: Dict[str, Any] = {"pick": None, "rows": []}   # Howie's ranking computed one pick early
        while time.time() < t_end:
            try:
                if not configured:
                    # the room states the league shape and your seat; take it
                    # and wipe the board before the first pick lands
                    cfg = room_config(self.room_text(), self.page.title())
                    if cfg.get("draft_position"):
                        service.update_config(self.settings, cfg)
                        service.reset_draft(self.settings, "live")
                        log_event(self.settings, "configured", **cfg)
                        configured = True
                    else:
                        time.sleep(POLL_SECONDS)
                        continue
                league = self.settings.league
                total = league.num_teams * league.roster_size
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
                nxt = state.next_pick_no()
                if nxt > total:
                    log_event(self.settings, "complete", picks=len(state.events))
                    break
                my_next = next((k for k in range(nxt, min(nxt + 3, total + 1))
                                if snake_team_for_pick(league, k) == league.draft_position), None)
                clock = self.is_on_clock()
                if clock and nxt != last_clock_pick:
                    last_clock_pick = nxt
                    taken = state.taken_uids()
                    rows = [r for r in (cached["rows"] or []) if r["uid"] not in taken] if cached["pick"] == nxt else []
                    if not rows:  # nothing pre-computed for this pick: compute now
                        rows = service.pick_payload(self.settings, state, sims=0, top_n=3)["rows"]
                    log_event(self.settings, "on_clock", pick=nxt,
                              best=[{"name": r["name"], "pos": r["pos"], "delta": r["delta"]} for r in rows[:3]])
                    if rows and self.autopilot:
                        ok = self.click_draft(rows[0]["name"])
                        chosen = rows[0]
                        if not ok and len(rows) > 1:
                            chosen = rows[1]
                            ok = self.click_draft(chosen["name"])
                        log_event(self.settings, "draft_click", name=chosen["name"], ok=ok,
                                  fallback="queue" if (not ok and self.queued) else None)
                    # the pick after this one may also be ours (the turn): prepare it
                    cached = {"pick": None, "rows": []}
                elif not clock and my_next is not None and self.autopilot and cached["pick"] != my_next:
                    # one or two picks away: compute Howie's ranking now and put the
                    # top two in the room's queue so the timer fallback is his
                    rows = service.pick_payload(self.settings, state, sims=0, top_n=3)["rows"]
                    cached = {"pick": my_next, "rows": rows}
                    for r in rows[:2]:
                        if r["name"] not in self.queued and self.queue(r["name"]):
                            log_event(self.settings, "queued", name=r["name"], for_pick=my_next)
            except Exception as e:  # keep the loop alive; the log shows what broke
                log_event(self.settings, "error", error=f"{e.__class__.__name__}: {e}"[:300])
                time.sleep(2)
            time.sleep(POLL_SECONDS if my_next is None else 0.3)
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
