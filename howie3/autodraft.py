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


ROSTER_LINE_RE = re.compile(r"^(QB|RB|WR|TE|FLEX|D/ST|K|BE|BN|IR)$")
ROSTER_NAME_RE = re.compile(r"^(?P<name>[A-Z][\w.'-]*\.? [\w.'/-]+(?: [\w.'/-]+)*?)(?: \((?P<pos>QB|RB|WR|TE|K|D/ST)\))?$")


def parse_roster(text: str) -> List[dict]:
    """ESPN's roster panel: 'POS' line, then 'D. Prescott' (bench rows add
    '(QB)'), then the bye. Returns [{slot, name, pos}] — pos is the slot
    position for starters, the bracketed one for bench rows."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for i, ln in enumerate(lines[:-1]):
        if ROSTER_LINE_RE.match(ln):
            nm = lines[i + 1]
            if nm in ("Empty", "-"):
                continue
            m = ROSTER_NAME_RE.match(nm)
            if not m:
                continue
            pos = m["pos"] or (ln if ln not in ("FLEX", "BE", "BN", "IR") else None)
            out.append({"slot": ln, "name": m["name"], "pos": "DST" if pos == "D/ST" else pos})
    return out


def on_clock(text: str) -> bool:
    return "You are on the clock" in text or "You're on the clock!" in text


SLOT_RE = re.compile(r"Round 1, Pick (\d+)")   # 'Your first pick: …' before the start, 'You're on the clock in: N Picks / Round 1, Pick N' after
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


# ---------------------------------------------------------------- reasoning

def reasoning(pick: dict, positional: dict, sims: int) -> dict:
    """What the Howie panel shows while the engine thinks: the top
    candidates with value / outcome band / availability at the next pick,
    the cost of waiting at each position, and the rules that fired."""
    rows = pick["rows"][:4]
    best = rows[0] if rows else None
    alt = rows[1] if len(rows) > 1 else None
    pos_cost = {r["pos"]: {"cost": r["cost"], "avail_next": r["avail_next"], "tier_drop": r["tier_drop"],
                           "player": r["player"]} for r in positional.get("rows", [])}
    why = []
    if best:
        if alt:
            why.append(f"{best['name']} over {alt['name']}: {abs(alt['delta'])} pts of final-lineup value"
                       + (f" ({alt['pos']} {int(alt['avail_next'] * 100)}% there next pick vs {best['pos']} {int(best['avail_next'] * 100)}%)" if alt["pos"] != best["pos"] else ""))
        pc = pos_cost.get(best["pos"])
        if pc:
            why.append(f"waiting on {best['pos']} costs {pc['cost']} (next tier −{pc['tier_drop']}, {int(pc['avail_next'] * 100)}% chance the best is still there)")
        if best.get("rules"):
            why.append("rules: " + ", ".join(f["text"] for f in best["rules"]))
        if best.get("status"):
            why.append(f"status: {best['status']['text']}")
    return {
        "sims": sims,
        "candidates": [{"name": r["name"], "pos": r["pos"], "value": r["value"], "delta": r["delta"],
                        "p10": r.get("p10"), "p90": r.get("p90"), "avail_next": r["avail_next"],
                        "avail_src": r.get("avail_src")} for r in rows],
        "positional": {pos: {"cost": v["cost"], "avail_next": v["avail_next"]} for pos, v in pos_cost.items()},
        "next_pick": pick["next_pick"], "why": why,
    }


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
        self.player_ids: Dict[str, str] = {}   # name -> ESPN data-player-id (learned from QUEUE buttons)

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

    def roster_panel(self) -> List[dict]:
        """The user's roster as the room shows it (ground truth for 'mine')."""
        try:
            for sel in ("[class*='roster']", "[class*='Roster']", "aside"):
                loc = self.page.locator(sel).first
                if loc.count():
                    txt = loc.inner_text(timeout=400)
                    if "Roster Limits" in txt or "\nQB\n" in txt:
                        return parse_roster(txt)
            return parse_roster(self.room_text())
        except Exception:
            return []

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
    @staticmethod
    def room_names(name: str) -> List[str]:
        """How the room may spell this player: defenses are 'Rams D/ST' /
        'Packers D/ST' in ESPN, 'LA D/ST' / 'GB D/ST' in Howie."""
        m = re.match(r"^([A-Z]{2,3}) D/ST$", name)
        if m:
            from .graph import TEAM_NAMES
            full = TEAM_NAMES.get(m.group(1), "")
            nick = full.split()[-1] if full else ""
            return [f"{nick} D/ST", f"{full} D/ST", nick, name] if nick else [name]
        return [name]

    def _search(self, name: str):
        """ESPN's player search is an autocomplete: type, then click the
        suggestion (button.player--search--match); the table then shows that
        player's row. Matched by text (names carry apostrophes and periods
        that break CSS selectors). Returns the suggestion's player id, or
        None when no suggestion appeared (then NOTHING may be clicked)."""
        box = self.page.get_by_placeholder("Player Name").first
        matches = self.page.locator("button.player--search--match")
        found = False
        for spelling in self.room_names(name):
            box.fill("", timeout=1500)
            box.fill(spelling, timeout=1500)
            try:
                matches.first.wait_for(state="visible", timeout=1500)
                found = True
                break
            except Exception:
                continue
        if not found:
            self._suggestion_ok = False
            return self.player_ids.get(name)
        self._suggestion_ok = True
        target = None
        for i in range(min(matches.count(), 8)):
            m = matches.nth(i)
            try:
                if (m.get_attribute("data-player-search-playername", timeout=200) or "").strip() == name:
                    target = m
                    break
            except Exception:
                continue
        if target is None:
            target = matches.filter(has_text=name.split()[-1]).first
        pid = target.get_attribute("data-player-search-playerid", timeout=500)
        try:
            target.click(timeout=2500)
        except Exception:
            try:
                target.click(force=True, timeout=1500)
            except Exception:
                pass
        if pid:
            self.player_ids[name] = pid
        self.page.wait_for_timeout(250)
        return pid

    def _row_button(self, pid, label: str, name: str = ""):
        """The `label` button on THIS player's row: start at his name in the
        table and walk up to the nearest ancestor that holds exactly one such
        button — the row, never the table (which holds everyone's)."""
        pattern = re.compile(label, re.I)
        # the row buttons carry ESPN's player id (the same id the search
        # suggestion carries): the most direct match when we have it
        if pid:
            byid = self.page.locator(f"button[data-player-id='{pid}']").filter(has_text=pattern)
            for i in range(min(byid.count(), 4)):
                b = byid.nth(i)
                try:
                    if b.is_visible():
                        return b
                except Exception:
                    continue
        # after a SUCCESSFUL suggestion click the table holds one player: a
        # lone visible button with this label is his — but only if the row
        # around it names him (never click an unfiltered list's top row)
        if getattr(self, "_suggestion_ok", False):
            lone = self.page.locator("button").filter(has_text=pattern)
            visible = []
            for i in range(min(lone.count(), 6)):
                try:
                    if lone.nth(i).is_visible():
                        visible.append(lone.nth(i))
                except Exception:
                    continue
            if len(visible) == 1:
                tokens = [t for sp in self.room_names(name) for t in sp.replace("D/ST", "").split() if len(t) > 2] or [name]
                try:
                    row_txt = visible[0].locator("xpath=ancestor::*[4]").inner_text(timeout=300)
                except Exception:
                    row_txt = ""
                if any(t in row_txt for t in tokens):
                    return visible[0]
        names = self.page.locator(".playerinfo__playername", has_text=name) if name else self.page.locator(".playerinfo__playername")
        for i in range(min(names.count(), 6)):
            el = names.nth(i)
            try:
                if not el.is_visible() or (name and name not in el.inner_text(timeout=200)):
                    continue  # the cell may carry an injury tag: "Christian McCaffrey Q"
                for depth in range(1, 8):
                    anc = el.locator(f"xpath=ancestor::*[{depth}]")
                    if not anc.count():
                        break
                    btns = anc.first.locator("button").filter(has_text=pattern)
                    n = btns.count()
                    if n == 1 and btns.first.is_visible():
                        return btns.first
                    if n > 1:
                        break  # climbed into a container holding other rows
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

    def _debug_buttons(self, pid) -> str:
        try:
            btns = self.page.locator("button").filter(has_text=re.compile("draft|queue", re.I))
            return " | ".join(btns.nth(i).inner_text(timeout=200).strip()[:20] for i in range(min(btns.count(), 6))) or "(no draft/queue buttons)"
        except Exception as e:
            return f"(debug failed: {e.__class__.__name__})"

    def click_draft(self, name: str) -> bool:
        pid = self._search(name)
        btn = self._row_button(pid, "draft", name)
        if btn is None or not self._click(btn):
            log_event(self.settings, "draft_button_missing", name=name, pid=pid, buttons=self._debug_buttons(pid))
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
        pid = self._search(name)
        btn = self._row_button(pid, "queue", name)
        if btn is None or not self._click(btn):
            log_event(self.settings, "queue_button_missing", name=name, pid=pid, buttons=self._debug_buttons(pid))
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
        last_foresight = (0.0, -1)   # (time, next_pick) the last "next 3 picks" forecast was logged for
        while time.time() < t_end:
            try:
                if not configured:
                    # the room states the league shape and your seat; take it
                    # and wipe the board before the first pick lands
                    cfg = room_config(self.room_text(), self.page.title())
                    st0 = DraftState.load(self.settings)
                    in_progress = st0.events and st0.next_pick_no() <= self.settings.league.num_teams * self.settings.league.roster_size
                    if cfg.get("draft_position") and not in_progress:
                        # a fresh room: take its shape, archive whatever finished draft is on the
                        # board, wipe. A draft in progress is never wiped (the post-draft page and
                        # a rejoin both show the same banner).
                        service.update_config(self.settings, cfg)
                        service.reset_draft(self.settings, "live")
                        log_event(self.settings, "configured", **cfg)
                        configured = True
                    elif "/football/draft" in self.page.url and DraftState.load(self.settings).events:
                        # rejoining a draft in progress: keep the log and the league as configured
                        log_event(self.settings, "rejoined", picks=len(DraftState.load(self.settings).events))
                        configured = True
                    else:
                        time.sleep(POLL_SECONDS)
                        continue
                league = self.settings.league
                total = league.num_teams * league.roster_size
                picks = parse_picks(self.picks_text())
                new = [p for p in picks if (p["round"], p["pick"]) not in self.seen]
                if new:
                    nums = [(p["round"] - 1) * league.num_teams + p["pick"] for p in new]
                    r = service.sync_picks(self.settings, [p["name"] for p in new], source="autodraft",
                                           pick_numbers=nums)
                    for p in new:
                        self.seen.add((p["round"], p["pick"]))
                    log_event(self.settings, "sync",
                              picks=[f"{p['round']}.{p['pick']} {p['name']} ({p['owner']})" for p in new],
                              unresolved=r["unresolved"], gaps=r.get("gaps", []), next_pick=r["next_pick"])
                # every ~8 s: the roster panel is the truth for which picks are ours
                self._ticks = getattr(self, "_ticks", 0) + 1
                if self._ticks % 8 == 0:
                    roster = self.roster_panel()
                    if roster:
                        fix = service.reconcile_roster(self.settings, roster)
                        if fix["changed"]:
                            log_event(self.settings, "roster_fix", **fix)
                state = DraftState.load(self.settings)
                nxt = state.next_pick_no()
                if nxt > total:
                    log_event(self.settings, "complete", picks=len(state.events))
                    break
                my_next = next((k for k in range(nxt, min(nxt + 3, total + 1))
                                if snake_team_for_pick(league, k) == league.draft_position), None)
                clock = self.is_on_clock()
                # foresight: the next three of our picks as the board stands (both modes);
                # refreshed as picks land, at most every 12 s, never while on the clock
                if not clock and nxt != last_foresight[1] and time.time() - last_foresight[0] > 12:
                    fs = service.lookahead_payload(self.settings, state, n=3)
                    last_foresight = (time.time(), nxt)
                    log_event(self.settings, "foresight", next_pick=nxt, picks=fs["picks"])
                if clock and nxt != last_clock_pick:
                    last_clock_pick = nxt
                    taken = state.taken_uids()
                    rows = [r for r in (cached["rows"] or []) if r["uid"] not in taken] if cached["pick"] == nxt else []
                    why = cached.get("why") if (rows and cached["pick"] == nxt) else None
                    if not rows:  # nothing pre-computed for this pick: compute now (deterministic, fast)
                        pk = service.pick_payload(self.settings, state, sims=0, top_n=4)
                        rows = pk["rows"]
                        why = reasoning(pk, service.positions_payload(self.settings, state), 0)
                    log_event(self.settings, "on_clock", pick=nxt,
                              best=[{"name": r["name"], "pos": r["pos"], "delta": r["delta"]} for r in rows[:3]],
                              why=why)
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
                    # one or two picks away: run the Monte Carlo on the top candidates
                    # (there is time now, not on the clock), stream the reasoning, and
                    # put the top two in the room's queue so the timer fallback is his
                    t0 = time.time()
                    pk = service.pick_payload(self.settings, state, sims=100, top_n=5)
                    rows = pk["rows"]
                    why = reasoning(pk, service.positions_payload(self.settings, state), 100)
                    cached = {"pick": my_next, "rows": rows, "why": why}
                    log_event(self.settings, "thinking", for_pick=my_next, seconds=round(time.time() - t0, 1), **why)
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
