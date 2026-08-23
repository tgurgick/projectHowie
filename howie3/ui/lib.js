// Howie cockpit — pure helpers. Loaded before app.js in the browser (plain
// globals) and imported by tests/ui/lib.test.mjs under `node --test`.
// No DOM access here: everything is a function of its arguments.

/** HTML-escape any value (null/undefined -> ''). */
function esc(v) {
  return String(v == null ? '' : v).replace(/[&<>"']/g, c => (
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

/** Marks a string as already-safe HTML so `h` interpolates it verbatim. */
class Raw {
  constructor(s) { this.s = s; }
  toString() { return this.s; }
}
function raw(s) { return s instanceof Raw ? s : new Raw(String(s == null ? '' : s)); }

/**
 * Auto-escaping template tag: every interpolated value is escaped unless it
 * is a Raw (e.g. a nested h`` template) — arrays are joined with the same
 * rule, so `${rows.map(r => h`…`)}` works without join(). Returns a Raw, so
 * `el.innerHTML = h`…`` renders via toString(). This is the ONLY way the
 * cockpit builds HTML from data (player names, research facts, SQL results
 * and model output are all untrusted).
 */
function h(strings, ...vals) {
  let out = '';
  const one = v => (v instanceof Raw ? v.s : Array.isArray(v) ? v.map(one).join('') : esc(v));
  strings.forEach((s, i) => { out += s; if (i < vals.length) out += one(vals[i]); });
  return new Raw(out);
}

/** Newlines -> <br>, everything escaped. For terminal output. */
function textHtml(s) { return new Raw(esc(s).replace(/\n/g, '<br>')); }

const TEAM_CODES = new Set(['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC',
  'LA','LAC','LV','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SEA','SF','TB','TEN','WAS']);

/**
 * Classify a command-line entry.
 *   '/cmd arg…'         -> {kind:'cmd', cmd, arg, rest[]}
 *   '?question' / '…?'  -> {kind:'ask', question}
 *   a player name       -> {kind:'player', hit}  (exact match, the navigated
 *                          selection, a unique player hit, or a unique prefix)
 *   a team / room       -> {kind:'team', team:'GB'} (abbreviation, team name,
 *                          "GB WR room", the navigated selection, or a unique hit)
 *   anything else       -> {kind:'nomatch', suggestions[]}
 * `items` are the autocomplete hits for the current text; `selIdx` is the
 * arrow-key selection (-1 when the user hasn't navigated).
 */
function classifyInput(line, items, selIdx) {
  const text = (line || '').trim();
  if (!text) return {kind: 'empty'};
  if (text.startsWith('/')) {
    const [cmd, ...rest] = text.slice(1).split(/\s+/);
    return {kind: 'cmd', cmd: cmd.toLowerCase(), rest, arg: rest.join(' ').trim()};
  }
  if (text.startsWith('?') || /^(howie|hey howie)\b/i.test(text)) {
    return {kind: 'ask', question: text.replace(/^\?\s*|^(hey\s+)?howie[,:]?\s*/i, '').trim() || text};
  }
  const players = (items || []).filter(x => x && x.uid);
  const teams = (items || []).filter(x => x && !x.uid && (x.kind === 'team' || x.kind === 'unit') && x.team);
  if (selIdx != null && selIdx >= 0 && items && items[selIdx]) {
    const sel = items[selIdx];
    if (sel.uid) return {kind: 'player', hit: sel};
    if (sel.team) return {kind: 'team', team: sel.team, hit: sel};
  }
  const lower = text.toLowerCase();
  if (/^[a-z]{2,3}$/.test(lower) && (TEAM_CODES.has(lower.toUpperCase()) || teams.some(t => t.team.toLowerCase() === lower))) {
    return {kind: 'team', team: lower.toUpperCase()};   // abbreviation beats a player-name prefix
  }
  const teamExact = teams.find(t => t.name.toLowerCase() === lower);
  if (teamExact) return {kind: 'team', team: teamExact.team, hit: teamExact};
  const exact = players.find(p => p.name.toLowerCase() === lower);
  if (exact) return {kind: 'player', hit: exact};
  if (players.length === 1 && lower.length >= 3) return {kind: 'player', hit: players[0]};
  const prefix = players.filter(p => p.name.toLowerCase().startsWith(lower));
  if (prefix.length === 1 && lower.length >= 4) return {kind: 'player', hit: prefix[0]};
  if (!players.length && teams.length && lower.length >= 3) {
    const uniq = [...new Set(teams.map(t => t.team))];
    if (uniq.length === 1) return {kind: 'team', team: uniq[0], hit: teams[0]};
  }
  if (/\?$/.test(text) || text.split(/\s+/).length >= 5) return {kind: 'ask', question: text};
  return {kind: 'nomatch', suggestions: [...players.slice(0, 4).map(p => p.name), ...teams.slice(0, 2).map(t => t.name)]};
}

/** Which pick action a key combination means while a draft is in progress. */
function pickAction(ev, drafting) {
  if (ev.shiftKey) return 'mine';
  if (ev.altKey || ev.metaKey || ev.ctrlKey) return 'card';
  return drafting ? 'taken' : 'card';
}

/** Color class for an availability probability. */
function availClass(p) { return p > 0.55 ? 'acc' : p > 0.25 ? 'amber' : 'red'; }

/** Sign-formatted integer delta. */
function fmtDelta(d) { const n = Math.round(d); return (n > 0 ? '+' : '') + n; }

/** One-line "why" for a candidate on the clock. */
function reasonLine(card, risk) {
  const bits = [];
  if (card.mv_vs_wait != null) bits.push(`${fmtDelta(card.mv_vs_wait)} vs waiting at ${card.pos}`);
  if (card.avail_next != null) bits.push(`${Math.round(card.avail_next * 100)}% there at ${card.next_pick}`);
  const r = risk && risk.positions && risk.positions[card.pos];
  if (r && r.level !== 'ok') bits.push(`${card.pos} ${r.level === 'danger' ? 'THIN' : 'DUE'}`);
  return bits.join(' · ');
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {esc, Raw, raw, h, textHtml, classifyInput, pickAction, availClass, fmtDelta, reasonLine, TEAM_CODES};
}
