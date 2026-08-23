// Howie cockpit — application script. All HTML built from data goes through
// the auto-escaping h`` tag in lib.js; raw() marks the few deliberate
// exceptions (HTML we wrote ourselves).
const $ = (id) => document.getElementById(id);
const TOKEN = (document.querySelector('meta[name=howie-token]') || {}).content || '';
let ST = null, PICK = null, FILTER = 'ALL', RISK = null, notesTimer = null, strategy = {rules: [], notes: ''};

async function api(path, body) {
  const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Howie-Token': TOKEN}, body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const j = await r.json();
  if (j.error) { if (!path.startsWith('/api/config')) banner(j.error); throw new Error(j.error); }
  return j;
}
function banner(msg) { const b = $('banner'); b.style.display = msg ? 'block' : 'none'; b.textContent = msg || ''; }

// ---------------- state + board ----------------

async function refresh(fast) {
  try { [ST, RISK] = await Promise.all([api('/api/state'), api('/api/risk')]); }
  catch (e) { return; }  // banner already shows the server's message
  renderHeader(); renderRail(); renderTermHint();
  PICK = await api('/api/pick?top=25');
  renderBoard();
  if ($('strategyView').style.display !== 'none' && !fast) loadStrategyTab();
}

function drafting() { return !!(ST && ST.log.length && !ST.complete); }

function renderHeader() {
  $('draftpos').innerHTML = ST.complete
    ? 'DRAFT COMPLETE'
    : h`ROUND ${ST.round} · PICK ${ST.next_pick_no} ${ST.you_are_on_clock ? raw('<span class="clockchip">ON THE CLOCK</span>') : h`<span class="dim">team ${ST.on_clock_team} up</span>`}`;
  $('modechip').textContent = `${ST.mode} · ${ST.league.teams}t slot ${ST.league.slot} · ${ST.league.scoring}`;
  banner(ST.complete ? 'DRAFT COMPLETE — final roster under ROSTER. Reset (↺) to run another mock.' : '');
}

function renderRail() {
  $('rosterCount').textContent = `${ST.roster.filter(s => s.name).length} / ${ST.roster.length}`;
  const lvl = (slot) => { const pos = slot === 'FLX' ? null : slot; const r = pos && RISK && RISK.positions[pos]; return r && r.level !== 'ok' ? r.level : ''; };
  const why = (slot) => { const r = RISK && RISK.positions[slot]; return r && r.reasons.length ? r.reasons.join(' · ') : ''; };
  const risks = RISK ? Object.entries(RISK.positions).filter(([p, r]) => r.level !== 'ok') : [];
  const chips = risks.map(([p, r]) => h`<span class="riskchip ${r.level}" title="${r.reasons.join(' · ')}">${p} ${r.level === 'danger' ? 'THIN' : 'DUE'}</span>`);
  $('riskChips').innerHTML = h`${chips}`;
  $('riskDetail').innerHTML = risks.length
    ? h`${risks.map(([p, r]) => h`<div><b class="${r.level === 'danger' ? 'red' : 'amber'}">${p}</b> — ${r.reasons.join(' · ')}</div>`)}`
    : '<span class="dim">no positional risk right now</span>';
  const mine = ST.roster.filter(s => s.name);
  $('rosterStrip').innerHTML = h`<span class="mono dim" style="font-size:10px;letter-spacing:1.5px">ROSTER</span>${
    mine.length ? mine.map(s => h`<span><span class="kindtag">${s.pos}</span> ${s.name}</span>`) : raw('<span class="dim">empty</span>')
  }<span class="dim">· ${ST.roster.length - mine.length} open</span>${chips}<span style="flex:1"></span><span class="dim" style="cursor:pointer" onclick="showTab('roster')">full roster →</span>`;
  $('roster').innerHTML = h`${ST.roster.map(s => h`<div class="slotrow ${s.name ? 'filled' : lvl(s.slot)}" title="${s.name ? '' : why(s.slot)}">
      <span class="slot">${s.slot}</span>
      <span style="color:${s.name ? 'var(--hi)' : 'var(--dim)'}; overflow:hidden; text-overflow:ellipsis;">${s.name || '—'}</span>
      <span style="flex:1"></span><span class="mono dim" style="font-size:11px">${s.proj ?? ''}</span>
    </div>`)}`;
  $('log').innerHTML = ST.log.length ? h`${ST.log.map(l => h`<div class="logrow ${l.mine ? 'mine' : ''}">
      <span class="mono dim">${l.pick_no}</span><span>${l.name}</span>
      <span class="mono dim" style="font-size:10px">${l.position || ''} · ${l.mine ? 'YOU' : 'T' + l.team}</span>
    </div>`)}` : '<span class="dim" style="font-size:12px">no picks yet</span>';
  $('nextPicks').textContent = ST.my_next_picks.join(' · ') || '—';
}

let COMPACT = localStorage.getItem('boardCompact') !== '0';
function toggleCompact() { COMPACT = !COMPACT; localStorage.setItem('boardCompact', COMPACT ? '1' : '0'); renderBoard(); }

function renderBoard() {
  $('availHead').textContent = `AVAIL @${PICK.next_pick}`;
  $('boardTable').classList.toggle('compact', COMPACT);
  $('moreLink').textContent = COMPACT ? 'MORE ▸' : 'LESS ▾';
  const mc = PICK.mc, span = mc ? mc.outcome_span : null;
  const mcRows = mc ? Object.fromEntries(mc.rows.map(r => [r.uid, r])) : {};
  let rows = PICK.rows.filter(r => FILTER === 'ALL' || r.pos === FILTER);
  if (mc) rows = [...rows].sort((a, b) =>
    (mcRows[b.uid] ? mcRows[b.uid].value : b.value - 900) - (mcRows[a.uid] ? mcRows[a.uid].value : a.value - 900));
  rows = rows.slice(0, 10);
  const best = rows.length ? (mc && mcRows[rows[0].uid] ? mcRows[rows[0].uid].value : rows[0].value) : 0;
  $('rows').innerHTML = h`${rows.map((r, i) => {
    const m = mcRows[r.uid];
    const value = m ? m.value : r.value;
    const delta = Math.round(value - best);
    const dc = delta === 0 ? 'green' : (delta > -8 ? 'mid' : 'red');
    const p = r.avail_next;
    const lab = r.avail_src && r.avail_src !== 'model';
    const availbar = h`<span class="bar" title="${lab ? 'availability: ' + r.avail_src + ' (mock lab blended with the ADP model)' : 'availability: ADP model'}"><i style="width:${Math.round(p * 100)}%; background:var(--${availClass(p)})"></i></span><span class="mono mid" style="font-size:12px">${Math.round(p * 100)}%</span>${lab ? raw('<span class="labtag" title="blended with mock-lab drafts">LAB</span>') : ''}`;
    let dist = raw('<span class="dim mono" style="font-size:11px">sims…</span>');
    if (m && span) {
      const [lo, hi] = span, w = hi - lo;
      const L = (v) => ((v - lo) / w * 100).toFixed(1) + '%';
      const W = (a, b) => ((b - a) / w * 100).toFixed(1) + '%';
      const c1 = m.p10 + (m.p90 - m.p10) * .25, c2 = m.p10 + (m.p90 - m.p10) * .75;
      dist = h`<div class="dist"><div class="band" style="left:${L(m.p10)}; width:${W(m.p10, m.p90)}"></div><div class="core" style="left:${L(c1)}; width:${W(c1, c2)}"></div><div class="tick" style="left:${L(m.value)}"></div></div>`;
    }
    const tags = (r.rules || []).map(f => h`<span class="ruletag ${f.type}">${f.text}</span>`);
    const stchip = r.status ? h`<span class="stchip ${r.status.level}" title="player status (roster feed / research)">${r.status.text}</span>` : '';
    const isBest = i === 0 && FILTER === 'ALL';
    return h`<tr class="${isBest ? 'best' : ''}" onclick="openCard('${r.uid}')" style="cursor:pointer">
      <td class="mono dim">${i + 1}</td>
      <td><span class="kindtag">${r.pos}</span> <b style="font-weight:500">${r.name}</b> <span class="mono dim" style="font-size:11px">${r.team || ''}</span>${isBest ? raw('<span class="besttag">BEST</span>') : ''}${stchip}${tags}</td>
      <td class="mono r mid c-proj">${r.proj}</td>
      <td class="mono r mid c-adp">${r.adp ? r.adp.toFixed(1) : '—'}</td>
      <td>${availbar}</td>
      <td class="mono r" style="font-weight:600">${value}</td>
      <td class="c-dist">${dist}</td>
      <td class="mono r ${dc}" style="font-weight:600">${fmtDelta(delta)}</td>
      <td class="mono dim c-plan" style="font-size:11px">${(r.plan || []).join(' ')}</td>
    </tr>`;
  })}`;
  $('mcstatus').textContent = mc ? `MC ${mc.sims} sims ready` : 'MC running…';
  $('boardFoot').textContent = `Value = expected final starting-lineup points with bench insurance (take this player now, then draft optimally). Δ vs best. ` +
    (span ? `Outcome bars: p10–p90 across ${mc.sims} simulated seasons, scale ${span[0]}–${span[1]}, tick = mean.` : 'Monte Carlo refinement runs after every pick.');
}

// ---------------- position filter chips ----------------

['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST'].forEach(p => {
  const el = document.createElement('span');
  el.className = 'poschip' + (p === 'ALL' ? ' active' : '');
  el.textContent = p;
  el.onclick = () => { FILTER = p; document.querySelectorAll('#poschips .poschip').forEach(c => c.classList.toggle('active', c.textContent === p)); renderBoard(); };
  $('poschips').appendChild(el);
});

// ---------------- actions ----------------

async function mark(uid, mine) {
  const r = await api('/api/mark', {uid, mine});
  termPrint('out', `${mine ? 'drafted' : 'marked taken'}: ${r.name} (pick ${r.pick_no})` + (r.bots ? ` · bots made ${r.bots.length} picks` : '') + ' · /undo reverts');
  closeCard(); await refresh(true);
}
async function undoPick() { const r = await api('/api/undo', {}); termPrint('out', r.undone ? `undid ${r.undone.name} (pick ${r.undone.pick_no})` : 'nothing to undo'); await refresh(true); }
async function startMock() {
  if (ST && ST.log.length && !confirm('Start a fresh mock draft? Current draft state is cleared.')) return;
  await api('/api/mock/start', {}); await refresh(true);
}
async function resetDraft() {
  if (!confirm('Reset the draft log?')) return;
  await api('/api/reset', {mode: 'live'}); await refresh(true);
}

// ---------------- tabs + strategy ----------------

function showTab(t, opts = {}) {
  $('boardView').style.display = t === 'board' ? '' : 'none';
  $('strategyView').style.display = t === 'strategy' ? 'block' : 'none';
  $('dataView').style.display = t === 'data' ? 'block' : 'none';
  $('simView').style.display = t === 'sim' ? 'block' : 'none';
  $('rosterView').style.display = t === 'roster' ? 'block' : 'none';
  $('teamView').style.display = t === 'team' ? 'block' : 'none';
  for (const [id, key] of [['tabRoster', 'roster'], ['tabBoard', 'board'], ['tabStrategy', 'strategy'], ['tabData', 'data'], ['tabSim', 'sim'], ['tabTeam', 'team']]) $(id).classList.toggle('active', t === key);
  if (t === 'team' && !opts.noload) loadTeamTab();
  if (t === 'strategy') loadStrategyTab();
  if (t === 'data') loadDataTab();
  if (t === 'sim') loadSimTab();
}

async function loadAnchors() {
  const a = await api('/api/anchors');
  const [s0, s1, s2] = a.seasons;
  const KEY = {QB: ['300+ pass yds', '3+ pass TD'], RB: ['100+ rush yds', '2+ TD'], WR: ['100+ rec yds', 'TD'], TE: ['75+ rec yds', 'TD']};
  const rows = [h`<div class="anchorrow dim" style="font-size:10px;letter-spacing:1px"><span></span><span class="r">${s0}</span><span class="r">${s1}</span><span class="r">${s2}</span><span></span></div>`];
  for (const pos of Object.keys(KEY)) for (const m of KEY[pos]) {
    const t = (a.league[pos] || {})[m] || {};
    const v = (y) => t[y] != null ? Math.round(t[y] * 100) + '%' : '—';
    const d = (t[s2] || 0) - (t[s0] || 0);
    const arrow = Math.abs(d) < 0.03 ? raw('<span class="dim">→</span>') : (d > 0 ? raw('<span class="green">▲</span>') : raw('<span class="red">▼</span>'));
    rows.push(h`<div class="anchorrow"><span><span class="kindtag">${pos}</span> ${m}</span><span class="mono r mid">${v(s0)}</span><span class="mono r mid">${v(s1)}</span><span class="mono r" style="font-weight:600">${v(s2)}</span><span class="r">${arrow}</span></div>`);
  }
  $('leagueAnchors').innerHTML = h`${rows}`;
  const r = a.roster;
  if (!r.starters.length) { $('rosterAnchors').innerHTML = '<span class="dim" style="font-size:12px">Draft some starters — anchors appear as your roster fills.</span>'; return; }
  $('rosterAnchors').innerHTML = h`
    <div style="display:flex;gap:18px;margin-bottom:10px">
      <div><div class="dim" style="font-size:9px;letter-spacing:1.5px">BIG GAMES / WEEK</div><div class="mono" style="font-size:18px;font-weight:600">${r.expected_booms_per_week}</div></div>
      <div><div class="dim" style="font-size:9px;letter-spacing:1.5px">TDs / WEEK</div><div class="mono" style="font-size:18px;font-weight:600">${r.expected_tds_per_week}</div></div>
      <div><div class="dim" style="font-size:9px;letter-spacing:1.5px">P(ANY BOOM)</div><div class="mono green" style="font-size:18px;font-weight:600">${Math.round(r.p_any_boom * 100)}%</div></div>
    </div>${r.starters.map(st => h`<div class="anchorrow" style="grid-template-columns:1fr 110px 50px 40px"><span><span class="kindtag">${st.position}</span> ${st.name}</span><span class="mono dim" style="font-size:10px">${st.boom}</span><span class="mono r" style="font-weight:600">${Math.round(st.boom_rate * 100)}%</span><span class="mono r dim" style="font-size:10px">${st.tds_per_game} td</span></div>`)}
    <div class="dim" style="font-size:11px;margin-top:8px">Big game = the position's headline milestone; counts over each player's last two seasons, tier baseline for rookies.</div>`;
}

async function loadStrategyTab() {
  loadAnchors();
  strategy = await api('/api/strategy');
  renderRules();
  if (!$('notes').matches(':focus')) $('notes').value = strategy.notes;
  const pos = await api('/api/positions');
  $('posHead').textContent = `POSITIONAL IMPACT AT PICK ${pos.current_pick} — DRAFT NOW vs WAIT UNTIL ${pos.next_pick}`;
  const vals = pos.rows.flatMap(r => [r.now, r.wait]);
  const lo = Math.min(...vals) - 8, hi = Math.max(...vals) + 4, w = hi - lo;
  $('posRows').innerHTML = h`${pos.rows.map((r, i) => {
    const W = (v) => Math.max(((v - lo) / w * 100), 2).toFixed(1) + '%';
    const cc = r.cost >= 10 ? 'amber' : (r.cost > 0 ? 'mid' : 'dim');
    return h`<div class="posrow">
      <div><span class="kindtag">${r.pos}</span> ${i === 0 ? raw('<span class="besttag">TOP</span>') : ''}<br>
        <span class="mid" style="font-size:12px">${r.player} · ${r.player_proj}</span></div>
      <div>
        <div style="display:flex;align-items:center;gap:8px;margin:2px 0"><span class="mono dim" style="font-size:9px;width:28px">NOW</span><div class="nwbar" style="flex:1"><i style="width:${W(r.now)};background:${i === 0 ? 'var(--acc)' : '#3a4a40'}"></i></div><span class="mono" style="font-size:12px;width:38px;text-align:right">${r.now}</span></div>
        <div style="display:flex;align-items:center;gap:8px;margin:2px 0"><span class="mono dim" style="font-size:9px;width:28px">WAIT</span><div class="nwbar" style="flex:1"><i style="width:${W(r.wait)};background:#2a3a30"></i></div><span class="mono mid" style="font-size:12px;width:38px;text-align:right">${r.wait}</span></div>
      </div>
      <div class="mono r ${cc}" style="font-size:16px;font-weight:600">+${r.cost}</div>
      <div class="mid" style="font-size:11px; white-space:normal">${r.player.split(' ').slice(-1)[0]} ${Math.round(r.avail_next * 100)}% at ${pos.next_pick} · next tier −${r.tier_drop} pts</div>
    </div>`;
  })}`;
}

function renderRules() {
  $('rules').innerHTML = strategy.rules.length ? h`${strategy.rules.map((r, i) => h`<div class="rulerow ${r.on ? '' : 'off'}">
      <span class="dot" onclick="toggleRule(${i})"></span>
      <span class="mono" style="font-size:12px">${r.text}</span>
      ${r.inert ? raw('<span class="ruletag ban" title="Matches no known pattern — will not affect the board">INERT</span>') : ''}
      <span style="flex:1"></span>
      <span class="mono dim" style="font-size:10px">${r.on ? 'ON' : 'OFF'}</span>
      <button style="padding:1px 7px" onclick="delRule(${i})">×</button>
    </div>`)}` : '<span class="dim" style="font-size:12px">no rules pinned</span>';
}
async function saveStrategy() {
  strategy = await api('/api/strategy', {rules: strategy.rules, notes: $('notes').value});
  renderRules();
  (strategy.conflicts || []).forEach(c => termPrint('dim', h`<span class="conflict">rule conflict:</span> ${c}`));
  $('notesSaved').textContent = '— saved';
  setTimeout(() => $('notesSaved').textContent = '', 1500);
  refresh(true);
}
function addRule() {
  const t = $('newRule').value.trim();
  if (!t) return;
  strategy.rules.push({text: t, on: true}); $('newRule').value = '';
  saveStrategy();
}
function toggleRule(i) { strategy.rules[i].on = !strategy.rules[i].on; saveStrategy(); }
function delRule(i) { strategy.rules.splice(i, 1); saveStrategy(); }
$('notes').addEventListener('input', () => { clearTimeout(notesTimer); notesTimer = setTimeout(saveStrategy, 900); });
$('newRule').addEventListener('keydown', e => { if (e.key === 'Enter') addRule(); });

// ---------------- player card ----------------

let CARD = null;
let PREP = localStorage.getItem('cardPrep') || 'auto';  // auto: fold prep sections while on the clock
function togglePrep() { PREP = $('drawer').classList.contains('clock') ? 'always' : 'auto'; localStorage.setItem('cardPrep', PREP); applyCardMode(); }
function applyCardMode() {
  const clock = PREP === 'auto' && drafting() && ST.you_are_on_clock;
  $('drawer').classList.toggle('clock', clock);
  const t = $('prepToggle'); if (t) t.textContent = clock ? 'SHOW PREP ▸' : 'FOLD PREP ▾';
}

async function openCard(uid) {
  const c = await api('/api/card?uid=' + encodeURIComponent(uid));
  const room = c.room, facts = [...(c.facts || []), ...(c.team_facts || [])].filter(f => f.source !== 'derived').slice(0, 3);
  const derived = [...(room && room.facts ? room.facts : []), ...(c.team_facts || [])].filter(f => f.source === 'derived').slice(0, 3);
  const bandBar = () => {
    const lo = c.band.p10 * .85, hi = c.band.p90 * 1.1, w = hi - lo;
    const L = (v) => ((v - lo) / w * 100).toFixed(1) + '%', W2 = (a, b) => ((b - a) / w * 100).toFixed(1) + '%';
    return h`<div class="dist" style="width:72px;margin:4px auto 0"><div class="band" style="left:${L(c.band.p10)};width:${W2(c.band.p10, c.band.p90)}"></div><div class="tick" style="left:${L(c.band.p50)}"></div></div>`;
  };
  const derivedHtml = derived.map(f => h`<div class="factcard">${f.text}<div class="fmeta">derived from 2025 data</div></div>`);
  const researchHtml = facts.length ? facts.map(f => h`<div class="factcard">${f.text}<div class="fmeta">${f.source} · conf ${f.confidence}</div></div>`)
    : h`<div class="dim" style="font-size:11px;margin-top:6px">No researched facts yet — <span class="green" style="cursor:pointer" data-team="${c.team || ''}" onclick="handleTerm('/research ' + this.dataset.team, [])">/research ${c.team || ''}</span></div>`;
  const fired = (PICK && PICK.rows.find(r => r.uid === c.uid) || {}).rules || [];
  $('drawer').innerHTML = h`<div class="cardgrid">
    <div class="cardcol">
      <div class="cardhead">
        <div style="font-size:19px;font-weight:700;line-height:1.15">${c.name}</div>
        <div class="mono mid" style="font-size:11px;margin-top:3px"><span class="kindtag">${c.pos}</span> ${c.team || ''} · BYE ${c.bye || '—'} · ADP ${c.adp ? c.adp.toFixed(1) : '—'}${c.adp_stdev ? ' ± ' + c.adp_stdev.toFixed(1) : ''}</div>
        ${c.status ? h`<div class="stline"><span class="stchip ${c.status.level}">${c.status.text}</span> <span class="mid">${c.status_detail.note || c.status_detail.injury || ''}</span> <span class="dim">· ${c.status_detail.source} · ${c.status_detail.as_of}${c.status_detail.role && c.status_detail.role !== 'unknown' ? ' · ' + c.status_detail.role : ''}</span></div>` : ''}
        ${c.taken ? h`<div class="ctarow"><span class="mono amber" style="font-size:11px;letter-spacing:1.5px;padding:9px 0">TAKEN · PICK ${c.taken_pick} · ${(c.taken_by || '').toUpperCase()}</span></div>`
          : h`<div class="ctarow"><button class="danger" onclick="mark('${c.uid}', false)">MARK TAKEN</button><button class="primary" onclick="mark('${c.uid}', true)">DRAFT TO ME</button></div>`}
      </div>
      <div class="stat4">
        <div><div class="lbl">PROJ</div><div class="val">${c.proj}</div></div>
        <div><div class="lbl">VS WAIT (${c.pos})</div><div class="val green">${fmtDelta(c.mv_vs_wait)}</div></div>
        <div><div class="lbl">AVAIL @${c.next_pick}</div><div class="val amber">${Math.round(c.avail_next * 100)}%</div></div>
        <div><div class="lbl">P10–P90</div><div class="val" style="font-size:13px">${c.band.p10}–${c.band.p90}</div>${bandBar()}</div>
      </div>
      ${c.taken ? '' : h`<div class="reason"><b>${reasonLine(c, RISK)}</b>${fired.length ? h` · ${fired.map(f => h`<span class="ruletag ${f.type}">${f.text}</span>`)}` : ''}</div>`}
      <span class="preptoggle" id="prepToggle" onclick="togglePrep()"></span>
      ${c.trend.length ? h`<div class="shrink prep"><p class="sechead">TREND · PTS/GAME</p><div style="display:flex;align-items:flex-end;gap:14px;height:64px">${
        c.trend.map((t, i) => h`<div style="flex:1;text-align:center"><div class="mono" style="font-size:11px">${t.ppg}</div><div style="height:${Math.min(t.ppg * 2.2, 44)}px;background:${i === c.trend.length - 1 ? 'var(--acc)' : '#3a4a40'};margin:3px 0"></div><div class="mono dim" style="font-size:10px">${t.season}</div></div>`)}</div></div>` : ''}
    </div>
    <div class="cardcol prep">
      ${room ? h`<p class="sechead">ROOM · ${room.unit.replace('unit:', '')} — LAST-SEASON SHARE</p>${
        room.members.slice(0, 5).map(m => h`<div class="roomrow" title="${m.other_team ? 'share earned with ' + m.other_team + ' last season' : ''}"><span style="width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${m.name}${m.other_team ? raw('<span class="amber">*</span>') : ''}</span><div class="rbar"><i style="width:${Math.round((m.share || 0) * 300)}px;max-width:100%;${m.name === c.name ? 'background:var(--acc)' : (m.other_team ? 'background:#5a4a1a' : '')}"></i></div><span class="mono mid" style="width:38px;text-align:right">${m.share != null ? Math.round(m.share * 100) + '%' : '—'}</span></div>`)}${
        room.members.slice(0, 5).some(m => m.other_team) ? h`<div class="dim" style="font-size:10px;margin-top:2px"><span class="amber">*</span> share earned in a different offense (${room.members.filter(m => m.other_team).slice(0, 5).map(m => m.name.split(' ').slice(-1)[0] + ' ' + m.other_team).join(', ')})</div>` : ''}` : ''}
      <div class="fill">${derivedHtml}<p class="sechead" style="margin-top:10px">RESEARCH</p>${researchHtml}</div>
    </div>
    <div class="cardcol prep">
      ${c.playoff_sos.length ? h`<p class="sechead">PLAYOFF SOS · W15–17</p><div style="display:flex;gap:6px">${
        c.playoff_sos.map(s2 => h`<div style="flex:1;text-align:center;background:${s2.value >= 6 ? '#12241a' : s2.value >= 4.5 ? '#171509' : '#241210'};padding:8px 0"><div class="dim" style="font-size:9px">W${s2.week}</div><div class="mono" style="font-weight:600;color:${s2.value >= 6 ? 'var(--acc)' : s2.value >= 4.5 ? 'var(--amber)' : 'var(--red)'}">${s2.value}</div></div>`)}</div>` : ''}
      ${c.milestones && c.milestones.labels.length ? h`<div class="shrink" style="margin-top:12px"><p class="sechead">MILESTONE RATES · LAST TWO SEASONS</p>${
        c.milestones.labels.map(l => { const pr = c.milestones.player[l] || 0, tr = c.milestones.tier[l]; return h`<div class="roomrow"><span style="width:118px" class="mono" title="${l}">${l}</span><div class="rbar"><i style="width:${Math.round(pr * 100)}%;background:${pr >= (tr || 0) ? 'var(--acc)' : '#3a4a40'}"></i></div><span class="mono mid" style="width:74px;text-align:right">${Math.round(pr * 100)}%<span class="dim"> / ${tr != null ? Math.round(tr * 100) + '%' : '—'}</span></span></div>`; })}
        <div class="dim" style="font-size:10px;margin-top:4px">player / starter-tier average</div></div>` : ''}
    </div>
    <div class="cardcol prep">
      ${c.games && c.games.length ? h`<p class="sechead">GAME LOG ${c.milestones.seasons.join('–')} · ${c.games.length} GAMES <span class="dim" style="letter-spacing:0">hover a bar</span></p>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px" id="mchips"></div>
        <div class="glog" id="glog"></div>
        <div class="mono dim shrink" style="font-size:11px;margin-top:6px" id="mrate"></div>` : raw('<span class="dim" style="font-size:12px">No game history.</span>')}
    </div>
  </div>`;
  sideOpen(); $('drawer').scrollTop = 0; $('sideLabel').textContent = 'PLAYER'; $('sideHint').textContent = c.team ? `${c.pos} · ${c.team}` : '';
  CARD = c;
  applyCardMode();
  if (c.games && c.games.length) renderGlog(c.milestones.labels[0]);
}

function renderGlog(active) {
  const c = CARD, labels = c.milestones.labels;
  $('mchips').innerHTML = h`${labels.map((l, li) => h`<span class="mchip ${l === active ? 'active' : ''}" onclick="renderGlog(CARD.milestones.labels[${li}])">${l}</span>`)}`;
  const maxPts = Math.max(...c.games.map(g => g.pts), 1);
  const bars = [];
  let lastSeason = null;
  c.games.forEach((g, i) => {
    if (lastSeason !== null && g.season !== lastSeason) bars.push(raw('<div class="gbar sep"></div>'));
    lastSeason = g.season;
    const hgt = Math.max(3, Math.round(g.pts / maxPts * 56));
    bars.push(h`<div class="gbar ${g.flags[active] ? 'hit' : ''}" style="height:${hgt}px" data-i="${i}" onmouseenter="showTip(event, ${i})" onmouseleave="hideTip()"></div>`);
  });
  $('glog').innerHTML = h`${bars}`;
  $('glog').dataset.active = active;
  const pr = c.milestones.player[active], tr = c.milestones.tier[active];
  $('mrate').textContent = `${active}: ${Math.round(pr * 100)}% of games` + (tr != null ? ` · starter-tier average ${Math.round(tr * 100)}%` : '') + ` · bar height = fantasy points`;
}
function showTip(ev, i) {
  const g = CARD.games[i], active = $('glog').dataset.active;
  const stat = CARD.pos === 'QB' ? `${g.pass_yds} pass yds · ${g.pass_tds} pass TD · ${g.rush_tds} rush TD`
    : CARD.pos === 'RB' ? `${g.rush_yds} rush · ${g.rec} rec ${g.rec_yds} yds · ${g.rush_tds + g.rec_tds} TD`
    : `${g.targets} tgt · ${g.rec} rec · ${g.rec_yds} yds · ${g.rec_tds} TD`;
  placeTip(ev, h`<b>${g.season} W${g.week}</b> vs ${g.opp || '—'} · <span class="${g.flags[active] ? 'green' : 'dim'}">${g.flags[active] ? '✓' : '✗'} ${active}</span><br>${stat}<br><b>${g.pts} pts</b>`);
}
function placeTip(ev, html) {
  const t = $('tip'); t.innerHTML = html; t.style.display = 'block';
  t.style.left = Math.min(ev.clientX + 14, window.innerWidth - 260) + 'px';
  t.style.top = (ev.clientY - 10) + 'px';
}
function hideTip() { $('tip').style.display = 'none'; }
function closeCard() { $('drawer').innerHTML = ''; $('findings').innerHTML = ''; $('findings').classList.remove('on'); $('side').classList.add('collapsed'); CARD = null; }

// ---------------- DATA tab ----------------

let DPOS = 'RB', DIST = null, dataLoaded = false;
['QB', 'RB', 'WR', 'TE'].forEach(p => {
  const el = document.createElement('span');
  el.className = 'poschip' + (p === 'RB' ? ' active' : ''); el.textContent = p;
  el.onclick = () => { DPOS = p; document.querySelectorAll('#dpos .poschip').forEach(c => c.classList.toggle('active', c.textContent === p)); loadDist(); };
  $('dpos').appendChild(el);
});
['dstat', 'dtier'].forEach(id => $(id).addEventListener('change', loadDist));
$('dthr').addEventListener('input', () => renderDist());
$('dhl').addEventListener('input', () => renderDist());

async function loadDataTab() {
  if (!dataLoaded) { dataLoaded = true; await loadDist(); loadPresets(); loadResearch(); }
}

async function loadDist() {
  DIST = await api(`/api/data/games?pos=${DPOS}&stat=${encodeURIComponent($('dstat').value)}&tier=${encodeURIComponent($('dtier').value)}`);
  renderDist();
}

function renderDist() {
  if (!DIST) return;
  const rows = DIST.rows, W = $('dplot').clientWidth || 900, H = 300, padL = 60, padR = 20, padT = 14, rowH = (H - padT - 30) / DIST.seasons.length;
  const vals = rows.map(r => r[6]); const maxV = Math.max(...vals, 1), minV = Math.min(...vals, 0);
  const x = v => padL + (v - minV) / (maxV - minV) * (W - padL - padR);
  const thr = parseFloat($('dthr').value), hl = $('dhl').value.trim().toLowerCase();
  const jit = i => ((i * 9301 + 49297) % 233280) / 233280;  // deterministic jitter: stable hover targets
  let svg = `<svg id="dplot" class="plot" width="${W}" height="${H}">`;
  DIST.seasons.forEach((season, si) => {
    const y0 = padT + si * rowH;
    svg += `<text x="6" y="${y0 + rowH / 2 + 4}" fill="#9fb0a4" font-family="IBM Plex Mono" font-size="11">${esc(season)}</text>`;
    svg += `<line x1="${padL}" x2="${W - padR}" y1="${y0 + rowH - 2}" y2="${y0 + rowH - 2}" stroke="#1a221c"/>`;
  });
  for (let t = 0; t <= 5; t++) { const v = minV + (maxV - minV) * t / 5; svg += `<text x="${x(v)}" y="${H - 8}" fill="#5c6b60" font-family="IBM Plex Mono" font-size="10" text-anchor="middle">${Math.round(v)}</text>`; }
  rows.forEach((r, i) => {
    const si = DIST.seasons.indexOf(r[2]); const y0 = padT + si * rowH;
    const isHl = hl && r[1].toLowerCase().includes(hl);
    const above = !isNaN(thr) && r[6] >= thr;
    const fill = isHl ? '#3ddb84' : (above ? '#2f8a55' : '#2a3a30');
    svg += `<circle class="dot" data-i="${i}" cx="${x(r[6]).toFixed(1)}" cy="${(y0 + 6 + jit(i) * (rowH - 14)).toFixed(1)}" r="${isHl ? 4 : 2.6}" fill="${fill}" fill-opacity="${isHl ? 1 : 0.75}"/>`;
  });
  if (!isNaN(thr)) svg += `<line x1="${x(thr)}" x2="${x(thr)}" y1="${padT}" y2="${H - 28}" stroke="#e8b84b" stroke-dasharray="3,3"/>`;
  svg += '</svg>';
  $('dplot').outerHTML = svg;
  $('dplot').addEventListener('mouseover', e => {
    const i = e.target.dataset && e.target.dataset.i; if (i === undefined) return;
    const r = DIST.rows[i];
    placeTip(e, h`<b>${r[1]}</b> · ${r[2]} W${r[3]} ${r[5] || ''} vs ${r[4] || '—'}<br>${DIST.stat}: <b>${r[6]}</b> · ${r[7]} pts`);
  });
  $('dplot').addEventListener('mouseout', hideTip);
  $('dplot').addEventListener('click', e => {
    const i = e.target.dataset && e.target.dataset.i; if (i === undefined) return;
    const r = DIST.rows[i];
    const his = DIST.rows.filter(x => x[0] === r[0]);
    const avg = his.reduce((a, x) => a + x[6], 0) / his.length;
    $('dhl').value = r[1]; $('dthr').value = avg.toFixed(1); hideTip(); renderDist();
  });
  $('dmeta').textContent = `${rows.length} games · ${DIST.tier === 'starter' ? 'starter tier' : 'all players'}`;
  if (!isNaN(thr)) {
    const hlRows = hl ? rows.filter(r => r[1].toLowerCase().includes(hl)) : [];
    $('dthrline').innerHTML = h`${DIST.seasons.map((season, i) => {
      const g = rows.filter(r => r[2] === season), a = g.filter(r => r[6] >= thr).length;
      return h`${i ? ' · ' : ''}${season}: <b>${Math.round(100 * a / Math.max(g.length, 1))}%</b> of games ≥ ${thr}`;
    })}${hlRows.length ? h` · <span class="green">${hlRows[0][1]}</span>: <b>${Math.round(100 * hlRows.filter(r => r[6] >= thr).length / hlRows.length)}%</b> of his ${hlRows.length} games` : ''}`;
  } else $('dthrline').textContent = 'Set a threshold to see % of games above it, per season.';
}

function hist(svgId, samples, color, marks, note) {
  const W = $(svgId).clientWidth || 440, H = 180, padL = 10, padB = 22;
  if (!samples.length) { $(svgId).outerHTML = `<svg id="${svgId}" class="plot" width="${W}" height="${H}"><text x="${W / 2}" y="${H / 2}" fill="#5c6b60" font-family="IBM Plex Mono" font-size="11" text-anchor="middle">${esc(note)}</text></svg>`; return; }
  const lo = Math.min(...samples, ...marks.map(m => m.v)), hi = Math.max(...samples, ...marks.map(m => m.v));
  const nb = 28, bw = (hi - lo) / nb || 1, bins = new Array(nb).fill(0);
  samples.forEach(v => bins[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
  const maxB = Math.max(...bins), x = v => padL + (v - lo) / (hi - lo || 1) * (W - 2 * padL), cw = (W - 2 * padL) / nb;
  let svg = `<svg id="${svgId}" class="plot" width="${W}" height="${H}">`;
  bins.forEach((b, i) => { const hh = b / maxB * (H - padB - 10); svg += `<rect x="${padL + i * cw + 1}" y="${H - padB - hh}" width="${cw - 2}" height="${hh}" fill="${color}" fill-opacity="0.6"/>`; });
  marks.forEach(m => { svg += `<line x1="${x(m.v)}" x2="${x(m.v)}" y1="8" y2="${H - padB}" stroke="${m.c}" stroke-width="2" stroke-dasharray="${m.dash || '0'}"/><text x="${x(m.v) + 4}" y="${m.y || 18}" fill="${m.c}" font-family="IBM Plex Mono" font-size="10">${esc(m.l)}</text>`; });
  for (let t = 0; t <= 4; t++) { const v = lo + (hi - lo) * t / 4; svg += `<text x="${x(v)}" y="${H - 6}" fill="#5c6b60" font-family="IBM Plex Mono" font-size="10" text-anchor="middle">${Math.round(v)}</text>`; }
  $(svgId).outerHTML = svg + '</svg>';
}

async function loadSim() {
  const q = $('simq').value.trim(); if (!q) return;
  const hits = await api('/api/search?q=' + encodeURIComponent(q));
  const p = hits.find(x => x.uid); if (!p) { $('simmeta').textContent = 'no player found'; return; }
  const sres = await api('/api/data/sim?uid=' + encodeURIComponent(p.uid));
  const marks = [{v: sres.p10, c: '#5c6b60', l: 'p10', dash: '3,3'}, {v: sres.p50, c: '#3ddb84', l: 'p50 ' + sres.p50}, {v: sres.p90, c: '#5c6b60', l: 'p90', dash: '3,3'}];
  sres.actual.forEach((a, i) => marks.push({v: a.total, c: '#e8b84b', l: `${a.season} actual ${a.total}`, y: 34 + i * 13}));
  hist('simplot', sres.samples, '#3ddb84', marks, '');
  $('simmeta').textContent = `${sres.name} (${sres.position}) · proj ${sres.proj} · engine value ${sres.value} · ${sres.samples.length} simulated seasons`;
  $('simnote').textContent = `model: ${sres.model.weekly_mu} pts/wk · weekly cv ${sres.model.cv} · P(play) ${sres.model.p_play} · season shock σ ${sres.model.season_sigma}. Amber lines = his real past seasons.`;
}

async function loadRosterSim() {
  const r = await api('/api/data/roster_sim');
  if (!r.samples.length) { hist('rsplot', [], '', [], 'Draft players — your roster\'s simulated season appears here.'); $('rsnote').textContent = ''; return; }
  hist('rsplot', r.samples, '#7dd3fc', [{v: r.p10, c: '#5c6b60', l: 'p10', dash: '3,3'}, {v: r.mean, c: '#7dd3fc', l: 'mean ' + r.mean}, {v: r.p90, c: '#5c6b60', l: 'p90', dash: '3,3'}], '');
  $('rsnote').textContent = `your roster (${r.players.length}: ${r.players.slice(0, 4).join(', ')}${r.players.length > 4 ? '…' : ''}) · starting-lineup points over ${r.samples.length} simulated seasons`;
}

async function loadPresets() {
  const p = await api('/api/data/query?q=');
  $('qpresets').innerHTML = h`${p.presets.map(x => h`<span class="preset" data-sql="${x.sql}" onclick="$('qq').value = 'sql: ' + this.dataset.sql; runQuery()">${x.label}</span>`)}`;
}
$('qq').addEventListener('keydown', e => { if (e.key === 'Enter') runQuery(); });
function queryFor(el) { $('qq').value = el.dataset.q; runQuery(); }
function simFor(el) { $('simq').value = el.dataset.name; loadSim(); }

function qtable(columns, rows) {
  return h`<table class="qtable"><thead><tr>${columns.map(c => h`<th>${c}</th>`)}</tr></thead><tbody>${
    rows.map(row => h`<tr>${columns.map(c => h`<td>${row[c] === null || row[c] === undefined ? '—' : (typeof row[c] === 'number' ? +row[c].toFixed(2) : row[c])}</td>`)}</tr>`)}</tbody></table>`;
}

async function runQuery() {
  const q = $('qq').value.trim(); if (!q) return;
  const r = await api('/api/data/query?q=' + encodeURIComponent(q));
  if (r.mode === 'sql') {
    if (r.error) { $('qout').innerHTML = h`<span class="red mono" style="font-size:12px">${r.error}</span>`; return; }
    if (!r.rows.length) { $('qout').innerHTML = '<span class="dim">no rows</span>'; return; }
    $('qout').innerHTML = qtable(r.columns, r.rows);
    return;
  }
  if (!r.entity) { $('qout').innerHTML = '<span class="dim">nothing found</span>'; return; }
  const e = r.entity, d = r.detail, ctx = d.context || {};
  const parts = [h`<div style="display:flex;gap:10px;align-items:center;margin-bottom:8px"><span class="kindtag">${e.kind}</span><b>${e.name}</b><span class="mono dim" style="font-size:11px">${e.position || ''} ${e.team || ''}</span>${
    d.uid ? h`<button onclick="openCard('${d.uid}')">CARD</button><button data-name="${e.name}" onclick="simFor(this)">SIMULATE</button>` : ''}</div>`];
  if (r.hits.length > 1) parts.push(h`<div class="dim" style="font-size:11px;margin-bottom:8px">also: ${r.hits.slice(1).map(x => h`<span class="preset" data-q="${x.name}" onclick="queryFor(this)">${x.name}</span> `)}</div>`);
  if (d.projection) parts.push(h`<div class="mono mid" style="font-size:12px;margin-bottom:8px">2026 · proj <b>${d.projection.proj}</b> · engine value ${d.projection.value} · ADP ${d.projection.adp ?? '—'} · bye ${d.projection.bye ?? '—'}</div>`);
  if (d.seasons) parts.push(h`<table class="qtable"><thead><tr><th>season</th><th>g</th><th>pts</th><th>ppg</th><th>rush yds</th><th>rec yds</th><th>tgt</th><th>td</th></tr></thead><tbody>${
    d.seasons.map(s => h`<tr><td>${s.season}</td><td>${s.g}</td><td>${s.pts}</td><td>${s.ppg}</td><td>${s.rush_yds || 0}</td><td>${s.rec_yds || 0}</td><td>${s.tgt || 0}</td><td>${s.tds || 0}</td></tr>`)}</tbody></table>`);
  if (d.rooms) parts.push(h`<table class="qtable"><thead><tr><th>pos</th><th>player</th><th class="r">2026 proj</th><th class="r">ADP</th><th class="r">last-season share</th></tr></thead><tbody>${
    d.rooms.map(m => h`<tr><td>${m.position}</td><td>${m.name}${m.other_team ? h` <span class="amber" title="share earned with ${m.other_team}">*${m.other_team}</span>` : ''}</td><td class="r">${m.proj ?? '—'}</td><td class="r">${m.adp ?? '—'}</td><td class="r">${m.share != null ? Math.round(m.share * 100) + '%' : '—'}</td></tr>`)}</tbody></table>${
    d.rooms.some(m => m.other_team) ? raw('<div class="dim" style="font-size:10px;margin-top:4px"><span class="amber">*</span> last-season share earned with the team shown, not this offense</div>') : ''}`);
  if (ctx.room) parts.push(h`<div class="factcard">Room ${ctx.room.unit.replace('unit:', '')}: ${ctx.room.members.map(m => `${m.name} ${m.share != null ? Math.round(m.share * 100) + '%' : '—'}`).join(' · ')}</div>`);
  const facts = [...(ctx.facts || []), ...(ctx.team_facts || []), ...((ctx.room && ctx.room.facts) || [])];
  if (facts.length) parts.push(h`${facts.slice(0, 6).map(f => h`<div class="factcard">${f.text}<div class="fmeta">${f.source}${f.confidence != null ? ' · conf ' + f.confidence : ''}</div></div>`)}`);
  $('qout').innerHTML = h`${parts}`;
}

// ---------------- query builder ----------------

$('bMeasure').addEventListener('change', () => { $('bThr').style.display = $('bMeasure').value === 'games_over' ? '' : 'none'; });
async function runBuild() {
  const q = new URLSearchParams({entity: $('bEntity').value, pos: $('bPos').value, season: $('bSeason').value,
    measure: $('bMeasure').value, stat: $('bStat').value, thr: $('bThr').value || 100,
    min_games: $('bMin').value || 1, limit: $('bLimit').value, order: $('bOrder').value});
  const r = await api('/api/data/build?' + q.toString());
  $('bsql').textContent = r.sql;
  if (r.error) { $('qout').innerHTML = h`<span class="red mono" style="font-size:12px">${r.error}</span>`; return; }
  if (!r.rows.length) { $('qout').innerHTML = '<span class="dim">no rows</span>'; return; }
  $('qout').innerHTML = h`<div class="dim" style="font-size:11px;margin-bottom:6px">${r.label}</div>${qtable(r.columns, r.rows)}`;
}

// ---------------- LAB tab: mock draft lab ----------------

let MOCK = null, MOCKPOS = 'ALL', mockTimer = null, MOCK_loadedAfter = false;
['ALL', 'QB', 'RB', 'WR', 'TE'].forEach(p => {
  const el = document.createElement('span');
  el.className = 'poschip' + (p === 'ALL' ? ' active' : ''); el.textContent = p;
  el.onclick = () => { MOCKPOS = p; document.querySelectorAll('#mockPos .poschip').forEach(c => c.classList.toggle('active', c.textContent === p)); renderMock(); };
  $('mockPos').appendChild(el);
});
$('mockPick').addEventListener('change', renderMock);
$('mockN').addEventListener('change', () => { MOCK_loadedAfter = false; });

async function loadSimTab() { loadRosterSim(); await loadMockResults(); pollMock(); }

async function loadMockResults() {
  MOCK = await api('/api/sim/mock/results');
  const sel = $('mockPick'), cur = sel.value;
  sel.innerHTML = h`${MOCK.my_picks.map((k, i) => h`<option value="${k}">round ${i + 1} · pick ${k}</option>`)}`;
  if (cur) sel.value = cur;
  $('mockMeta').textContent = `${MOCK.drafts} drafts stored (${MOCK.local} local, ${MOCK.external} imported)`;
  renderMock();
}

function renderMock() {
  if (!MOCK) return;
  const k = $('mockPick').value || MOCK.my_picks[0];
  const pk = MOCK.per_pick[String(k)] || {rows: [], n_drafts: 0};
  const rows = pk.rows.filter(r => MOCKPOS === 'ALL' || r.pos === MOCKPOS)
    .sort((a, b) => (a.adp ?? 999) - (b.adp ?? 999)).slice(0, 18);
  if (!rows.length) { $('mockRows').innerHTML = '<tr><td colspan="8" class="dim">No drafts yet — run some, or import an external mock.</td></tr>'; return; }
  $('mockRows').innerHTML = h`${rows.map(r => {
    const d = r.avail_sim - r.avail_model, dc = Math.abs(d) < 0.1 ? 'dim' : (d > 0 ? 'green' : 'amber');
    const pc = r.avail_sim >= .7 ? 'green' : r.avail_sim >= .35 ? 'amber' : 'red';
    return h`<tr style="cursor:pointer" onclick="openCard('${r.uid}')"><td style="font-family:var(--sans)">${r.name}</td><td>${r.pos}</td><td class="r">${r.proj}</td><td class="r">${r.adp}</td><td class="r">${r.sim_adp ?? '—'}</td><td class="r ${pc}" style="font-weight:600">${Math.round(r.avail_sim * 100)}%</td><td class="r mid">${Math.round(r.avail_model * 100)}%</td><td class="r ${dc}">${fmtDelta(d * 100)}</td></tr>`;
  })}`;
  $('mockMeta').textContent = `${MOCK.drafts} drafts stored (${MOCK.local} local, ${MOCK.external} imported) · ${pk.n_drafts} reach pick ${k} · the engine blends these rates into AVAIL once 10+ drafts exist`;
}

async function runMocks() {
  MOCK_loadedAfter = false;
  const r = await api('/api/sim/mock/run', {n: parseInt($('mockN').value), policy: $('mockPolicy').value});
  if (!r.started) { $('mockStatus').textContent = 'a run is already in progress'; }
  $('mockProg').style.display = ''; pollMock();
}
async function pollMock() {
  clearTimeout(mockTimer);
  const st = await api('/api/sim/mock/status');
  if (st.running) {
    $('mockProg').style.display = ''; $('mockProg').firstElementChild.style.width = Math.round(100 * st.done / Math.max(st.total, 1)) + '%';
    $('mockStatus').textContent = `running ${st.done} / ${st.total}…`;
    mockTimer = setTimeout(pollMock, 1200);
  } else {
    if (st.total) { $('mockStatus').textContent = st.error ? 'error: ' + st.error : `done · ${st.total} drafts added`; $('mockProg').firstElementChild.style.width = '100%'; }
    if (st.total && !MOCK_loadedAfter) { MOCK_loadedAfter = true; await loadMockResults(); askHowie('mock'); }
  }
}
async function importMock() {
  const text = $('mockImport').value.trim(); if (!text) return;
  try {
    const r = await api('/api/sim/mock/import', {text, source: $('mockSource').value.trim() || 'external'});
    $('mockImportNote').textContent = `stored ${r.stored} picks · ${r.drafts} drafts total` + (r.unresolved.length ? ` · unresolved: ${r.unresolved.slice(0, 4).join(', ')}` : '');
    $('mockImport').value = ''; await loadMockResults();
  } catch (e) { $('mockImportNote').textContent = 'import failed: ' + e.message; }
}

// ---------------- shared autocomplete (every search box) ----------------

function attachAutocomplete(input, onPick, opts = {}) {
  const wrap = input.parentElement; const drop = document.createElement('div'); drop.className = 'acdrop' + (opts.up ? ' up' : ''); wrap.appendChild(drop);
  let items = [], sel = 0, timer = null, navigated = false;
  input._acItems = () => items; input._acSel = () => (navigated ? sel : -1);
  input._acClear = () => { items = []; navigated = false; render(); };
  const render = () => {
    if (!items.length) { drop.style.display = 'none'; return; }
    drop.style.display = 'block';
    drop.innerHTML = h`${items.map((r, i) => h`<div class="acitem ${i === sel ? 'sel' : ''}" data-i="${i}"><span class="kindtag ${r.kind === 'player' ? 'P' : ''}">${r.kind === 'player' ? 'P' : r.kind === 'unit' ? 'U' : 'T'}</span><span class="${r.taken ? 'dim' : ''}">${r.name}</span><span class="mono dim" style="font-size:10px">${r.position || ''} ${r.team || ''}${r.proj ? ' · ' + r.proj : ''}${r.taken ? raw(' · <span class="amber">taken</span>') : ''}${r.status ? h` · <span class="${r.status.level === 'out' ? 'red' : 'amber'}">${r.status.text}</span>` : ''}</span></div>`)}`;
  };
  const pick = (i) => { const r = items[i]; if (!r) return; input.value = r.name; items = []; render(); onPick && onPick(r); };
  drop.addEventListener('mousedown', e => { const el = e.target.closest('.acitem'); if (el) { e.preventDefault(); pick(+el.dataset.i); } });
  input.addEventListener('input', () => {
    clearTimeout(timer); const q = input.value.trim();
    if (!q || q.toLowerCase().startsWith('sql:') || q.startsWith('/') || q.startsWith('?')) { items = []; render(); return; }
    timer = setTimeout(async () => {
      const hits = await api('/api/search?q=' + encodeURIComponent(q));
      items = (input.dataset.ac === 'player' ? hits.filter(x => x.kind === 'player') : hits).slice(0, 6); sel = 0; navigated = false; render();
    }, 120);
  });
  input.addEventListener('keydown', e => {
    if (opts.onEnter && (e.key === 'Enter' || e.key === 'Tab')) {
      if (e.key === 'Tab' && !items.length) return;
      e.preventDefault(); clearTimeout(timer);
      const snapshot = items, selIdx = navigated ? sel : -1;
      items = []; navigated = false; render();
      opts.onEnter(e, snapshot, selIdx);
      return;
    }
    if (!items.length) return;
    if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, items.length - 1); navigated = true; render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); navigated = true; render(); e.preventDefault(); }
    else if (e.key === 'Enter') { pick(sel); e.preventDefault(); }
    else if (e.key === 'Escape') { items = []; render(); }
  });
  input.addEventListener('blur', () => setTimeout(() => { items = []; render(); }, 150));
}
attachAutocomplete($('simq'), () => loadSim());
attachAutocomplete($('dhl'), () => renderDist());
attachAutocomplete($('qq'), () => runQuery());
attachAutocomplete($('rPlayer'), (r) => showFacts(r.name));

// ---------------- Howie insights ----------------

let LAST_SUGG = [];
function renderHowie(title, r) {
  const box = $('findings');
  if (!r.available) { box.innerHTML = h`<h4>${title}</h4><span class="dim" style="font-size:12px">${r.reason}</span>`; }
  else {
    LAST_SUGG = r.suggestions || [];
    const sugg = LAST_SUGG.map((sg, i) => h`<div class="sugg"><span class="kindtag">${sg.type}</span><span><b>${sg.text}</b>${sg.why ? h` <span class="dim">— ${sg.why}</span>` : ''}</span><span style="flex:1"></span><button class="primary" onclick="applySuggestion(${i}, this)">APPLY</button></div>`);
    box.innerHTML = h`<h4 title="${r.model}">${title}</h4><ol style="margin:0;padding:0">${(r.learnings || []).map(l => h`<li>${l}</li>`)}</ol>${
      sugg.length ? h`<div style="margin-top:8px">${sugg}</div>` : raw('<div class="dim" style="font-size:11px;margin-top:6px">No strategy changes suggested.</div>')}`;
  }
  box.classList.add('on'); sideOpen(); box.scrollTop = 0;
  if (!$('drawer').innerHTML) { $('sideLabel').textContent = 'HOWIE'; $('sideHint').textContent = 'findings'; }
  termPrint('dim', `${title.toLowerCase()}: ${(r.learnings || []).length} findings, ${(r.suggestions || []).length} suggestions → side panel`);
}
async function applySuggestion(i, btn) {
  const sg = LAST_SUGG[i]; if (!sg) return;
  strategy = await api('/api/strategy');
  if (sg.type === 'rule') strategy.rules.push({text: sg.text, on: true});
  else strategy.notes = (strategy.notes ? strategy.notes + '\n' : '') + '— ' + sg.text;
  const saved = await api('/api/strategy', {rules: strategy.rules, notes: strategy.notes});
  (saved.conflicts || []).forEach(c => termPrint('dim', h`<span class="conflict">rule conflict:</span> ${c}`));
  btn.textContent = 'APPLIED'; btn.disabled = true;
  refresh(true);
}
function mockInsightData(k) {
  const pk = MOCK.per_pick[String(k)] || {rows: []};
  return {drafts: MOCK.drafts, pick: +k, my_picks: MOCK.my_picks,
    gaps: pk.rows.filter(r => Math.abs(r.avail_sim - r.avail_model) >= 0.08).slice(0, 14),
    usually_there: [...pk.rows].sort((a, b) => (a.adp ?? 999) - (b.adp ?? 999)).slice(0, 14)};
}
async function draftInsightData() {
  const [pick, pos] = await Promise.all([api('/api/pick?top=10'), api('/api/positions')]);
  return {state: {round: ST.round, pick: ST.next_pick_no, roster: ST.roster.filter(r => r.name), recent_log: ST.log.slice(0, 10)},
    candidates: (pick.mc ? pick.mc.rows : pick.rows).slice(0, 10), positional: pos.rows};
}
async function askHowie(kind) {
  const strat = await api('/api/strategy');
  if (kind === 'mock') {
    if (!MOCK) return;
    const k = $('mockPick').value || MOCK.my_picks[0];
    termPrint('dim', 'Howie is reading the mock results…');
    renderHowie(`HOWIE ON YOUR MOCKS · PICK ${k}`, await api('/api/lab/insights', {kind: 'mock', data: mockInsightData(k), strategy: strat}));
  } else if (kind === 'player') {
    const q = $('simq').value.trim(); if (!q) return;
    const hits = await api('/api/search?q=' + encodeURIComponent(q)); const p = hits.find(x => x.uid); if (!p) return;
    const [sim, card] = await Promise.all([api('/api/data/sim?uid=' + encodeURIComponent(p.uid)), api('/api/card?uid=' + encodeURIComponent(p.uid))]);
    const data = {player: sim.name, position: sim.position, proj: sim.proj, engine_value: sim.value, p10: sim.p10, p50: sim.p50, p90: sim.p90,
      actual_seasons: sim.actual, model: sim.model, milestone_rates: card.milestones, room: card.room, facts: card.facts, team_facts: card.team_facts, avail_next: card.avail_next, mv_vs_wait: card.mv_vs_wait};
    termPrint('dim', `Howie is thinking about ${sim.name}…`);
    renderHowie(`HOWIE ON ${sim.name.toUpperCase()}`, await api('/api/lab/insights', {kind: 'player', data, strategy: strat}));
  } else {
    termPrint('dim', 'Howie is reading the board…');
    renderHowie(`HOWIE ON PICK ${ST.next_pick_no}`, await api('/api/lab/insights', {kind: 'draft', data: await draftInsightData(), strategy: strat}));
  }
}

// ---------------- TEAM report ----------------

let TEAM = localStorage.getItem('teamTab') || 'PHI', teamListLoaded = false;
let teamSeq = 0;
async function loadTeamTab(team) {
  const seq = ++teamSeq;  // the latest request wins; an earlier load must not overwrite it
  showTab('team', {noload: true});
  if (!teamListLoaded) {
    const st = await api('/api/research/status');
    $('teamSel').innerHTML = h`${st.teams.map(t => h`<option value="${t.team}">${t.team} — ${t.name}</option>`)}`;
    $('teamSel').addEventListener('change', () => loadTeamTab($('teamSel').value));
    teamListLoaded = true;
  }
  const r = await api('/api/team?team=' + encodeURIComponent(team || TEAM));
  if (seq !== teamSeq) return;
  TEAM = r.team; localStorage.setItem('teamTab', TEAM);
  $('teamSel').value = TEAM;
  renderTeam(r);
}
function renderTeam(r) {
  const cov = r.coverage || {};
  $('teamMeta').textContent = `${r.name} · bye ${r.bye || '—'} · official depth chart ${r.depth_as_of ? r.depth_as_of.slice(0, 10) : 'not loaded (howie data refresh --steps depth)'} · board at pick ${r.current_pick}, next ${r.next_pick}`;
  const fresh = cov.latest ? `researched ${cov.latest}` : 'not researched';
  $('teamCoverage').innerHTML = h`${fresh} · ${cov.players_researched || 0} / ${cov.targets || 0} players · ${cov.facts || 0} facts<br><span class="dim">refresh: run the research-teams workflow for ${r.team}</span>`;
  const fact = f => h`<div class="factcard"><span class="kindtag">${f.kind}</span> ${f.text}${f.value != null ? h` <span class="mono green">(${f.value})</span>` : ''}<div class="fmeta">${f.source} · conf ${f.confidence}</div></div>`;
  const sosPos = Object.keys(r.playoff_sos || {});
  $('teamHead').innerHTML = h`<div class="facts"><p class="sechead">OFFENSE — COACHING · SCHEME · LINE</p>${
    r.team_facts.length || r.ol_facts.length ? h`${r.team_facts.map(fact)}${r.ol_facts.map(fact)}` : raw('<span class="dim" style="font-size:12px">No researched team facts yet.</span>')}</div>
    <div><p class="sechead">PLAYOFF SOS · W15–17 <span class="dim" style="letter-spacing:0">(higher = easier)</span></p>${
      sosPos.length ? h`${sosPos.map(pos => h`<div style="display:flex;align-items:center;gap:8px"><span class="kindtag">${pos}</span><div class="sosrow" style="flex:1">${
        r.playoff_sos[pos].map(s2 => h`<div style="background:${s2.value >= 6 ? '#12241a' : s2.value >= 4.5 ? '#171509' : '#241210'};color:${s2.value >= 6 ? 'var(--acc)' : s2.value >= 4.5 ? 'var(--amber)' : 'var(--red)'}">W${s2.week} ${s2.value}</div>`)}</div></div>`)}` : raw('<span class="dim" style="font-size:12px">no SoS</span>')}</div>`;
  $('teamRooms').innerHTML = h`${['QB', 'RB', 'WR', 'TE'].map(pos => {
    const room = r.rooms[pos] || {rows: []};
    return h`<div class="room">
      <div class="roomhead"><p class="sechead" style="margin:0">${pos} ROOM</p>${
        room.vacated != null ? h`<span class="mono amber" style="font-size:11px">${Math.round(room.vacated * 100)}% of last season's volume left this room</span>` : ''}</div>
      <table><thead><tr><th>#</th><th>PLAYER</th><th>ROLE</th><th class="r">PROJ</th><th class="r">VALUE</th><th class="r">ADP</th><th class="r">AVAIL @${r.next_pick}</th><th class="r">'25 SHARE</th><th>STATUS</th></tr></thead><tbody>${
        room.rows.map(row => h`<tr class="${row.taken ? 'taken' : ''} ${row.rank == null ? 'unlisted' : ''}" style="cursor:${row.uid ? 'pointer' : 'default'}" ${row.uid ? h`data-uid="${row.uid}" onclick="openCard(this.dataset.uid)"` : ''}>
          <td class="mono dim">${row.rank != null ? row.rank : '—'}${row.slot ? h`<span class="slotlbl">${row.slot}</span>` : ''}</td>
          <td><b style="font-weight:500">${row.name}</b>${row.board_rank ? h`<span class="boardtag">BOARD #${row.board_rank}</span>` : ''}${row.taken ? raw('<span class="slotlbl">TAKEN</span>') : ''}</td>
          <td><span class="rolechip ${row.role_disagrees ? 'disagree' : ''}" title="${row.role_disagrees ? 'research disagrees with the official depth chart' : 'from research'}">${row.role || ''}${row.role_disagrees ? ' ≠ chart' : ''}</span></td>
          <td class="mono r mid">${row.proj ?? '—'}</td><td class="mono r">${row.value ?? '—'}</td>
          <td class="mono r mid">${row.adp ? row.adp.toFixed(1) : '—'}</td>
          <td class="mono r">${row.avail_next != null ? Math.round(row.avail_next * 100) + '%' : '—'}</td>
          <td class="mono r mid">${row.share != null ? Math.round(row.share * 100) + '%' : '—'}${row.other_team ? h`<span class="amber" title="earned with ${row.other_team}">*</span>` : ''}</td>
          <td>${row.status ? h`<span class="stchip ${row.status.level}" style="margin-left:0">${row.status.text}</span>` : ''}</td>
        </tr>`)}</tbody></table>${
      room.facts.length ? h`<div style="margin-top:6px">${room.facts.map(fact)}</div>` : ''}
    </div>`;
  })}`;
}

// ---------------- research ----------------

let RSTATUS = null;
async function loadResearch() {
  RSTATUS = await api('/api/research/status');
  const sel = $('rTeam'), cur = sel.value;
  sel.innerHTML = h`${RSTATUS.teams.map(t => h`<option value="${t.team}">${t.team} — ${t.name}${t.facts ? ' (' + t.facts + ')' : ''}</option>`)}`;
  if (cur) sel.value = cur;
  const covered = RSTATUS.teams.filter(t => t.facts);
  $('rCoverage').innerHTML = h`${covered.length} / 32 teams researched${covered.length ? h`<br>${covered.map(t => h`<span class="mono">${t.team}</span> ${t.facts} facts · ${(t.latest || '').slice(0, 10)}<br>`)}` : ''}`;
  showFacts(sel.value);
}
$('rTeam').addEventListener('change', () => showFacts($('rTeam').value));
async function showFacts(q) {
  const r = await api('/api/research/facts?q=' + encodeURIComponent(q));
  if (!r.facts.length) { $('rFacts').innerHTML = h`<span class="dim" style="font-size:12px">No researched facts for ${r.entity ? r.entity.name : q} yet.</span>`; return; }
  $('rFacts').innerHTML = h`<div class="mid" style="font-size:12px;margin-bottom:6px"><b>${r.entity.name}</b> · ${r.facts.length} facts</div>${
    r.facts.map(f => h`<div class="factcard"><span class="kindtag">${f.kind}</span> <span class="mono dim" style="font-size:10px">${f.entity_id.replace(/^(player|team|unit):/, '')}</span><br>${f.text}${f.value != null ? h` <span class="mono green">(${f.value})</span>` : ''}<div class="fmeta">${f.source} · conf ${f.confidence} · ${(f.created || '').slice(0, 10)}${f.expires ? ' · expires ' + f.expires : ''}</div></div>`)}`;
}

// ---------------- panels + terminal (the Howie command line) ----------------

function chatOpen() { $('chat').classList.remove('collapsed'); localStorage.setItem('chatOpen', '1'); }
function toggleChat() { const c = $('chat'); c.classList.toggle('collapsed'); localStorage.setItem('chatOpen', c.classList.contains('collapsed') ? '0' : '1'); }
function sideOpen() { $('side').classList.remove('collapsed'); }
function clearChat() { $('termOut').innerHTML = ''; localStorage.removeItem('howieChat'); }
// termPrint escapes strings (newlines become <br>); pass an h`` template for markup.
function termPrint(kind, content) {
  chatOpen();
  const out = $('termOut');
  const el = document.createElement('div'); el.className = 'tl ' + kind;
  el.innerHTML = content instanceof Raw ? content.s : textHtml(content).s;
  out.appendChild(el);
  while (out.children.length > 300) out.removeChild(out.firstChild);
  out.scrollTop = out.scrollHeight;
  try { localStorage.setItem('howieChat', out.innerHTML); } catch (e) {}
}
function openTerminal() { chatOpen(); $('termIn').focus(); if (!$('termOut').children.length) termPrint('dim', 'Howie. Type a player, /help for commands, or ?question to ask.'); }
(function restoreChat() {
  const saved = localStorage.getItem('howieChat'); if (saved) { $('termOut').innerHTML = saved; $('termOut').scrollTop = 1e9; }
  if (localStorage.getItem('chatOpen') === '1') chatOpen();
})();
function renderTermHint() {
  $('termHint').textContent = drafting() ? '⏎ taken · ⇧⏎ mine · ⇥ card · team → report · ?ask · /help' : '⏎ card · ⇧⏎ draft to me · team → report · ?ask · /help';
  $('termIn').placeholder = drafting() ? 'player → ⏎ marks taken, ⇧⏎ drafts to you · /undo · ?ask Howie' : 'player name → card · /mine · /taken · /mock · ?ask Howie · /help';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeCard(); closePops(); } if (e.key === '`' && (e.ctrlKey || e.metaKey)) { openTerminal(); e.preventDefault(); } });

attachAutocomplete($('termIn'), null, {up: true, onEnter: (e, items, selIdx) => {
  const line = $('termIn').value.trim(); if (!line) return;
  $('termIn').value = '';
  handleEntry(line, items, selIdx, e);
}});

async function findPlayer(q) { const hits = await api('/api/search?q=' + encodeURIComponent(q)); return hits.find(x => x.uid); }

// One keystroke per pick: the classification decides, the modifier decides the action.
async function handleEntry(line, items, selIdx, ev) {
  let cls = classifyInput(line, items, selIdx);
  if ((cls.kind === 'nomatch' || (cls.kind === 'ask' && !/^[?]|^(hey\s+)?howie/i.test(line))) && !(items && items.length)) {
    // Enter landed before the dropdown fetch came back — look it up now
    try { cls = classifyInput(line, await api('/api/search?q=' + encodeURIComponent(line)), -1); } catch (e) {}
  }
  if (cls.kind === 'player') {
    let action = ev.key === 'Tab' ? 'card' : pickAction(ev, drafting());
    if (cls.hit.taken && action !== 'card') {
      termPrint('cmd', '› ' + cls.hit.name);
      termPrint('dim', `${cls.hit.name} is already off the board — opening the card`);
      action = 'card';
    } else termPrint('cmd', '› ' + cls.hit.name + (action === 'card' ? '' : action === 'mine' ? '  ⇧⏎ mine' : '  ⏎ taken'));
    if (action === 'card') return openCard(cls.hit.uid);
    try { await mark(cls.hit.uid, action === 'mine'); } catch (err) { termPrint('dim', 'could not record the pick: ' + err.message); }
    return;
  }
  if (cls.kind === 'team') {
    termPrint('cmd', '› ' + (cls.hit ? cls.hit.name : cls.team) + '  → team report');
    return loadTeamTab(cls.team).catch(e => termPrint('dim', e.message));
  }
  if (cls.kind === 'nomatch' && /^[a-z .'-]{3,}$/i.test(line) && line.split(/\s+/).length <= 3) {
    // nickname or partial team name ("philly", "niners"): the server resolves those
    try { await loadTeamTab(line); termPrint('cmd', '› ' + line + '  → team report'); return; } catch (e) {}
  }
  if (cls.kind === 'nomatch') {
    termPrint('cmd', '› ' + line);
    termPrint('dim', `no player or team matches "${line}"` + (cls.suggestions.length ? ` — did you mean ${cls.suggestions.join(', ')}?` : '') + ' · prefix with ? to ask Howie');
    return;
  }
  return handleTerm(line, items, cls);
}

async function handleTerm(line, acItems, cls) {
  cls = cls || classifyInput(line, acItems, -1);
  termPrint('cmd', '› ' + line);
  if (cls.kind === 'ask' || cls.kind === 'nomatch') {
    const question = cls.question || line;
    termPrint('dim', 'Howie is thinking…');
    try {
      const r = await api('/api/ask', {question});
      (r.tools || []).forEach(t => termPrint('dim', '→ ' + t));
      termPrint('howie', r.answer || (r.notes || []).join(' ') || 'no answer');
    } catch (err) { termPrint('dim', 'Howie unavailable: ' + err.message); }
    return;
  }
  if (cls.kind === 'player') return openCard(cls.hit.uid);
  if (cls.kind === 'team') return loadTeamTab(cls.team);
  if (cls.kind !== 'cmd') return;
  const {cmd, rest, arg} = cls;
  try {
    if (cmd === 'help') {
      termPrint('out', ['NAME → while drafting ⏎ marks taken, ⇧⏎ drafts to you, ⇥ opens the card (⏎ = card before the draft starts)',
        '/card NAME · open a player card', '/team ABBR · team report (depth chart, status, facts)', '/mine NAME · draft to me', '/taken NAME · mark taken', '/undo', '/board · top picks now',
        '/read · Howie reads the board', '/sim NAME · simulate a season', '/mock N [howie|adp] · run mock drafts', '/research TEAM|NAME · deep research',
        '/sql SELECT … · read-only query', '/data Q · look up a player/team/room', '/strategy · show rules & notes', '/ask Q or ?Q · ask Howie'].join('\n'));
    } else if (cmd === 'ask') { return handleTerm('?' + arg, acItems); }
    else if (cmd === 'team') { await loadTeamTab(arg || TEAM); }
    else if (cmd === 'card') { const p = await findPlayer(arg); p ? openCard(p.uid) : termPrint('dim', 'no player found'); }
    else if (cmd === 'mine' || cmd === 'taken') { const p = await findPlayer(arg); p ? await mark(p.uid, cmd === 'mine') : termPrint('dim', 'no player found'); }
    else if (cmd === 'undo') { await undoPick(); }
    else if (cmd === 'board') { const pk = await api('/api/pick?top=5'); const rows = (pk.mc ? pk.mc.rows : pk.rows).slice(0, 5); termPrint('out', rows.map((r, i) => `${i + 1}. ${r.name} (${r.pos}) ${r.value} ${r.delta ? fmtDelta(r.delta) : ''} · avail@${pk.next_pick} ${Math.round(r.avail_next * 100)}%`).join('\n')); }
    else if (cmd === 'read') { await askHowie('draft'); }
    else if (cmd === 'sim') { const p = await findPlayer(arg); if (!p) { termPrint('dim', 'no player found'); return; } const sr = await api('/api/data/sim?uid=' + encodeURIComponent(p.uid)); termPrint('out', `${sr.name}: proj ${sr.proj} · simulated p10 ${sr.p10} / p50 ${sr.p50} / p90 ${sr.p90}` + (sr.actual.length ? ' · actual ' + sr.actual.map(a => `${a.season}: ${a.total}`).join(', ') : '')); $('simq').value = sr.name; }
    else if (cmd === 'mock') {
      const n = parseInt(rest[0]) || 25, pol = rest[1] || 'adp';
      const r = await api('/api/sim/mock/run', {n, policy: pol});
      termPrint('out', r.started ? `running ${n} mock drafts (${pol})… results land in LAB` : 'a run is already in progress');
      MOCK_loadedAfter = false;
      const wait = async () => {
        const st = await api('/api/sim/mock/status'); if (st.running) { setTimeout(wait, 1500); return; }
        await loadMockResults(); termPrint('out', `done · ${MOCK.drafts} drafts stored`); termPrint('dim', 'Howie is reading the results…');
        const k = MOCK.my_picks[Math.min(ST.round - 1, MOCK.my_picks.length - 1)];
        const strat = await api('/api/strategy');
        renderHowie(`HOWIE ON YOUR MOCKS · PICK ${k}`, await api('/api/lab/insights', {kind: 'mock', strategy: strat, data: mockInsightData(k)}));
      };
      wait();
    }
    else if (cmd === 'research') {
      const quick = /^quick\b/i.test(arg); const target = arg.replace(/^quick\s*/i, '').trim();
      if (!quick) { termPrint('out', `Deep research runs in Claude Code as a subagent workflow (research → fact-check → import):\n  run the research-teams workflow for ${target || 'PHI DAL'}\nValidated facts appear on cards and in DATA › RESEARCH. For a quick single-shot instead: /research quick ${target || 'PHI'}`); return; }
      termPrint('dim', `quick research on ${target}… (30–90s, web search)`);
      const isTeam = /^[A-Z]{2,3}$/i.test(target);
      const r = await api('/api/research/run', isTeam ? {team: target.toUpperCase()} : {player: target});
      if (!r.available) { termPrint('dim', r.reason); return; }
      termPrint('out', `imported ${r.imported} facts` + (r.skipped && r.skipped.length ? ` · ${r.skipped.length} skipped` : ''));
      (r.facts || []).slice(0, 4).forEach(f => termPrint('out', `• [${f.kind}] ${f.text}`));
    }
    else if (cmd === 'sql') { const r = await api('/api/data/query?q=' + encodeURIComponent('sql: ' + arg)); if (r.error) { termPrint('dim', r.error); return; } termPrint('out', r.rows.slice(0, 12).map(row => r.columns.map(c => `${c}=${row[c]}`).join('  ')).join('\n') || 'no rows'); }
    else if (cmd === 'data') { const r = await api('/api/data/query?q=' + encodeURIComponent(arg)); if (!r.entity) { termPrint('dim', 'nothing found'); return; } const d = r.detail; termPrint('out', `${r.entity.kind}: ${r.entity.name}` + (d.projection ? ` · 2026 proj ${d.projection.proj} · ADP ${d.projection.adp ?? '—'}` : '') + (d.seasons ? '\n' + d.seasons.map(x => `${x.season}: ${x.pts} pts in ${x.g} g (${x.ppg}/g)`).join('\n') : '') + (d.rooms ? '\n' + d.rooms.filter(m => m.proj).slice(0, 10).map(m => `${m.position} ${m.name} ${m.proj}`).join('\n') : '')); }
    else if (cmd === 'strategy') { const st = await api('/api/strategy'); termPrint('out', (st.rules.map(r => `${r.on ? '●' : '○'} ${r.text}${r.inert ? ' (inert)' : ''}`).join('\n') || 'no rules') + (st.notes ? '\n' + st.notes : '')); }
    else termPrint('dim', 'unknown command — /help');
  } catch (err) { termPrint('dim', 'error: ' + err.message); }
}

// ---------------- header popovers: reset / config ----------------

function closePops() { document.querySelectorAll('.pop').forEach(p => p.classList.remove('open')); }
async function togglePop(id) {
  const was = $(id).classList.contains('open'); closePops();
  if (!was) { if (id === 'popConfig') await loadConfig(); $(id).classList.add('open'); }
}
document.addEventListener('mousedown', e => { if (!e.target.closest('.pop') && !e.target.closest('.icon')) closePops(); });
async function loadConfig() {
  const c = await api('/api/config');
  for (const k of Object.keys(c)) { const el = $('cfg_' + k); if (el) el.value = c[k]; }
  $('cfgNote').textContent = '';
}
async function saveConfig() {
  const body = {};
  document.querySelectorAll('#popConfig [id^="cfg_"]').forEach(el => { body[el.id.slice(4)] = el.value; });
  try { await api('/api/config', body); $('cfgNote').textContent = 'saved'; await refresh(true); setTimeout(closePops, 500); }
  catch (e) { $('cfgNote').textContent = e.message; }
}

// ---------------- wordmark: typed on load ----------------
(function typeWordmark() {
  const el = $('wordmark'), text = 'HOWIE'; let i = 0;
  const step = () => { el.innerHTML = text.slice(0, i) + '<span class="cur">_</span>'; if (i < text.length) { i++; setTimeout(step, 110); } };
  setTimeout(step, 200);
})();

// ---------------- boot + poll ----------------

refresh();
setInterval(() => { if (!document.hidden) refresh(true); }, 2500);
