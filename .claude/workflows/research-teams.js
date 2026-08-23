export const meta = {
  name: 'research-teams',
  description: 'Deep-research NFL teams with subagents — every draft-relevant player\'s status (injury, suspension, cut risk, role) plus team facts — validate, import into Howie',
  whenToUse: 'Before the draft, weekly in-season, or when news breaks: "research PHI DAL", "run the research-teams workflow for all", "research stale teams"',
  phases: [
    { title: 'Research', detail: 'one subagent per team, web research → facts JSON' },
    { title: 'Validate', detail: 'skeptic checks each fact against its source and the contract' },
    { title: 'Import', detail: 'condense, write data/research/<TEAM>.json, howie graph import' },
  ],
}

// args: ["PHI", "DAL"], "all", or "stale" (teams with missing/old research per `howie research stale`)
const ALL = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC",
             "LA","LAC","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"]
let teams
if (args === 'stale') {
  const STALE = { type: 'object', properties: { teams: { type: 'array', items: { type: 'string' } } }, required: ['teams'] }
  const found = await agent('Run `python3 -m howie3 research stale --days 7` in this repo and return the team abbreviations it prints (empty list if it prints "nothing stale").', { label: 'stale', schema: STALE, effort: 'low' })
  teams = (found && found.teams) || []
} else {
  teams = (args === 'all' || !args) ? ALL : (Array.isArray(args) ? args : String(args).split(/[\s,]+/)).map(t => t.toUpperCase()).filter(Boolean)
}
log(`researching ${teams.length} team(s): ${teams.join(' ') || '(none)'}`)

const FACT = { type: 'object', properties: {
  entity: { type: 'string' }, kind: { type: 'string' }, text: { type: 'string' },
  value: { type: ['number', 'null'] }, confidence: { type: 'number' }, source: { type: 'string' }, expires: { type: 'string' },
}, required: ['entity', 'kind', 'text', 'confidence', 'source'] }
const PLAYER = { type: 'object', properties: {
  name: { type: 'string' },
  status: { type: 'string', enum: ['active', 'questionable', 'injured', 'out_season', 'suspended', 'holdout', 'cut_risk', 'released', 'retired'] },
  games_out: { type: 'integer', minimum: 0, maximum: 17 }, injury: { type: ['string', 'null'] },
  role: { type: 'string', enum: ['starter', 'committee', 'backup', 'depth', 'unknown'] },
  cut_risk: { type: 'number', minimum: 0, maximum: 1 }, note: { type: 'string' },
  confidence: { type: 'number' }, source: { type: 'string' },
}, required: ['name', 'status', 'role', 'confidence', 'source'] }
const FACTS = {
  type: 'object',
  properties: { players: { type: 'array', items: PLAYER }, facts: { type: 'array', items: FACT } },
  required: ['players', 'facts'],
}
const VERDICTS = {
  type: 'object',
  properties: { players: { type: 'array', items: PLAYER }, kept: { type: 'array', items: FACT }, dropped: { type: 'array', items: { type: 'string' } } },
  required: ['players', 'kept', 'dropped'],
}

const researchPrompt = (team) => `Read skills/research-team.md in this repo and follow it for ${team}. First run \`python3 -m howie3 research targets ${team} --json\` — that list is your checklist: return ONE status record for EVERY player on it (is he hurt now and for how many games, suspended, holdout, cut risk, role; healthy starters are status "active" with a short note), using web search on beat reporters, team sites, injury reports and depth charts for the 2026 season. Then 6-12 team/unit facts (coaching/scheme change, volume redistribution, offensive line) per the contract: entity refs team:${team} / unit:${team}-QB|RB|WR|TE|OL / player:<Full Name exactly as rostered>, kinds coach_change|scheme_note|role_note|injury_note|oline_grade|volume_prior, numeric value where quantitative, honest confidence, dated source, expires (draft week + 2 weeks). Names exactly as on the target list.`

const validatePrompt = (team, found) => `You are a skeptical fact checker for ${team}. Run \`python3 -m howie3 research targets ${team} --json\` and make sure EVERY target has exactly one player record (add a record with status "active", role "unknown", confidence 0.3 and source "unverified" for any the researcher skipped). For each player record and each fact, verify with a quick web search that the claim is current for the 2026 season and attributable to a real source; an injury or suspension claim must be dated within the last 60 days or it is stale. Drop facts that are stale, unsourced, duplicated, or malformed (entity must be team:ABBR, unit:ABBR-POS, or player:<Full Name>); downgrade a player record you cannot verify to confidence <= 0.4 rather than inventing. Fix wrong team abbreviations. Return players (complete), kept facts, and dropped facts with reasons.\n\nRESEARCH:\n${JSON.stringify(found)}`

const results = await pipeline(
  teams,
  (team) => agent(researchPrompt(team), { label: `research:${team}`, phase: 'Research', schema: FACTS }),
  (found, team) => found && (found.facts.length || found.players.length)
    ? agent(validatePrompt(team, found), { label: `validate:${team}`, phase: 'Validate', schema: VERDICTS })
    : { players: [], kept: [], dropped: ['no research returned'] },
  (verdict, team) => verdict && (verdict.kept.length || verdict.players.length)
    ? agent(`Write this JSON to data/research/${team}.json (create the directory if needed) exactly as given, then run \`python3 -m howie3 graph import data/research/${team}.json\`. If the import fails on an unresolvable player name, fix the name in the file (check \`python3 -m howie3 research targets ${team}\`) and re-run once. Report the imported count and any remaining unresolved names verbatim.\n\n${JSON.stringify({ season: 2026, as_of: args && args.as_of ? args.as_of : undefined, players: verdict.players, facts: verdict.kept })}`,
            { label: `import:${team}`, phase: 'Import', effort: 'low' })
        .then(out => ({ team, players: verdict.players.length, kept: verdict.kept.length, dropped: verdict.dropped, import: String(out).slice(0, 300) }))
    : { team, players: 0, kept: 0, dropped: (verdict && verdict.dropped) || [], import: 'skipped' },
)
const summary = results.filter(Boolean)
log(`done: ${summary.map(r => `${r.team} ${r.players} players / ${r.kept} facts`).join(', ')}`)
return summary
