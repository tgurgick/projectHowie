export const meta = {
  name: 'research-teams',
  description: 'Deep-research NFL teams with subagents, validate, condense, import into Howie\'s knowledge graph',
  whenToUse: 'Before the draft or when news breaks: "research PHI DAL", "run the research-teams workflow for all teams"',
  phases: [
    { title: 'Research', detail: 'one subagent per team, web research → facts JSON' },
    { title: 'Validate', detail: 'skeptic checks each fact against its source and the contract' },
    { title: 'Import', detail: 'condense, write data/research/<TEAM>.json, howie graph import' },
  ],
}

// args: ["PHI", "DAL"] or "all"
const ALL = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC",
             "LA","LAC","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"]
const teams = (args === 'all' || !args) ? ALL : (Array.isArray(args) ? args : String(args).split(/[\s,]+/)).map(t => t.toUpperCase()).filter(Boolean)
log(`researching ${teams.length} team(s): ${teams.join(' ')}`)

const FACTS = {
  type: 'object',
  properties: {
    facts: { type: 'array', items: { type: 'object', properties: {
      entity: { type: 'string' }, kind: { type: 'string' }, text: { type: 'string' },
      value: { type: ['number', 'null'] }, confidence: { type: 'number' }, source: { type: 'string' }, expires: { type: 'string' },
    }, required: ['entity', 'kind', 'text', 'confidence', 'source'] } },
  },
  required: ['facts'],
}
const VERDICTS = {
  type: 'object',
  properties: { kept: { type: 'array', items: { type: 'object', properties: {
    entity: { type: 'string' }, kind: { type: 'string' }, text: { type: 'string' },
    value: { type: ['number', 'null'] }, confidence: { type: 'number' }, source: { type: 'string' }, expires: { type: 'string' },
  }, required: ['entity', 'kind', 'text', 'confidence', 'source'] } }, dropped: { type: 'array', items: { type: 'string' } } },
  required: ['kept', 'dropped'],
}

const researchPrompt = (team) => `Read skills/research-team.md in this repo, then research the ${team} offense for the 2026 season using web search (beat reporters, team sites, injury reports, depth charts). Cover: coaching/scheme changes and what the coordinator's past offenses looked like; who left/arrived and who absorbs vacated targets and carries; injuries and camp battles; offensive line quality. Return ONLY the facts contract: 6-12 facts, one claim each, entity refs team:${team} / unit:${team}-QB|RB|WR|TE|OL / player:<Full Name exactly as rostered>, kinds coach_change|scheme_note|role_note|injury_note|oline_grade|volume_prior, numeric value where the claim is quantitative, honest confidence, dated source, expires (draft week + 2 weeks).`

const validatePrompt = (team, facts) => `You are a skeptical fact checker for ${team}. For EACH fact below, verify with a quick web search that the claim is current for the 2026 season and attributable to a real source; drop anything stale (last-season news presented as current), unsourced, duplicated, or malformed per the contract (entity must be team:ABBR, unit:ABBR-POS, or player:<Full Name>). Fix obviously wrong team abbreviations. Return kept facts (unchanged or lightly corrected) and a list of dropped facts with reasons.\n\nFACTS:\n${JSON.stringify(facts)}`

const results = await pipeline(
  teams,
  (team) => agent(researchPrompt(team), { label: `research:${team}`, phase: 'Research', schema: FACTS }),
  (found, team) => found && found.facts.length
    ? agent(validatePrompt(team, found.facts), { label: `validate:${team}`, phase: 'Validate', schema: VERDICTS })
    : { kept: [], dropped: ['no facts found'] },
  (verdict, team) => verdict && verdict.kept.length
    ? agent(`Write this JSON to data/research/${team}.json (create the directory if needed) with top-level {"season": 2026, "facts": [...]}, then run \`python3 -m howie3 graph import data/research/${team}.json\` and report the imported count and any unresolved names verbatim.\n\n${JSON.stringify({ season: 2026, facts: verdict.kept })}`,
            { label: `import:${team}`, phase: 'Import', effort: 'low' })
        .then(out => ({ team, kept: verdict.kept.length, dropped: verdict.dropped, import: String(out).slice(0, 300) }))
    : { team, kept: 0, dropped: (verdict && verdict.dropped) || [], import: 'skipped' },
)
const summary = results.filter(Boolean)
log(`done: ${summary.map(r => `${r.team} ${r.kept} kept`).join(', ')}`)
return summary
