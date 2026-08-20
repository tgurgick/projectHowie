# Howie Improvement Plan

## Purpose

This document turns the current code review into an implementation roadmap for
Howie v3.

The project has two distinct data concerns:

1. **Local data construction:** users may independently scrape, download, or
   import the source data needed to build their own local database.
2. **Portable strategy context:** Howie may use derived, source-independent
   abstractions—such as simulation summaries and availability models—to power
   the tool without distributing the underlying scraped data.

Raw scraped datasets, source exports, databases containing those datasets, and
provider-specific row-level payloads must not be committed to Git or bundled in
releases. This is both an ownership constraint and a reproducibility boundary.

## Current state

The v3 foundation is substantially in place:

- single SQLite schema and migration layer;
- shared CLI/TUI command registry;
- marginal-value board and roster-conditional pick valuation;
- empirical outcome distributions and Monte Carlo season simulation;
- native Anthropic tool-calling agent;
- 18 tests currently passing with `python3 -m pytest -q`.

The project is not yet ready for a clean-clone release because the source
package under `howie3/data/` is still ignored by Git, local/raw data handling is
not clearly separated from portable artifacts, and several correctness and
safety boundaries need to be hardened.

## Design model: two data planes

### Local data plane

This remains on the user's machine:

- scraped PFF, FantasyPros, FFC, or other provider exports;
- nflverse downloads and historical weekly data;
- the generated `data/howie.db`;
- provider-specific identifiers and source row payloads;
- local intelligence reports that contain source text or citations.

The repository should contain source adapters and schemas, but not their raw
outputs.

Recommended local layout:

```text
~/.howie/
  config/
    league.json
  raw/
    pff/
    adp/
    nflverse/
  db/
    howie.db
  cache/
  strategy-context/
```

Support `HOWIE_DATA_DIR` as an override. The CLI should explain how to build
the local database and should fail with an actionable message when a source
export is missing.

### Portable strategy-context plane

This is the only data intended for upload, sharing, or optional service use.
It must contain derived abstractions rather than scraped source rows.

Examples:

- league shape, scoring format, draft position, and roster rules;
- canonical player identifier plus position/team labels;
- projection bands, ranks, tiers, and normalized value scores;
- availability probabilities at relevant picks;
- outcome summaries such as mean, standard deviation, p10, p50, p90;
- simulation configuration, seed, run metadata, and aggregate results;
- recommended strategy branches and sensitivity summaries;
- model/schema version and provenance class, without copying provider payloads.

Do not include:

- raw CSV/HTML/JSON/parquet source files;
- full weekly stat tables;
- provider scrape responses;
- copied source text or large intelligence reports;
- full per-player simulation samples unless explicitly approved as a separate
  product decision.

The default portable artifact should be an aggregate JSON document such as:

```json
{
  "schema_version": 1,
  "artifact_type": "strategy_context",
  "created_at": "2026-08-18T00:00:00Z",
  "league": {
    "teams": 12,
    "draft_position": 8,
    "scoring": "half",
    "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
  },
  "players": [
    {
      "uid": "local-or-canonical-id",
      "position": "RB",
      "tier": 0,
      "projection_band": "elite",
      "availability": {"8": 0.23, "17": 0.00},
      "outcomes": {"mean": 284, "std": 52, "p10": 220, "p50": 286, "p90": 348}
    }
  ],
  "simulation": {
    "runs": 10000,
    "seed": 7,
    "summary": {"best_strategy_mean": 2810, "best_strategy_p10": 2500}
  }
}
```

The format should be versioned, validated, and documented as an abstraction
contract. It should be possible to regenerate it from any compatible local
database without requiring a particular provider.

## Prioritized implementation roadmap

### P0 — repository and data-boundary fixes

#### 1. Make source code trackable without tracking raw data

The broad `data/` ignore pattern currently hides `howie3/data/`, while broad
`*.json` rules also hide local configuration such as `data/league_config.json`.

Implement:

- change the raw-data ignore rule to target the repository-level `/data/`;
- keep raw exports, databases, and caches ignored;
- ensure `howie3/data/` and its source adapters are tracked;
- move the default league configuration into a tracked, non-user-specific
  location such as `howie3/defaults/league.json`, or provide it through an
  explicit example file;
- add `data/.gitkeep` only if an empty local data directory is needed;
- add a repository check that fails if tracked files match raw-data patterns.

Acceptance criteria:

- a fresh clone contains all v3 Python source;
- a fresh clone contains no scraped CSV, database, or source response;
- `pip install .` includes the v3 source adapters and schema;
- `howie3 data status` gives a clear “build local data first” message.

Relevant files:

- `.gitignore`
- `setup.py`
- `howie3/config.py`
- `howie3/README.md`

#### 2. Define and implement the strategy-context artifact

Create a small module, for example `howie3/context_artifact.py`, with:

- typed models for league context, player abstractions, simulation summaries,
  and strategy recommendations;
- JSON serialization and schema versioning;
- validation on read and write;
- explicit redaction of raw provider fields;
- stable handling for local-only player IDs;
- import/export commands such as:
  - `howie context export --out strategy-context.json`
  - `howie context inspect strategy-context.json`
  - `howie context import strategy-context.json`.

The export path must operate from derived tables or in-memory simulation
results. It must not copy arbitrary database rows.

Acceptance criteria:

- exported artifacts contain only approved fields;
- an artifact can power draft-board and pick-ranking views without the raw
  database;
- schema incompatibilities produce a clear error;
- tests prove that raw columns and source payloads cannot pass through export.

#### 3. Harden refresh behavior

Update `howie3/data/refresh.py` and the CLI so that:

- unknown step names are rejected;
- dependency ordering is enforced;
- failed steps produce a nonzero process exit code;
- a partial refresh is clearly marked and never presented as complete;
- each source reports row counts, skipped rows, unmatched names, and source
  timestamp;
- refresh writes only to the user's local data directory.

### P1 — correctness and security

#### 4. Fix identity and referential-integrity edge cases

Address the crosswalk fallback that can create `mfl:nan` when both GSIS and MFL
IDs are missing. Then add foreign keys or post-ingest integrity checks for
`weekly_stats`, `projections`, and `adp`.

Relevant files:

- `howie3/data/sources/dynastyprocess.py`
- `howie3/schema.sql`
- `howie3/data/names.py`

Required tests:

- missing external IDs;
- duplicate names with different teams/positions;
- unresolved names recorded in `unmatched_names`;
- no dangling player references after refresh.

#### 5. Recalibrate simulation expectations

The simulator currently divides a projection by projected games and then applies
an independent play-probability model. This can make the expected simulated
season total diverge from the source projection.

Choose and document one calibration rule. The preferred invariant is:

```text
expected simulated season points ~= input projection
```

before SoS and intentional roster effects are applied.

Add tests for:

- projection preservation;
- bye-week handling;
- injury/availability changes;
- bench-insurance value;
- reproducibility for a fixed seed;
- sensitivity to SoS adjustments.

Relevant files:

- `howie3/value/distributions.py`
- `howie3/value/simulate.py`
- `tests/test_howie3.py`

#### 6. Secure all tool boundaries

The v3 agent should remain read-only by default. For database queries:

- accept only a single read-only statement;
- reject comments, multiple statements, pragmas, writes, and dangerous
  functions;
- apply a row limit and query timeout;
- use SQLite authorizer/progress-handler controls where practical;
- avoid returning raw source fields through agent results.

The legacy workspace tools also need path confinement, because they remain
available through `howie-legacy` and `howie-cli`:

- reject absolute paths and `..` traversal;
- resolve paths and enforce containment under the session workspace;
- honor the `overwrite` flag;
- require confirmation before writes;
- replace pickle session files with a safe serialized format or explicitly
  reject untrusted session files.

Relevant files:

- `howie3/agent.py`
- `howie_cli/core/workspace.py`
- `howie_cli/tools/file_tools.py`
- `howie_cli/core/context.py`

#### 7. Validate configuration instead of silently defaulting

Add validation for team count, draft position, roster size, position slots,
and scoring type. Unknown scoring types must fail rather than silently become
half-PPR.

Relevant file:

- `howie3/config.py`

### P2 — architecture and product quality

#### 8. Complete the legacy cutover

Keep v2 available temporarily, but make the boundary explicit:

- v3 is the only supported implementation;
- legacy commands are labeled deprecated;
- legacy database paths are not used by v3;
- shared path/configuration utilities are removed from duplicated code;
- deletion of `howie_enhanced.py`, `howie_cli/`, and old database schemas is a
  tracked milestone after user sign-off.

#### 9. Separate optional dependencies

The current requirements install AI providers, plotting, ML, scraping, and
spreadsheet libraries together. Split these into extras such as:

- `core`;
- `data`;
- `ai`;
- `visualization`;
- `dev`.

Pin or lock tested dependency versions and add a supported Python-version CI
matrix.

#### 10. Improve observability and reproducibility

Add:

- structured refresh logs;
- source adapter version and fetch timestamp;
- local database schema version in status output;
- strategy-context artifact version in every recommendation;
- deterministic simulation seed display;
- clear distinction between source-derived facts and model-generated advice.

#### 11. Align documentation with the actual product

Update the root README and v3 README to explain:

- no raw scraped data is distributed;
- how a user builds the local database independently;
- which commands require local data;
- how to export/import strategy context;
- what is stored locally versus what may be uploaded;
- v2 deprecation and the v3 entry points.

Remove references to commands or test scripts that do not exist.

## Recommended implementation order

1. Narrow `.gitignore` and verify the clean-clone package boundary.
2. Add the strategy-context schema, serializer, redaction tests, and CLI.
3. Harden refresh validation and exit codes.
4. Fix identity and simulation invariants.
5. Secure the v3 agent and legacy file/session tools.
6. Add CI, dependency extras, and clean-clone integration tests.
7. Deprecate and eventually remove v2 after sign-off.

## Definition of done

Howie is ready for the next release when:

- a clean clone can install and run v3 without bundled scraped data;
- users can independently build a local database from documented adapters;
- raw scraped data is absent from Git, packages, logs, and portable artifacts;
- strategy-context export contains only approved derived abstractions;
- draft recommendations can run from either the local database or an imported
  strategy-context artifact;
- refresh failures are machine-detectable;
- configuration, identity, simulation, and security tests pass;
- v2 is clearly isolated or removed.
