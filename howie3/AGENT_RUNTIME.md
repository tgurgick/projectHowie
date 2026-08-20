# Agent runtime: before and after

This document describes the v3 agent boundary. Raw scraped source data stays
local and ignored by Git. Users can construct that data plane independently;
the portable strategy artifact contains only derived fields and simulation
summaries needed by the tool.

## Before

The original agent was a synchronous generator around `messages.create`:

- It used a fixed eight-iteration loop and inferred continuation from the
  presence of `tool_use` blocks.
- It did not inspect `stop_reason`, enforce a time budget, retry transient
  provider failures, or distinguish a repeated call from useful progress.
- Tool calls ran serially and every tool result was returned as an untyped,
  arbitrarily clipped string.
- The CLI and TUI consumed display strings directly, so orchestration state was
  difficult to test or observe.
- The model default was resolved at import time, and tool schemas were terse
  enough that the prompt carried much of the contract.

## After

The runtime now has an explicit control plane in `howie3/agent.py`:

- `run_agent_async` is the primary entry point. It emits typed events for start,
  text, tool calls, tool results, retries, stops, errors, and completion.
- `AgentRunConfig` makes the safety envelope visible: max turns, total tool
  calls, repeated-call threshold, result size, provider retries, deadline, and
  recursion depth.
- The loop uses the provider's `stop_reason`; `end_turn`, `tool_use`,
  `max_tokens`, `pause_turn`, and unexpected stops are handled explicitly.
- Independent read-only tools run concurrently, while results preserve the
  model's requested order.
- Tool failures are returned as `is_error` tool results so the model can adapt;
  provider errors are retried only when their status indicates a transient
  failure.
- The sync `run_agent_events` bridge keeps Click and the TUI responsive and
  streams structured events. `run_agent` remains as a text compatibility
  wrapper for older integrations.
- Model selection is resolved at invocation time through `HOWIE_MODEL`, with a
  stable explicit baseline instead of an import-time alias.
- Tool descriptions now state when to use each tool, what it returns, its
  limits, and valid examples. The tools continue to operate on the local data
  plane or derived strategy context; no scraped raw data is uploaded by this
  runtime.

## Next domain layer

The control plane is now ready for fantasy-football-specific improvements:

1. Add a typed strategy-context adapter that exposes simulations, replacement
   levels, roster constraints, and uncertainty as compact semantic records.
2. Add evaluation fixtures for draft questions where the answer is a ranked
   decision plus rationale, not just a text match.
3. Add human approval events before any future state-changing action, such as
   saving a draft plan or publishing a league artifact.
