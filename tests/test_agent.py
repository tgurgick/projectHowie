"""Deterministic tests for the agent control plane (no provider calls)."""

import asyncio
from dataclasses import dataclass
from typing import Any, List

from howie3.agent import AgentEventType, AgentRunConfig, run_agent_async
from howie3.config import Settings


@dataclass
class FakeResponse:
    content: list
    stop_reason: str


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


TOOLS = [{
    "name": "echo",
    "description": "Return the supplied value.",
    "input_schema": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    },
}]


def run(coro):
    return asyncio.run(coro)


def test_agent_uses_stop_reason_and_returns_tool_results_to_provider():
    client = FakeClient([
        FakeResponse([
            {"type": "tool_use", "id": "call-1", "name": "echo", "input": {"value": "evidence"}},
        ], "tool_use"),
        FakeResponse([{"type": "text", "text": "Use the evidence."}], "end_turn"),
    ])

    async def collect():
        return [event async for event in run_agent_async(
            "What should I do?",
            Settings(),
            client=client,
            tools=TOOLS,
            tool_handlers={"echo": lambda args, _settings: args["value"]},
        )]

    events = run(collect())
    assert [event.kind for event in events] == [
        AgentEventType.START,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.TEXT,
        AgentEventType.DONE,
    ]
    second_request = client.messages.calls[1]["messages"]
    assert second_request[-1]["role"] == "user"
    assert second_request[-1]["content"][0]["tool_use_id"] == "call-1"
    assert second_request[-1]["content"][0]["content"] == "evidence"


def test_agent_runs_independent_tools_concurrently_and_preserves_order():
    client = FakeClient([
        FakeResponse([
            {"type": "tool_use", "id": "a", "name": "echo", "input": {"value": "a"}},
            {"type": "tool_use", "id": "b", "name": "echo", "input": {"value": "b"}},
        ], "tool_use"),
        FakeResponse([{"type": "text", "text": "Done."}], "end_turn"),
    ])
    started: List[str] = []

    async def echo(args, _settings):
        started.append(args["value"])
        await asyncio.sleep(0)
        return args["value"]

    async def collect():
        return [event async for event in run_agent_async(
            "Compare two options",
            Settings(),
            client=client,
            tools=TOOLS,
            tool_handlers={"echo": echo},
        )]

    events = run(collect())
    results = [event.text for event in events if event.kind == AgentEventType.TOOL_RESULT]
    assert started == ["a", "b"]
    assert results == ["a", "b"]


def test_agent_stops_repeated_tool_calls_before_infinite_loop():
    repeated = FakeResponse([
        {"type": "tool_use", "id": "same", "name": "echo", "input": {"value": "same"}},
    ], "tool_use")
    client = FakeClient([repeated, repeated])

    async def collect():
        return [event async for event in run_agent_async(
            "Keep checking",
            Settings(),
            client=client,
            tools=TOOLS,
            tool_handlers={"echo": lambda args, _settings: args["value"]},
            config=AgentRunConfig(max_repeated_tool_calls=1),
        )]

    events = run(collect())
    assert events[-1].kind == AgentEventType.STOP
    assert events[-1].error == "repeated_tool_call"
    assert len(client.messages.calls) == 2


def test_agent_retries_transient_provider_failure():
    class RateLimitError(Exception):
        status_code = 429

    client = FakeClient([
        RateLimitError("slow down"),
        FakeResponse([{"type": "text", "text": "Recovered."}], "end_turn"),
    ])

    async def collect():
        return [event async for event in run_agent_async(
            "Retry this",
            Settings(),
            client=client,
            config=AgentRunConfig(api_retries=1, retry_base_seconds=0),
        )]

    events = run(collect())
    assert any(event.kind == AgentEventType.RETRY for event in events)
    assert any(event.kind == AgentEventType.TEXT and event.text == "Recovered." for event in events)
