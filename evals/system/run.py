"""
System prompt eval runner.

Tests tool-calling decisions and response quality of the tutor agent.

Usage:
    python -m evals.system.run                    # run all cases
    python -m evals.system.run --case greeting    # run cases matching 'greeting'
    python -m evals.system.run --verbose           # show full responses
"""

from __future__ import annotations

import json

from ..common import (
    load_prompt,
    generate_with_system,
    make_cli,
    MODEL,
    TEMPERATURE,
    get_client,
)
from .dataset import CASES


# ── Tool definitions (matching production tools) ──────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_all_knowledge",
            "description": "Search articles, user knowledge (learnings + authored articles), and user's uploaded documents. Primary search tool for answering general questions.",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string"}},
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_user_knowledge",
            "description": "Search only what the user already knows — their past learnings and articles they have authored. Does NOT search uploaded documents. Use in quiz mode to baseline existing knowledge.",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string"}},
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_learnings",
            "description": "Get the user's most recent learnings ordered by time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_content",
            "description": "Read content by content_id. Shows native image/PDF files when available; otherwise reads text. Optional start/end (1-indexed, inclusive) scope it: pages for a PDF, lines for text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_id": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["content_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spaced_learnings",
            "description": "Get items due for spaced repetition review (overdue or never reviewed), most overdue first. Use when the user wants to review or asks what is due.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_contents",
            "description": "List the user's uploaded files and saved URLs (title + content_id), newest first. Use when the user asks what they have uploaded or to find a content_id.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search the web. Use ONLY as a last resort — when search_all_knowledge returns nothing useful and you cannot answer from the knowledge base. Do not use in quiz mode or when study content is available.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _build_system_prompt(learner_profile: str, current_content: str) -> str:
    return load_prompt("system").format(
        learner_context=f"\n{learner_profile}\n" if learner_profile else "",
        authored_articles="",
        session_content="",
        content_block=f"\n{current_content}\n" if current_content else "",
    )


# Cap on tool round-trips when resolving a case's mocked tool calls, so a model
# that keeps calling tools can't loop forever.
MAX_TOOL_ROUNDS = 6


async def execute(case: dict) -> dict:
    """Run the system prompt with tools on a case. Returns {content, tool_calls}.

    If the case supplies `mock_tool_results`, resolve the model's tool calls in a
    bounded loop: feed back a mock result for each call, re-prompt, and repeat
    until the model returns a text response (or MAX_TOOL_ROUNDS is reached). This
    lets us judge the final teaching response even when the model needs several
    tool round-trips before answering — e.g. quiz mode calls get_spaced_learnings
    and then search_user_knowledge before asking its question. Tool calls from
    every round are accumulated for the tool-call assertions.
    """
    system = _build_system_prompt(case["learner_profile"], case["current_content"])
    first = await generate_with_system(system, case["user_message"], tools=TOOLS)

    # No mocks or no tool calls: nothing to resolve, return the single turn as-is.
    mock_results = case.get("mock_tool_results")
    if not mock_results or not first["tool_calls"]:
        return first

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": case["user_message"]},
    ]

    content = first["content"] or ""
    pending = first["tool_calls"]
    all_tool_calls = list(first["tool_calls"])
    call_seq = 0

    for _ in range(MAX_TOOL_ROUNDS):
        # Replay the assistant turn that issued `pending`, then a mock result per call.
        assistant_msg = {"role": "assistant", "content": content or None, "tool_calls": []}
        call_ids = []
        for tc in pending:
            call_id = f"call_{call_seq}"
            call_seq += 1
            call_ids.append(call_id)
            assistant_msg["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                }
            )
        messages.append(assistant_msg)
        for call_id, tc in zip(call_ids, pending):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": mock_results.get(tc["name"], "No results found."),
                }
            )

        resp = await get_client().chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        pending = [
            {"name": tc.function.name, "args": json.loads(tc.function.arguments)}
            for tc in (msg.tool_calls or [])
        ]
        all_tool_calls.extend(pending)

        if not pending:
            break  # model produced its final text answer

    return {"content": content, "tool_calls": all_tool_calls}


def format_result(result: dict) -> list:
    lines = []
    tool_names = [tc["name"] for tc in result.get("tool_calls", [])]
    if tool_names:
        lines.append(f"Tools called: {tool_names}")
    if result.get("content"):
        lines.append(f"Response: {result['content'][:200]}...")
    return lines


main = make_cli(
    description="Run system prompt evals",
    suite_name="system",
    cases=CASES,
    execute_fn=execute,
    format_result=format_result,
)

if __name__ == "__main__":
    main()
