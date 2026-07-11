"""
Merge prompt eval runner.

Usage:
    python -m evals.merge.run                    # run all cases
    python -m evals.merge.run --case analogy     # run cases matching 'analogy'
    python -m evals.merge.run --verbose           # show LLM output for each case
"""

from __future__ import annotations

import json

from ..common import load_prompt, generate, make_cli
from .dataset import CASES


async def execute(case: dict) -> dict:
    """Run the merge prompt on a case. Returns {output, ops, raw}.

    'output' is the combined list of updated + added content strings, used by
    the assertion runner (contains_idea, preserves_voice, count_range, etc.).
    'ops' holds the raw structured response for verbose display.
    """
    # Assign stable fake IDs so the LLM can reference them
    existing_items = [
        {"id": f"existing-{i}", "content": e}
        for i, e in enumerate(case["existing"])
    ]
    existing_text = (
        "\n".join(f"- id={item['id']}: {item['content']}" for item in existing_items)
        if existing_items
        else "(none)"
    )
    new_text = "\n".join(f"- {n}" for n in case["new_learnings"])

    prompt = load_prompt("merge").format(existing=existing_text, new_learnings=new_text)
    raw = await generate(prompt)

    valid_ids = {item["id"] for item in existing_items}

    try:
        ops = json.loads(raw.strip())
        if not isinstance(ops, dict):
            raise ValueError("Expected a JSON object")

        # Validate referenced IDs against the pool
        updates = [
            u for u in (ops.get("update") or [])
            if isinstance(u, dict) and u.get("id") in valid_ids and u.get("content")
        ]
        adds = [
            a.strip() for a in (ops.get("add") or [])
            if isinstance(a, str) and a.strip()
        ]

        # Combined content for assertion checks
        output = [u["content"] for u in updates] + adds
        return {
            "output": output,
            "ops": {"update": updates, "add": adds},
            "raw": raw,
        }
    except (json.JSONDecodeError, ValueError):
        return {"output": None, "ops": None, "raw": raw}


def format_result(result: dict) -> list:
    """Format structured ops for verbose display in the eval runner."""
    ops = result.get("ops")
    if ops is None:
        return [f"Output (parse failed): {result.get('raw', '')[:200]}"]
    lines = []
    for u in ops["update"]:
        lines.append(f"UPDATE {u['id']}: {u['content'][:120]}")
    for a in ops["add"]:
        lines.append(f"ADD: {a[:120]}")
    if not lines:
        lines.append("(no ops)")
    return lines


main = make_cli(
    description="Run merge prompt evals",
    suite_name="merge",
    cases=CASES,
    execute_fn=execute,
    format_result=format_result,
)

if __name__ == "__main__":
    main()
