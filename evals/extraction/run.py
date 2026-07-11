"""
Extraction prompt eval runner.

Usage:
    python -m evals.extraction.run                    # run all cases
    python -m evals.extraction.run --case redux       # run cases matching 'redux'
    python -m evals.extraction.run --verbose           # show LLM output for each case
"""

from __future__ import annotations

import json
import re

from ..common import load_prompt, generate, make_cli
from .dataset import CASES


_USER_BLOCK_RE = re.compile(r"<user>\s*(.*?)\s*</user>", re.DOTALL)


def _normalize(s: str) -> str:
    """Collapse whitespace runs and trim — used for substring-matching user_quote."""
    return " ".join(s.split())


def _extract_user_texts(conversation: str) -> list[str]:
    """Pull the raw text of every <user> block — used to anchor user_quote substrings."""
    return [m.group(1) for m in _USER_BLOCK_RE.finditer(conversation)]


async def execute(case: dict) -> dict:
    """Run the extraction prompt on a case. Returns {output, raw}.

    Mirrors production: the prompt returns objects with `learning` and
    `user_quote`. We validate that `user_quote` is a verbatim substring of some
    <user> block (whitespace-normalized) and emit only the `learning` strings.
    """
    prompt = load_prompt("extract").format(conversation=case["conversation"])
    raw = await generate(prompt)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"output": None, "raw": raw}

    if not isinstance(parsed, list):
        return {"output": None, "raw": raw}

    user_texts = [_normalize(t) for t in _extract_user_texts(case["conversation"])]
    output = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        learning = (item.get("learning") or "").strip()
        quote = _normalize((item.get("user_quote") or ""))
        if not learning or not quote:
            continue
        if not any(quote in ut for ut in user_texts):
            continue
        output.append(learning)
    return {"output": output, "raw": raw}


def format_result(result: dict) -> list:
    return [f"Output: {result.get('output')}"]


main = make_cli(
    description="Run extraction prompt evals",
    suite_name="extraction",
    cases=CASES,
    execute_fn=execute,
    format_result=format_result,
)

if __name__ == "__main__":
    main()
