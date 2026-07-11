"""
Shared eval infrastructure: LLM client, assertions, runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable

from openai import AsyncOpenAI

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────

MODEL = os.getenv("EVAL_MODEL", "gpt-5.4-nano")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5.4-nano")
TEMPERATURE = 0.2  # match production setting

# Prompts live as flat .txt files under the repo's top-level prompts/ dir.
# Versioning is git's job; pass --prompt-file <path> to A/B any alternate file.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Set by make_cli's --prompt-file flag. When set, overrides the kind-based
# lookup so a runner can test against any prompt file on disk.
PROMPT_FILE_OVERRIDE: Path | None = None


def load_prompt(kind: str) -> str:
    """Load a prompt template by kind from prompts/<kind>.txt.

    A --prompt-file override (set by make_cli) takes precedence, so a runner
    can A/B against any alternate prompt file on disk.
    """
    if PROMPT_FILE_OVERRIDE is not None:
        return PROMPT_FILE_OVERRIDE.read_text()
    candidate = PROMPTS_DIR / f"{kind}.txt"
    if candidate.exists():
        return candidate.read_text()
    raise FileNotFoundError(f"Prompt '{kind}.txt' not found in {PROMPTS_DIR}")


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Lazily construct (and cache) the AsyncOpenAI client on first use.

    Constructing at import time would require OPENAI_API_KEY just to run
    `--help` or list cases, since importing any suite imports this module.
    Deferring it keeps non-API operations keyless.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


# ── LLM helpers ───────────────────────────────────────────────────────


async def generate(prompt: str, model: str = None, temperature: float = None) -> str:
    """Generate a completion and return the raw text."""
    resp = await get_client().chat.completions.create(
        model=model or MODEL,
        temperature=temperature if temperature is not None else TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


async def generate_with_system(
    system: str,
    user: str,
    model: str = None,
    tools: list = None,
) -> dict:
    """Generate with system + user messages. Returns {content, tool_calls}."""
    kwargs = dict(
        model=model or MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tools:
        kwargs["tools"] = tools

    resp = await get_client().chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    return {
        "content": msg.content or "",
        "tool_calls": [
            {"name": tc.function.name, "args": json.loads(tc.function.arguments)}
            for tc in (msg.tool_calls or [])
        ],
    }


async def judge_yes_no(prompt: str) -> bool:
    """Ask the judge model a yes/no question."""
    resp = await get_client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip().lower().startswith("yes")


async def semantic_match(expected: str, candidates: list) -> bool:
    """Check if any candidate semantically matches the expected string."""
    if not candidates:
        return False

    candidates_text = "\n".join(f"- {c}" for c in candidates)
    return await judge_yes_no(
        f"""You are a strict semantic similarity judge.

Does ANY of the following extracted items capture the same core meaning as the reference?

Reference: "{expected}"

Items:
{candidates_text}

Rules:
- The wording does NOT need to match exactly — judge by meaning.
- The item must cover the same core concept. Partial overlap is NOT a match.
- Respond with ONLY "yes" or "no"."""
    )


# ── Assertion engine ──────────────────────────────────────────────────
# All assertion types across all eval suites live here.


async def check_assertion(assertion: dict, result: dict, case: dict) -> dict:
    """
    Check a single assertion against the eval result.

    `result` is whatever the execute function returned — typically:
      - {"output": [...], "raw": "..."} for extraction/merge
      - {"content": "...", "tool_calls": [...]} for system
    """
    atype = assertion["type"]

    # ── Output-based assertions (extraction, merge) ───────────────
    output = result.get("output")
    expected = case.get("expected", [])

    if atype == "exact_empty":
        if output == []:
            return {"passed": True, "reason": "Correctly returned []"}
        return {
            "passed": False,
            "reason": f"Expected [] but got {len(output)} items: {output}",
        }

    if atype == "exact_count":
        if len(output) == len(expected):
            return {"passed": True, "reason": f"Correct count: {len(output)}"}
        return {
            "passed": False,
            "reason": f"Expected {len(expected)} items, got {len(output)}",
        }

    if atype == "semantic":
        if len(output) > len(expected) + 1:
            return {
                "passed": False,
                "reason": f"Too many items: expected ~{len(expected)}, got {len(output)}: {output}",
            }
        results = await asyncio.gather(
            *[semantic_match(exp, output) for exp in expected]
        )
        missing = [exp for exp, matched in zip(expected, results) if not matched]
        if not missing:
            return {
                "passed": True,
                "reason": f"All {len(expected)} expected items matched",
            }
        return {
            "passed": False,
            "reason": f"Missing {len(missing)}/{len(expected)}: {missing}",
        }

    if atype == "count_range":
        lo, hi = assertion["min"], assertion["max"]
        n = len(output)
        if lo <= n <= hi:
            return {"passed": True, "reason": f"Count {n} in [{lo}, {hi}]"}
        return {"passed": False, "reason": f"Count {n} not in [{lo}, {hi}]"}

    if atype == "contains_idea":
        idea = assertion["idea"]
        matched = await semantic_match(idea, output)
        if matched:
            return {"passed": True, "reason": f"Found: {idea[:60]}..."}
        return {"passed": False, "reason": f"Missing idea: {idea}"}

    if atype == "does_not_contain":
        idea = assertion["idea"]
        matched = await semantic_match(idea, output)
        if not matched:
            return {"passed": True, "reason": f"Correctly absent: {idea[:60]}..."}
        return {"passed": False, "reason": f"Should not contain: {idea}"}

    if atype == "preserves_voice":
        phrase = assertion["phrase"].lower()
        # Check across output list items or raw content string
        text_to_check = (
            " ".join(output).lower()
            if output is not None
            else result.get("content", "").lower()
        )
        if phrase in text_to_check:
            return {"passed": True, "reason": f"Preserved: '{assertion['phrase']}'"}
        return {
            "passed": False,
            "reason": f"Lost user voice: '{assertion['phrase']}' not found in output",
        }

    if atype == "no_hallucination":
        all_inputs = "\n".join(
            f"- {s}" for s in case.get("existing", []) + case.get("new_learnings", [])
        )
        all_outputs = "\n".join(f"- {s}" for s in output)
        hallucinated = not await judge_yes_no(
            f"""You are a strict factual grounding judge.

Does the output contain ONLY information that is present in the inputs? No new facts, no expanded explanations beyond what's stated.

Inputs:
{all_inputs}

Output:
{all_outputs}

Respond with ONLY "yes" (all grounded) or "no" (contains new information)."""
        )
        if not hallucinated:
            return {"passed": True, "reason": "No hallucination detected"}
        return {"passed": False, "reason": "Output contains information not in inputs"}

    # ── Tool-call assertions (system) ─────────────────────────────
    tool_calls = result.get("tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls]
    content = result.get("content", "")

    if atype == "no_tool_calls":
        if not tool_calls:
            return {"passed": True, "reason": "No tool calls (correct)"}
        return {
            "passed": False,
            "reason": f"Expected no tools but called: {tool_names}",
        }

    if atype == "calls_tool":
        target = assertion["tool"]
        if target in tool_names:
            return {"passed": True, "reason": f"Called {target}"}
        return {
            "passed": False,
            "reason": f"Expected {target} but called: {tool_names or '(none)'}",
        }

    if atype == "calls_any_tool":
        if tool_calls:
            return {"passed": True, "reason": f"Called: {tool_names}"}
        return {
            "passed": False,
            "reason": "Expected at least one tool call but got none",
        }

    if atype == "does_not_call":
        target = assertion["tool"]
        if target not in tool_names:
            return {"passed": True, "reason": f"Did not call {target}"}
        return {"passed": False, "reason": f"Should not have called {target}"}

    if atype == "response_quality":
        criteria = assertion["criteria"]
        if not content and tool_calls:
            return {"passed": True, "reason": "Tool call only (no text to judge yet)"}
        passed = await judge_yes_no(
            f"""You are evaluating an AI tutor's response quality.

Criteria: {criteria}

Response:
\"\"\"
{content}
\"\"\"

Does the response meet ALL of the criteria above? Respond with ONLY "yes" or "no"."""
        )
        if passed:
            return {
                "passed": True,
                "reason": f"Quality check passed: {criteria[:60]}...",
            }
        return {"passed": False, "reason": f"Quality check failed: {criteria[:80]}..."}

    return {"passed": False, "reason": f"Unknown assertion type: {atype}"}


# ── Shared scoring & runner ───────────────────────────────────────────


async def score_case(case: dict, result: dict) -> dict:
    """Run all assertions for a case and return {passed, reason, details}."""
    checks = await asyncio.gather(
        *[check_assertion(a, result, case) for a in case["assertions"]]
    )
    failed = [c for c in checks if not c["passed"]]

    if not failed:
        return {
            "passed": True,
            "reason": f"All {len(checks)} assertions passed",
            "details": checks,
        }
    return {
        "passed": False,
        "reason": f"{len(failed)}/{len(checks)} assertions failed",
        "details": checks,
    }


async def run_eval(
    suite_name: str,
    cases: list,
    execute_fn: Callable[[dict], Awaitable[dict]],
    filter_str: str = None,
    verbose: bool = False,
    format_result: Callable[[dict], list] = None,
):
    """
    Generic eval runner.

    Args:
        suite_name: Display name for the suite (e.g., "extraction")
        cases: List of case dicts (each must have "name" and "assertions")
        execute_fn: async function(case) -> result dict passed to assertions
        filter_str: Optional name substring filter
        verbose: Show full output for passing cases too
        format_result: Optional function(result) -> list of extra lines to print
    """
    if filter_str:
        cases = [c for c in cases if filter_str.lower() in c["name"].lower()]

    if not cases:
        print(f"No cases matching '{filter_str}'")
        return []

    print(f"Running {len(cases)} {suite_name} eval cases")
    prompt_src = (
        str(PROMPT_FILE_OVERRIDE) if PROMPT_FILE_OVERRIDE is not None else "checked-in"
    )
    print(f"Model: {MODEL} | Judge: {JUDGE_MODEL} | Prompt: {prompt_src}")
    print("=" * 70)

    results = []
    start = time.time()

    for case in cases:
        result = await execute_fn(case)
        score = await score_case(case, result)
        entry = {**case, "result": result, **score}
        results.append(entry)

        status = "PASS" if score["passed"] else "FAIL"
        print(f"  {'✓' if score['passed'] else '✗'} {status}  {case['name']}")
        if verbose or not score["passed"]:
            print(f"         {score['reason']}")
            # Extra context from format_result
            if format_result:
                for line in format_result(result):
                    print(f"         {line}")
            for detail in score.get("details", []):
                mark = "✓" if detail["passed"] else "✗"
                print(f"           {mark} {detail['reason']}")
        print()

    elapsed = time.time() - start
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("=" * 70)
    print(
        f"Result: {passed}/{total} passed ({passed / total * 100:.0f}%) in {elapsed:.1f}s"
    )

    if passed < total:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['name']}: {r['reason']}")

    md_path = _write_results(
        suite_name, results, elapsed, prompt_src, format_result
    )
    print(f"\nSaved: {md_path.relative_to(Path.cwd())}")

    return results


# ── Results persistence ───────────────────────────────────────────────
# Each run writes a markdown + json under evals/results/. Files are
# .gitignored by default; commit a landmark run explicitly to keep it
# as a reference point.

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_sha() -> str:
    """Short git SHA + `-dirty` if the working tree has uncommitted changes."""
    try:
        cwd = Path(__file__).resolve().parent
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _write_results(
    suite_name: str,
    results: list,
    elapsed: float,
    prompt_src: str,
    format_result: Callable[[dict], list] | None,
) -> Path:
    """Append a new run entry to the top of `<suite>.md`.

    The file is a reverse-chronological log so the latest run is always
    at the top. Git history is the long-term record; teammates always
    see the most recent committed result without digging through dirs.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = _git_sha()
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = (passed / total * 100) if total else 0.0

    entry = [
        f"## {ts} · `{sha}`",
        "",
        f"- Model: `{MODEL}` | Judge: `{JUDGE_MODEL}`",
        f"- Prompt: {prompt_src}",
        f"- Result: **{passed}/{total} ({pass_rate:.0f}%)** in {elapsed:.1f}s",
        "",
    ]
    for r in results:
        mark = "✅" if r["passed"] else "❌"
        entry.append(f"### {mark} `{r['name']}`")
        if not r["passed"]:
            entry.append(f"- {r['reason']}")
            for detail in r.get("details", []):
                m = "✓" if detail["passed"] else "✗"
                entry.append(f"  - {m} {detail['reason']}")
        if format_result:
            extras = format_result(r.get("result") or {})
            if extras:
                entry.append("")
                entry.append("```")
                entry.extend(extras)
                entry.append("```")
        entry.append("")
    entry.append("---")
    entry.append("")
    entry_text = "\n".join(entry)

    md_path = RESULTS_DIR / f"{suite_name}.md"
    header = (
        f"# {suite_name} eval history\n\n"
        f"Reverse-chronological. Each entry is one run; latest at the top.\n\n"
        f"---\n\n"
    )
    existing = md_path.read_text() if md_path.exists() else header
    if not existing.startswith(f"# {suite_name} eval history"):
        existing = header + existing
    # Insert the new entry right after the file header's separator.
    head, sep, body = existing.partition("---\n\n")
    if sep:
        md_path.write_text(head + sep + entry_text + body)
    else:
        md_path.write_text(header + entry_text)
    return md_path


def make_cli(
    description: str, suite_name: str, cases: list, execute_fn, format_result=None
):
    """Create a standard CLI main() for an eval suite."""

    def main():
        global PROMPT_FILE_OVERRIDE
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--case", type=str, help="Filter cases by name substring")
        parser.add_argument("--verbose", action="store_true", help="Show all outputs")
        parser.add_argument(
            "--prompt-file",
            type=Path,
            default=None,
            help=(
                "Path to an alternate prompt .txt file to test against "
                "(default: read the current checked-in prompt). Useful "
                "for A/B comparing against a draft or `git show` output."
            ),
        )
        args = parser.parse_args()
        if args.prompt_file is not None:
            PROMPT_FILE_OVERRIDE = args.prompt_file
        asyncio.run(
            run_eval(
                suite_name=suite_name,
                cases=cases,
                execute_fn=execute_fn,
                filter_str=args.case,
                verbose=args.verbose,
                format_result=format_result,
            )
        )

    return main
