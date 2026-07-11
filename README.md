# decodelm-evals

A small, LLM-as-judge evaluation harness for the LLM features behind
[DecodeLM](https://decodelm.com) — an AI tutor built to make people think rather than
answer for them. It is a **regression-catcher and prompt A/B rig**, not a benchmark: the
model under test and the judge are small, and the datasets are hand-curated from real
failure modes. The value is in catching regressions and comparing prompts, not in
headline scores.

> **This repo is a demonstration extract, not the live system.** The prompts in
> [`prompts/`](prompts/) are **illustrative examples** written to match each dataset's
> behavior, _not_ DecodeLM's production prompts (those stay private). The datasets are
> **small, hand-written / mocked** sets, and the `retrieval` suite runs over a **tiny
> in-memory corpus** instead of the production vector index. The point is to show the
> harness and methodology, not to reproduce DecodeLM.

## Where these evals sit in DecodeLM

DecodeLM tutors a learner, then turns the conversation into durable, personalized
knowledge. Each eval suite guards one stage of that pipeline:

1. **Tutor session** (`system`) → the agent teaches Socratically, grounding answers in sources via tools.
2. **Extract learnings** (`extraction`) → pull out only what the learner actually demonstrated, in their own words.
3. **Merge into the knowledge base** (`merge`) → fold new learnings in without losing facts or the user's voice.
4. **Retrieve to ground the next answer** (`retrieval`) → find the right sources, then feed back into step 1.

| Suite          | Stage it guards            | What it checks                                                                                  | Method                                                                          |
| -------------- | -------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **system**     | the tutor's behavior       | Socratic when useful, direct when not; correct tool use; verdict-first grading; one-step pacing | tool-call assertions + a second turn on mocked tool results, judged for quality |
| **extraction** | what it learns about you   | pulls out only genuine, user-demonstrated learnings, in the user's voice                        | assertions + LLM judge + a deterministic faithfulness anchor                    |
| **merge**      | how that knowledge is kept | dedups and updates in place, loses no facts, keeps the user's analogies, corrects wrong items   | `contains_idea` / `preserves_voice` / `no_hallucination` judges                 |
| **retrieval**  | how sources are found      | the right source is retrieved and ranked high                                                   | recall@k, MRR, hit-rate vs ground truth                                         |

## Design

- **Framework** ([`evals/common.py`](evals/common.py)): a generic runner, an assertion
  engine, LLM-judge helpers, and a results writer shared by every suite. A suite is just a
  `dataset.py` (cases + assertions) and a `run.py` (how to execute one case).
- **Assertions** mix deterministic and judged checks:
  - deterministic: `exact_empty`, `exact_count`, `count_range`, `preserves_voice`
    (verbatim substring), tool-call checks (`calls_tool`, `no_tool_calls`, `does_not_call`).
  - judged (LLM-as-judge): `semantic` (does any output mean the same as expected?),
    `no_hallucination` (is every output fact grounded in the inputs?), `response_quality`
    (does the response meet free-text criteria, e.g. "corrects the misconception without
    opening with 'you're wrong'?").
- **Faithfulness anchoring** (extraction): each extracted learning must carry a
  `user_quote` that is a verbatim substring of a `<user>` turn — a cheap, deterministic
  check that catches the model attributing a learning the user never said.
- **Behavioral testing with mocked tools** (system): if a case supplies
  `mock_tool_results`, the runner feeds them back and takes a second turn, so it can judge
  the final teaching response after tool use.
- **Reproducible runs**: every run appends a timestamped, git-SHA-stamped entry to
  `results/<suite>.md`, so quality is diffable over time (`git log -p results/...`).
- **Prompt A/B**: `--prompt-file <path>` swaps the prompt under test, so you can baseline
  the checked-in prompt against a draft and compare adjacent result entries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Provide an OpenAI key via the environment or a `.env` file in the repo root
(git-ignored). Only an API key is required; the models below have working
defaults.

```bash
# .env
OPENAI_API_KEY=sk-...
EVAL_MODEL=gpt-4o-mini                   # optional; models are configurable
EVAL_JUDGE_MODEL=gpt-4o-mini             # optional
EVAL_EMBED_MODEL=text-embedding-3-small  # optional; retrieval suite only
```

## Run it

```bash
export OPENAI_API_KEY=sk-...            # or put it in the .env file above

python -m evals.extraction.run                 # faithfulness of learning extraction
python -m evals.merge.run                      # knowledge merge / dedup
python -m evals.system.run                     # tutor behavior + tool use
python -m evals.retrieval.run                  # recall@k / MRR over the demo corpus

# handy flags (assertion suites)
python -m evals.system.run --case socratic --verbose
python -m evals.extraction.run --prompt-file prompts/your-variant.txt   # A/B your own prompt
```

## Layout

```
evals/
  common.py            # framework: runner, assertion engine, LLM judges, results writer
  extraction/          # did we extract only genuine, user-voiced learnings?
  merge/               # did we fold learnings in without loss or drift?
  system/              # does the tutor behave to spec (Socratic, tool use)?
  retrieval/           # recall@k / MRR over an in-memory corpus
prompts/
  extract.txt          # EXAMPLE prompts (not production); swap in your own
  merge.txt
  system.txt
```
