"""
Merge prompt eval dataset.

IMPORTANT: Eval cases must NOT reuse data from the prompt's examples.
The prompt uses RAG/cheat-sheet and BERT/GPT as its examples — do not test those.

Each case has:
  - name: short identifier
  - existing: list of existing knowledge strings (assigned fake IDs by the runner)
  - new_learnings: list of new learning strings
  - assertions: list of checks to run on the output
      {"type": "count_range", "min": N, "max": N}  — number of updated+added items
      {"type": "contains_idea", "idea": "..."}      — output semantically contains this
      {"type": "preserves_voice", "phrase": "..."}  — user's exact phrasing is kept verbatim
      {"type": "does_not_contain", "idea": "..."}   — output must NOT contain this idea
      {"type": "no_hallucination"}                  — output contains only info from inputs
"""

CASES = [
    # ── Overlapping knowledge gets merged (update in-place) ───────────────
    {
        "name": "merge_overlapping_gradient_descent_notes",
        "description": (
            "Existing and new both discuss gradient descent"
            " — should update existing in-place"
        ),
        "existing": [
            "Gradient descent iteratively updates weights by moving in the"
            " direction that reduces the loss"
        ],
        "new_learnings": [
            "The learning rate controls step size in gradient descent"
            " — too big overshoots the valley, too small takes forever",
            "Stochastic gradient descent uses random mini-batches instead of"
            " the full dataset, which adds noise but speeds up training",
        ],
        "assertions": [
            {"type": "count_range", "min": 1, "max": 3},
            {
                "type": "contains_idea",
                "idea": "gradient descent updates weights to reduce loss",
            },
            {
                "type": "contains_idea",
                "idea": "learning rate controls step size — too big overshoots, too small is slow",
            },
            {
                "type": "contains_idea",
                "idea": "SGD uses mini-batches instead of full dataset",
            },
            {"type": "preserves_voice", "phrase": "overshoots the valley"},
        ],
    },
    # ── Distinct concepts: unrelated existing item must NOT be touched ────
    {
        "name": "keep_distinct_concepts_separate",
        "description": (
            "Unrelated existing item should be left alone;"
            " only the new learning is added"
        ),
        "existing": [
            "Docker containers package an application with its dependencies"
            " into an isolated environment"
        ],
        "new_learnings": [
            "A mutex ensures only one thread can access a shared resource at a time"
        ],
        "assertions": [
            # Only the mutex learning should be added; Docker item is untouched
            {"type": "count_range", "min": 1, "max": 1},
            {
                "type": "contains_idea",
                "idea": "a mutex ensures single-thread access to shared resources",
            },
            # Docker content must NOT be blended into the mutex note
            {
                "type": "does_not_contain",
                "idea": "Docker containers isolate applications",
            },
        ],
    },
    # ── User analogies must survive merging ───────────────────────────────
    {
        "name": "preserve_user_analogy",
        "description": "User's personal analogy must not be replaced with generic phrasing",
        "existing": [
            "A load balancer distributes incoming requests across multiple"
            " servers to prevent overload"
        ],
        "new_learnings": [
            "A load balancer is like a restaurant host — seats customers at"
            " different tables so no waiter gets slammed"
        ],
        "assertions": [
            {"type": "count_range", "min": 1, "max": 1},
            {"type": "preserves_voice", "phrase": "restaurant host"},
            {
                "type": "contains_idea",
                "idea": "load balancer distributes requests across servers",
            },
        ],
    },
    # ── No information loss on merge ──────────────────────────────────────
    {
        "name": "no_information_loss_on_merge",
        "description": "All unique facts from both inputs must appear in the output",
        "existing": [
            "CNNs use parameter sharing — the same filter slides across the whole image"
        ],
        "new_learnings": [
            "CNNs are translation invariant because the same filter detects"
            " a feature regardless of position",
            "Pooling layers reduce spatial dimensions and provide some translation tolerance",
        ],
        "assertions": [
            {"type": "count_range", "min": 1, "max": 3},
            {
                "type": "contains_idea",
                "idea": "CNNs use parameter sharing — same filter slides across the image",
            },
            {
                "type": "contains_idea",
                "idea": (
                    "CNNs are translation invariant because filters detect"
                    " features regardless of position"
                ),
            },
            {
                "type": "contains_idea",
                "idea": "pooling layers reduce spatial dimensions",
            },
        ],
    },
    # ── Correction: wrong existing item is updated with correct content ───
    {
        "name": "correction_updates_wrong_item",
        "description": (
            "A new learning that corrects a wrong existing item"
            " should update it in-place, not skip it"
        ),
        "existing": [
            "TCP is a stateless protocol — it does not track connection state between packets"
        ],
        "new_learnings": [
            "TCP is actually stateful — it maintains connection state through"
            " the three-way handshake and tracks sequence numbers"
        ],
        "assertions": [
            {"type": "count_range", "min": 1, "max": 1},
            {
                "type": "contains_idea",
                "idea": "TCP is stateful and maintains connection state",
            },
            {
                "type": "does_not_contain",
                "idea": "TCP is stateless",
            },
            {"type": "no_hallucination"},
        ],
    },
]
