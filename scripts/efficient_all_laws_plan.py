#!/usr/bin/env python3
"""Efficient post-V12 live plan: all laws covered, without 6k→20k one-by-one waste.

Strategy
--------
1. Keep 1k–4k passes already scored (contract/tort/criminal/land).
2. Run only four true longform stress tests: 5k / 10k / 15k / 20k.
3. Cover every other core matrix subject at 3,000 words (long enough for
   structure + OSCOLA gates, ~2–4× faster than the old escalating ladder).
4. Specialist subjects stay on the existing general+SQE enquiry suite
   (aviation, housing, cybercrime, …) instead of another 20k essay each.

Total live longform cases after 1k–4k: 4 stress + 12 compact = 16
(same subject count as before, far less wall-clock than 5k+6k+…+20k).
"""
from __future__ import annotations

from final_trial_sweep import QUESTIONS as FULL_LENGTH_QUESTIONS

# True longform stress (prove 5k–20k still works under release gates).
STRESS_WORDS = {5000, 10000, 15000, 20000}

# Compact longform for remaining core matrix subjects.
COMPACT_WORDS = 3000

# Subjects already passed at 1k–4k — do not redo in the efficient plan.
ALREADY_PASSED_PREFIXES = (
    "length_01000_",
    "length_02000_",
    "length_03000_",
    "length_04000_",
)


def build_efficient_questions() -> list[tuple[int, str, str, str]]:
    """Return (words, slug, register, stem) for the efficient remaining matrix."""
    by_words = {row[0]: row for row in FULL_LENGTH_QUESTIONS}
    out: list[tuple[int, str, str, str]] = []

    for words in sorted(STRESS_WORDS):
        out.append(by_words[words])

    for words, slug, register, stem in FULL_LENGTH_QUESTIONS:
        if words in STRESS_WORDS:
            continue
        if words <= 4000:
            continue  # already covered / passed
        # Remap former 6k–19k slots to compact 3k for the same subject+stem.
        out.append((COMPACT_WORDS, slug, register, stem))

    # Stable order: stress ascending, then compact in original subject order.
    stress = [row for row in out if row[0] in STRESS_WORDS]
    compact = [row for row in out if row[0] not in STRESS_WORDS]
    return stress + compact


EFFICIENT_QUESTIONS = build_efficient_questions()


def plan_summary() -> str:
    stress = [w for w, *_ in EFFICIENT_QUESTIONS if w in STRESS_WORDS]
    compact = [slug for w, slug, *_ in EFFICIENT_QUESTIONS if w not in STRESS_WORDS]
    return (
        f"efficient plan: stress={stress} compact_{COMPACT_WORDS}w="
        f"{len(compact)} subjects ({', '.join(compact)})"
    )


if __name__ == "__main__":
    print(plan_summary())
    for words, slug, register, stem in EFFICIENT_QUESTIONS:
        print(f"{words:5d}  {slug:28s}  {register}")
