#!/usr/bin/env python3
"""Efficient post-V12 live plan: all laws covered, without 6k→20k one-by-one waste.

Strategy
--------
1. Keep 1k–4k passes already scored (contract/tort/criminal/land).
2. Phase-1 longform stress: 5k / 10k / 12k only (skip 15k–20k until later).
3. Cover other core subjects that sat between 5k–12k at 3,000 words.
4. Specialist subjects stay on general+SQE (not another mega-essay each).

Phase 1 total after 1k–4k: 3 stress + ~6 compact, then general+SQE + publish.
15k–20k stress can be re-enabled later by raising MAX_WORDS.
"""
from __future__ import annotations

import os

from final_trial_sweep import QUESTIONS as FULL_LENGTH_QUESTIONS

# Phase-1 ceiling: prove up to 12k first (15k/20k deferred — too slow).
MAX_WORDS = int(os.environ.get("LEGAL_MAX_WORDS", "12000"))

# True longform stress within the phase-1 ceiling.
STRESS_WORDS = {w for w in (5000, 10000, 12000) if w <= MAX_WORDS}

# Compact longform for remaining core matrix subjects under the ceiling.
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
        if words > MAX_WORDS:
            continue  # defer 13k–20k (or whatever is above the phase ceiling)
        # Remap former mid-ladder slots to compact 3k for the same subject+stem.
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
        f"efficient plan max={MAX_WORDS}: stress={stress} compact_{COMPACT_WORDS}w="
        f"{len(compact)} subjects ({', '.join(compact)})"
    )


if __name__ == "__main__":
    print(plan_summary())
    for words, slug, register, stem in EFFICIENT_QUESTIONS:
        print(f"{words:5d}  {slug:28s}  {register}")
