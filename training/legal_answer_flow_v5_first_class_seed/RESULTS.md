# v5 First-Class Seed Result

Date: 12 July 2026

## Purpose

Test whether a small, legally curated three-part contract-law seed could correct the failed LexiAI problem answer without repeating the v4 overfitting pattern.

## Runs

- Seed: 12 iterations, learning rate `1e-6`; validation loss `1.438 -> 1.437`.
- Resume: 18 iterations, learning rate `3e-6`; validation loss `1.437 -> 1.433`.
- v4 iteration-20 checkpoint was also tested because it had the best historical validation loss.

## Live Gate

All candidates were tested against the same saved 2,000-word LexiAI/BrightCloud contract problem through the site. Each candidate still failed the final structure, citation, completeness, or legal-quality gate. The outputs were withheld and the planned Part 1 remained open.

## Deployment Decision

Rejected. The stable `legal_answer_flow_feedback_v3_clean_lora` adapter remains live.

## Required Next Dataset

Do not increase iterations on the current corpus. Build a larger, independently verified gold set across every subject, with separate train/validation/test questions, 500-800 word continuation parts, full inline OSCOLA, used-authority-only References sections, and explicit essay/problem coverage. Marker comments and scripts below 70 may inform the rubric but must not become answer targets.
