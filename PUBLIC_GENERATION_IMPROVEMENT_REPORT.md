# Public generation audit

## Changes

Short instructions and questions no longer receive the casual-chat instruction
or automatic 80-token cap. They retain the caller's requested generation budget.
The recent-history slice now retains four complete exchanges plus the current
user turn instead of starting with an orphaned assistant answer.

No existing fallback was added, removed, or changed. The checkpoint, response
routers, sampling profiles, and generated-response selection remain unchanged.

## Verification

363 tests passed, including regression tests covering five short tasks and
the roles and contents of the retained conversation window.

`evaluate_generation_regressions.py` records neural-only answers using fixed
seeds. It bypasses the math, identity, greeting, and other response routers.
Local outputs are in `reports/generation_budget_before.json`,
`reports/generation_budget_after.json`, and
`reports/generation_knowledge_candidate.json`.

Neither the current checkpoint nor the earlier full-knowledge candidate solved
the new Java-loop, refrigerator, TCP/UDP, string-reversal, supplied-text summary,
or conversation-name recall tasks in this run. The before/after comparison does
not establish a general knowledge gain. The candidate was not deployed.

The recurring CPU/GPU, misplaced-box, and survival-story fragments suggest
overlearning the earlier small challenge curriculum. The remaining problem is
learned generalization, not a missing canned response. Future checkpoint changes
need independent evaluations and varied training, not more repetitions of these
evaluation questions.
