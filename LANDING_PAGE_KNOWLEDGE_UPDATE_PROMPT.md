# ChudGPT landing-page update prompt

Update the existing ChudGPT landing site in place. Preserve its current visual design, routes, responsive behavior, accessibility, navigation, legal pages, API documentation, and working links. Do not invent performance claims.

## ChudGPT-Public V20 wording

Update the ChudGPT-Public V20 model guide and related model cards so they accurately say:

- ChudGPT-Public V20 is a 20,999,184-parameter experimental decoder-only language model developed by Astra.
- Its model context is 1,024 tokens. Client or server conversation storage may retain more text, but only the recent context that fits the model window can affect a generation.
- It supports general conversation, broad introductory knowledge, basic reasoning, instruction following, simple code, and exact math handled by the Public API's deterministic math system.
- It is a very small experimental model. It can produce incorrect facts, invalid code, awkward grammar, or harmless nonsense, so users should verify important answers.
- The Public API does not secretly substitute a different ChudGPT model when Public V20 generation fails.
- ChudGPT-Public-Music V1 remains a separate music checkpoint and endpoint; do not describe it as a fallback for Public V20.

Explain that current model work uses a broad curriculum spanning science, computing, geography, history, language, logic, constraint-following, debugging, creativity, and conversational context. Do not claim that a small fixed benchmark proves universal knowledge or that Public V20 can answer every ChudLab prompt.

## Evaluation wording

If the site displays benchmark information, distinguish these clearly:

- A fixed 10-case acceptance suite checks required reasoning, constraints, C#/Unity, debugging, logic, creativity, personality, introductory GPU/CPU knowledge, unusual wording constraints, and a label puzzle.
- A separate broader held-out suite is used to catch over-specialization.
- Passing a fixed acceptance suite is not evidence of complete general knowledge.
- ChudLab is a model playground for manual testing, not a guarantee that every submitted question will be answered correctly.

Do not publish experimental checkpoint scores unless they correspond to the actually deployed checkpoint. Do not call rejected training candidates releases.

## Clean up old update logs

Remove old, superseded update-log cards, duplicate changelog entries, stale restart notices, expired quick-tunnel URLs, and historical status messages from the visible landing page. Keep only the current release/update entry plus any permanent changelog archive page that is intentionally part of the site. If there is an update history array or JSON file, delete superseded entries rather than hiding them with CSS. Do not delete legal-policy revision history that must be retained.

The current update entry should be concise and say that the Public V20 knowledge/training pipeline was audited, broader evaluation was added, benchmark-only regressions are rejected before deployment, and runtime-specific exact math, identity, greetings, and safety behavior remain documented separately from neural generation.

## Verification

After editing:

1. Search the whole project for expired `trycloudflare.com` URLs and remove them from user-facing content.
2. Confirm the model guide, model directory, API guide, Discord terms/privacy links, and mobile navigation still work.
3. Verify that old update entries are actually gone from source data and not merely visually hidden.
4. Test desktop and mobile layouts, especially the model cards and update section.
5. Run the project's existing build, lint, and tests.
6. Report exactly which files changed and list every factual model claim added to the site.
