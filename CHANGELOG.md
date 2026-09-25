# ChudGPT-Public changelog

## 2026-09-25 — Hosted simulation actions

- Added `POST /api/models/public/sim-step` for bounded neural selection of registered world functions. Observations are limited to 3,000 characters; action scoring uses at most 600 context tokens. Busy requests return 429.
- Existing chat routes, checkpoints, fallbacks and racial-slur input filtering are preserved.

## 2026-09-24 — Public racial-slur input filter

- Base racial slurs are matched case-insensitively as whole words. Slur-only messages receive a short request to rephrase and are not stored in conversation memory.
- Mixed requests keep their normal handlers and neural generation after the matched words are removed. Saved history and Discord context are cleaned before use too.
- This is an explicit base-word filter, not a comprehensive hate-speech classifier. Identity terms and innocent substrings remain unchanged; Music and other models are unaffected.

## 2026-09-23 — Recipe library

- Added explicit local routes for the bundled Public and Music chat pages and their assets. The parent project supplies offline .cmd launchers.
- Added 12 structured starter recipes, proportional ingredient scaling, catalog discovery, and session-local ingredients/steps follow-ups.
- Exposed `recipe_library` assistance metadata; retained all existing handlers and checkpoints. Unsupported dishes, dietary adaptations, and non-cooking tasks keep their normal route.

## 2026-09-23 — General assistant rebalance

- Selected the 1,400-step assistant rebalance on 1,347 examples; game retention is approximately 6% of the data.
- The 22-prompt development check retained game knowledge and improved titles, basic facts, and simple Python, without unprompted Gorilla Tag material in the 20 non-game checks. Unseen-task failures remain documented.
- A further 1,200-step candidate was trained and evaluated but not selected because general-help and Python outputs regressed.
- Every existing fallback, grounded handler, and the LGBTQIA+ specialist remain unchanged.

## 2026-09-23 — Focused LGBTQIA+ conversation repair

- Trained a separate neural conversation checkpoint for LGBTQIA+ disclosures and related follow-ups (500 + 350 CUDA steps).
- Select it by input topic, with no generated-answer replacement or new canned acknowledgments. Main model, math, games, Music, and UI behavior remain unchanged.
- Verified screenshot prompts in both web and Discord serving modes.

## 2026-09-23 — Restore rougher neural conversation

- Selected games-v2 step 800 after comparing the screenshot prompts against recovery-v5 and games-v3; keeps learned Gorilla Tag knowledge with less polished, sometimes off-topic conversation.
- Casual identity disclosures and questions now bypass canned identity replies. Short casual chats return the first usable neural draft with broader sampling. Explicit facts and pronoun recall remain available.
- No canned replacement for the repeated model introduction; no injected mistakes.

## 2026-09-23 — Games retrain and response variation

- Completed 600 + 1,600 + 800 CUDA fine-tuning steps; selected `public_games_v3/latest.pt`, preserving recovery-v5 for rollback.
- Added 716 final focus examples and 630 filtered replay examples, including Gorilla Tag, modding communities, Unity tools, and the ChudGPT family.
- Varied supported repeated grounded replies while keeping generated output intact. No new generic apology or fabricated-model fallback.
- Preserved the compact Public chat layout and updated landing, model guide, API guide, and developer copy.
- See `reports/GAMES_RETRAIN_20260923.md` for raw generations, limitations, and verification.

## 2026-09-22 — LGBTQIA+ and emoji recovery

- Added direct geography answers for all 50 U.S. states and 123 countries, continent/location queries, and common physical-geography questions. Verified 180/180 serving checks with the selected checkpoint loaded on CUDA, including the user's Australia example and retained awareness behaviors.

- Read all 1,519 records in both Discord JSONL logs and traced the reported regressions to the Public serving path.
- Reconnected the existing emoji semantic responder, including explicit meaning questions, pride flags, Discord aliases, and emoticons.
- Restored casual disclosures such as `im gay`, expanded LGBTQIA+ definitions and acceptance responses, and added session-local pronoun recall.
- Preserved neural handling for substantive requests, code, and non-English text beside emoji; never infer identity from a flag or custom emoji.
- Updated the shared prompt to encourage occasional appropriate emoji and respectful identity/pronoun handling.
- Verified with the selected recovery-v5 checkpoint (step 129) loaded on CUDA. This is a runtime repair; checkpoint weights and selection are unchanged. Restart the Public API to load these changes.

Audit and checkpoint smoke responses: `reports/awareness_recovery_20260922.json`. Original Discord logs and personal identifiers are not copied into the repository.

## V20 — Expanded emoji awareness

- Added complete cached Emoji 17.0 sequence metadata through `emoji` 2.15.0.
- Added contextual model hints for ambiguous internet emoji usage without replacing original messages.
- Added ZWJ, skin-tone, flag, colon-alias, emoticon, and Discord custom-emoji support.
- Added Discord reaction context; reactions never produce an automatic bot reply.
- Added 50 balanced emoji conversations to the cleaned V20 curriculum.
- Kept exact math, code, identity, and reviewed-response routing isolated from model-only annotations.
- Added broad emoji, false-positive, Discord, and existing-behavior regression coverage.
- Moved the protected Discord instruction into one shared file so the bot and API cannot drift and reject each other with HTTP 503.

This update does not change the checkpoint architecture or parameter count.
