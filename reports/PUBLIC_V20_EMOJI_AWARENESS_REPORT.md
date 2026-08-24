# Public V20 expanded emoji-awareness report

## Scope

This release adds a cached local semantic layer around the existing 20,999,184-parameter Public V20 checkpoint. It does not alter the transformer architecture, route through another language model, or claim that the checkpoint has perfect cultural understanding.

## Data source and coverage

- Runtime package: `emoji==2.15.0`
- Unicode coverage exposed by the package: Emoji 17.0
- Recognized fully-qualified and component sequences: 5,225
- Coverage includes multi-codepoint ZWJ sequences, profession/family variants, skin tones, hearts, flags, symbols, and other standard sequences.
- Discord support recognizes `<:name:id>` and `<a:name:id>` while passing only the human-readable name to the model.
- Common colon aliases and classic text emoticons are supported without matching URLs, code fences, Windows paths, mentions, channel/role references, or ordinary punctuation.

## Runtime behavior

The raw user message is preserved. A short private suffix is added only to the model-facing generation context. Exact math, exact instruction/code routing, identity handling, meme facts, Discord context facts, and reviewed-response retrieval all receive clean normalized text, preventing annotation contamination.

Ambiguous emoji can have multiple candidate meanings. Contextual rules distinguish examples such as joking `😭💀`, grief with `😭`, praise with `🔥`, and literal fire/danger. The system prompt tells the neural model that surrounding conversation overrides a fixed emoji definition.

Discord reactions to a recent bot message are retained for the user's next turn. Adding a reaction alone never makes the bot send a message. Custom emoji snowflake IDs are not put into model context.

## Curriculum changes

- Complete cleaned V20 corpus: 6,912 unique conversations
- Focused curriculum: 1,719 conversations
- New balanced emoji conversations: 50
- Authored Public conversations: 719
- Reviewed prose: 700
- Legitimate structured requests: 120
- Reviewed code tasks: 180
- Rejected during rebuild: 2,080 topic/template leaks and 151 malformed or low-quality rows

The new examples cover humor, grief, criticism, praise, gratitude, skepticism, emoji-only turns, literal meanings, skin tones, flags, ZWJ sequences, Discord custom emoji names, aliases, and emoticons. They intentionally avoid making every response emoji-heavy.

## Verification

- Public test suite: 268 passed
- Discord bot test suite: 53 passed (one upstream Python `audioop` deprecation warning)
- Dedicated emoji tests: 58 passed
- Python compile checks: passed

Dedicated regressions cover standard categories, ZWJ families/professions, skin tones, flags, aliases, Discord static/animated custom emoji, ambiguous tone, emoji-only messages, URLs, code, JSON, Markdown, Windows paths, mentions, IDs, numbers, and punctuation.

## Performance benchmark

Measured locally with `benchmark_emoji_awareness.py`:

- Cold database build: 389.282 ms
- Peak traced database-build memory: 1.669 MiB
- Cached preprocessing: 0.06220 ms/message across 12,000 operations
- Sample token count: 116 raw versus 351 annotated (235 additional tokens across 12 deliberately mixed samples)

The cold build happens once per server process. Metadata remains cached after startup. Messages without recognized emoji receive no annotation or token increase. Combined emoji meanings suppress redundant per-symbol explanations, and annotations are capped at four entries.

No Discord network latency was added for emoji lookup; all metadata and processing are local.

## Known limitations

- Emoji meaning depends on culture, community, timing, and context; no static metadata can resolve every usage.
- A small 21M-parameter checkpoint can still misunderstand an unfamiliar meme or produce a weak generative reply.
- Discord custom emoji semantics are inferred only from the emoji name because image content is not downloaded or inspected.
- This release updates runtime semantics and training data but does not retrain checkpoint weights.
