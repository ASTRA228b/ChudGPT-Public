# ChudGPT-Public-Music V1 relevance update

## Outcome

Music V1 now cleans malformed title/style metadata, rejects weak topic matches more aggressively, validates lyric sections, and suppresses lyric lines proven by historical Music logs to be overrepresented. The active model remains the existing Music V1 checkpoint because the newly trained correction candidate produced worse English during generation tests.

## Reported defects fixed

- Removed the recurring style ending `with a clear pulse and a slightly unwise finale`.
- Normalized malformed labels including `static Title:` and `Stylas:`.
- Prevented unknown bracket labels from being presented as valid song sections.
- Ensured an Outro cannot appear before later lyric sections.
- Added a strong penalty when an ordinary song request returns only a title and style.
- Added topic-aware candidate scoring for subjects such as water, coding, computers, weather, food, space, robots, and ChudGPT itself.
- Added log-derived filtering for frequently repeated lyric lines, including the reported hallway/red-light/fan/spoon pattern. The filter learns which lines are overrepresented from local generation logs; it is not a hard-coded song response.

## Root cause

The checkpoint learned several high-frequency synthetic lyric frames. Sampling frequently selected those fluent-looking frames even when their nouns did not match the request. The runtime scorer rewarded formatting and length more than topical overlap, and the structure parser accepted misspelled metadata labels. Together, this allowed an irrelevant but song-shaped candidate to win.

## Dataset and training analysis

- Rebuilt Music corpus: 4,320 assistant outputs.
- Unique titles: 506.
- Unique styles: 1,252.
- Malformed structured examples after rebuilding: 0.
- Historical structured runtime log: 201 outputs; 63.18% failed the new relevance check.
- The exact repeated style suffix appeared 54 times in the historical log.
- A correction run used 3,240 training and 360 validation conversations for 600 CUDA steps.
- Best correction validation loss: 3.0165 at step 600.

The candidate checkpoint was not promoted. Despite lower validation loss, its ten-topic generation benchmark still failed 6/10 relevance checks and its grammar was visibly worse than the active checkpoint. This demonstrates that validation loss alone is not a sufficient promotion criterion.

## Runtime and observability changes

- Music requests now record their source as `api`, `webclient`, or `discord`.
- Dedicated records include prompt, reply, requested/actual model, scores, topic-relevance score, structure corrections, generation time, and checkpoint information.
- Music web and Discord clients identify their request source.
- Added analyzer metrics for malformed structure, duplicate titles/styles, suspicious nouns, common endings, topic failures, and bad examples.
- Private prompts and generated lyrics remain ignored by Git.

## Benchmark

The ten-topic benchmark covers water, ChudGPT, the model itself, a broken keyboard, thunderstorm, microwave, space, nothing, coding at 3 AM, and a dancing robot.

Active checkpoint after scoring/validation changes:

- Unique-title ratio: 1.00.
- Unique-style ratio: 1.00.
- Malformed structure: 0.
- Internal repeated-line findings: 0.
- Topic-relevance failures: 5/10.

This is a real improvement in structure and duplicate control, but the small checkpoint still produces weak English and imperfect subject adherence. Music V1 remains experimental; the report does not claim the neural model is fully corrected.

## Regression tests

- Public/model/runtime suite: **295 passed**.
- Discord bot suite: **83 passed**.
- New coverage verifies topic scoring, ChudGPT-self interpretation, label repair, section ordering, source logging, metadata-only rejection, exact suffix removal, and log-derived repeated-line removal.

## Known weaknesses

- The 21M-parameter checkpoint is still too small and undertrained for consistently coherent full songs.
- Semantic topic matching is deliberately lightweight and misses some indirect relationships.
- Historical repetition can only be suppressed after it appears often enough in the local Music log.
- A future checkpoint should use a more diverse, human-reviewed lyric corpus and a held-out human preference evaluation before promotion.
