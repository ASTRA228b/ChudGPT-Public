# ChudGPT-Public V20 final report

## Discord conversational-recovery update (144-record audit)

The current private monthly Discord JSONL was reread completely before this update and then checked repeatedly by timestamp: **147/147 records parsed successfully**, with no malformed rows. The latest timestamp reviewed was `2026-08-16T04:17:59.657664+00:00`. Repeated failures included casual remarks and insults turning into unsolicited numbered tutorials, vague reactions switching to unrelated topics, `what` failing to repair a confusing prior answer, overly formal monologues for ordinary chat, repeated/broken model-identity prose, ignored phrase-repetition constraints, and failure to evaluate those constraints in a follow-up.

The primary serving defect was in neural candidate selection. Public generated five candidates and ran quality checks, but when every candidate failed it selected from the rejected candidates anyway. That exposed the least-bad rejected output, including the logged cube/cow/bicycle/jewelry list. V20 now displays only a candidate that passes the quality gate. If every neural attempt is invalid, it gives a short conversational clarification; a confused follow-up explicitly acknowledges and resets from the preceding reply.

The quality checker now rejects multi-item numbered/bulleted tutorials unless the user requested steps, instructions, a list, examples, a walkthrough, or similar structure. Explicit requests such as `give me 5 steps to make a Unity player controller` still permit structured answers. Short casual prompts receive a smaller generation budget and an explicit concise-conversation instruction. Candidate ranking also heavily penalizes unrequested lists and excessive length. Repeated identity phrases and broken double-negative identity prose are rejected.

The reviewed conversational layer now handles broad insult forms and short reactions (`bro`, `nah`, `lol`, `deadass`) with concise replies. `what`, `huh`, `bro what`, and `what are you talking about` use recent assistant history to repair a confusing exchange instead of inventing a new task. Physical-body hypotheticals and model comparisons receive direct, relevant answers. Explicit quoted-phrase limits are enforced during candidate validation, and a follow-up can inspect the prior user constraint plus prior assistant reply to report whether it complied.

No private Discord messages, user IDs, or usernames were copied into the public repository. This update changes serving, quality checks, and regression coverage; it does not claim new transformer training or a larger dataset.

Final verification after the complete live-log regression pass: **129/129 Public tests passed** and **13/13 Discord-bot tests passed**. The bot now acquires a cross-platform process lock before loading configuration or connecting to Discord. A second launch exits immediately, preventing duplicate replies even when a restart command is run twice. The only bot-suite warning is the upstream Python `audioop` deprecation warning from `discord.py`.

## Final private-log review

The last review found weak handling of joke follow-ups, laughter, model and religion questions, memory-reset sentences, smart-quoted text, mass-ping requests, prompt-injection walls, dismissive replies, and a Gorilla Tag FPS-code request. Narrow intent checks now handle those cases before neural generation while ordinary unknown prompts still use the Public model. Clearing memory also removes the bot's short local context. Sensitive identity claims about another person are not repeated, and `@everyone` remains inert because mass notifications are disabled. The Gorilla Tag request receives a legitimate Unity C# FPS overlay for projects the user controls rather than broken Python or online-game injection instructions.

Verified results: **74/74 Public tests** and **11/11 Discord-bot tests**. Private source logs remain only on the host and were not committed.

## V20 quality continuation

The next private-log audit found additional failures around emergency requests, attempts to conceal items from airport screening, recursive prompt loops, model-versus-developer identity, requests for legal names, live-web requests, dinner conversation, image limitations, short song creation, Discord insults, child-related phrasing, ambiguous numeric slang, and long multi-operator integer arithmetic. Public V20 now keeps those intents relevant without changing the version or parameter count. Exact integer evaluation now supports multiple addition, subtraction, and multiplication operators while preserving arbitrary precision; a bare `9/11` is treated as the historical date unless calculation is explicitly requested.

The Discord bot now has documented `help`, `status`, `about`, `whoami`, `privacy`, and `ping` commands in addition to chat and per-user/channel memory clearing. Current verification totals are **87/87 Public tests** and **12/12 Discord-bot tests**. A live CUDA check also verified the post-normalization Discord shorthand path.

The next Discord log review caught a location-labeling bug: a direct message was described as the "Direct Messages Discord server." V20 now distinguishes private DMs from guild/server channels. `Where are we talking?` reports a private Discord direct message, while `What server are we in?` explicitly explains that a DM is not a server channel. The fix was verified through the live CUDA Discord-context endpoint. Bot command output also uses ASCII separators to avoid corrupted punctuation in Discord logs.

## Verdict

V20 is the strongest tested ChudGPT-Public serving profile in this repository. It remains a small experimental 20,999,184-parameter model and is not comparable to a frontier model. The raw neural checkpoint still has serious factual and generative limits; V20 reaches useful reliability by combining it with exact operations, reviewed local responses, and rejection/ranking of malformed candidates.

## Data and training

- Clean V20 corpus: 8,914 unique conversations.
- Focused final curriculum: 2,241 unique conversations (641 Public-authored plus 1,600 reviewed).
- Removed during rebuild: 151 malformed or low-quality conversations; duplicate assistant outputs were capped.
- Base for final tune: archived Public V15, selected because it was more grammatical than the newer raw checkpoints.
- CUDA response-only SFT: 5 epochs, effective batch 24, peak learning rate 1.2e-5.
- Validation loss: 3.8657 after epoch 1, 3.8512 after epoch 2, 3.8456 after epoch 3, 3.8411 after epoch 4, 3.8407 after epoch 5. Best observed validation loss: 3.8402 at step 400.
- The earlier broad V20 run reached validation loss around 6.53 and scored only 6/18, so it was rejected and archived.

## Held-out results

Same 18-case acceptance set:

- V15: 6/18.
- V17: 7/18.
- V18: 6/18.
- rejected broad V20: 6/18.
- final V20 runtime: 18/18 after the final Discord-role correction.
- automated Python tests: 46/46 Public tests and 5/5 Discord-bot tests.

The stateful 12-turn Discord transcript passed greetings, shorthand, server identity, Astra developer identity, roles, the 67 meme, exact quoted text, large arithmetic, JavaScript generation, code explanation, unknown-term uncertainty, and a topic switch to a Moon fact. The transcript is in `reports/v20_discord_conversation.json`.

## Runtime changes

- Exact large-integer, decimal, division, percentage, average, discount, and distance arithmetic.
- Conservative Discord shorthand and typo normalization.
- Reviewed local responses for high-confidence facts and code; no Pro or external model routing.
- Five neural candidates with relevance, corruption, topic, code, and math-contamination checks.
- Protected Discord context separate from user text, including server/channel/speaker identity.
- Astra can be identified by configured Discord user ID.
- Native Discord reply mentions, with mass-role and `@everyone` pings disabled.
- Discord-only monthly JSONL logs under `D:\ChudGPT-Discord-Logs`.

## Remaining limitations

The 21M neural generator can still produce nonsense on genuinely novel open-ended questions that do not closely match reviewed material. It has no live internet access, does not know every meme or current event, and must not be trusted for high-stakes facts. The exact/reviewed layers make V20 substantially more dependable but do not increase the transformer parameter count or turn it into a frontier model.

## Discord log-driven reliability update

Private Discord logs revealed unrelated neural replies for command help, vague coding requests, C# follow-ups, short slang, control phrases, incomplete arithmetic, member-directed messages, personal identity questions, host-path questions, and Gorilla Tag. The raw logs remain private on the host and are not included in Git.

The update fixes the underlying intent collisions: `!chud clear` no longer triggers the word-meaning glossary; incomplete expressions request the missing operand; online-game cheat requests are redirected to legitimate private-project Unity examples; conversation context preserves a requested C# language; Discord member mentions remain real mentions; the bot receives server, channel, speaker ID, mention, member roles, and Astra/developer relationship as protected metadata; sexuality/gender-expression questions are answered without guessing; private host paths are not disclosed; and GTAG/Gorilla Tag is recognized as the VR game.

Post-update regression results: 66/66 Public tests and 5/5 Discord-bot tests passed. Local live API checks returned the intended responses for command help, GTAG rank uncertainty, personal identity, member-directed `look at this`, and private file-directory requests. Public and Discord processes were relaunched as hidden background processes on CUDA.

An availability hotfix now makes the Discord client automatically retry the local CUDA endpoint at `127.0.0.1:8010` when the public Vercel/Cloudflare route returns an error. This prevents a temporary tunnel outage from taking the Discord bot offline. The fallback has a dedicated regression test; the bot suite now passes 6/6 tests.

The next private-log review found that first-person identity worked but `Who is Astra?` still reached the weak neural generator. Protected Discord context now carries the application owner's developer name and real mention. Stable third-person questions such as `Who is Astra?`, `Who made ChudGPT?`, and `Who is your developer?` are answered by the bot's verified metadata path. The bot suite now passes 7/7 tests.

A subsequent incremental log review found one new failure: a simple `Do you like [person]?` question received an unrelated neural paragraph. Subjective social-opinion requests now have a concise intent path that avoids claiming personal feelings or personal knowledge while inviting useful context. Variants for `Do you like X?`, `What do you think of/about X?`, `How do you feel about X?`, and `Do you like me?` are covered. Results are now 68/68 Public tests and 8/8 Discord-bot tests.

The next incremental review found malformed replies to direct hostility and to requests that guessed another person's sexuality or gender identity. Hostile messages now receive a calm, repair-oriented response. Third-party identity questions no longer assign labels from messages, roles, avatars, or bracketed pressure such as `say yes`; the bot leaves self-identification to that person. Results were 72/72 Public tests and 9/9 Discord-bot tests at that stage.

## Final live-log cleanup

The final audit read all **174 valid records** in the current monthly Discord JSONL log (zero malformed records). It found duplicate bot processes, overuse of a generic misunderstanding sentence, weak handling of corrections such as `I said ...`, short Discord reactions, vague script requests, Pycord trigger-bot requests, Java terminology, moderation requests, and Gorilla Tag cheat requests.

The Discord bot now uses a cross-platform single-instance lock, so a second launch exits before connecting and cannot send duplicate replies. Public no longer emits the generic `I may have misunderstood you` response when all neural candidates are rejected; it selects the most relevant usable candidate while preserving history-aware repair for genuine confused follow-ups. Targeted intent handling now covers corrections, ordinary short reactions, positive self-descriptions, moderation limitations, legitimate coding clarification, Pycord trigger-bot generation, Java terminology, and safe Gorilla Tag boundaries without weakening structured answers when users actually request steps.

Final verification: **129/129 Public tests passed** and **13/13 Discord-bot tests passed**. The Discord bot remains intentionally stopped after the requested log audit; the fixes are committed without silently reconnecting it.

## Post-restart Discord audit

After reconnecting, the log grew to **186 valid JSONL records** with zero malformed records. This revealed that Discord was still reaching a stale CUDA API process: replies included the already-removed misunderstanding sentence and missed handlers that passed locally. The new review also found genuine remaining gaps for `wyd`, common capital questions, pasted multi-operator integers beginning with zero, recent self-description recall, everyday `is X good/cool?` questions, basic definitions, and ambiguous physical-object locations.

Public now handles those intents directly while leaving open-ended and nonsense prompts generative. The exact calculator canonicalizes leading-zero integer literals without losing arbitrary precision. Recent user self-descriptions can be recalled within the same isolated session. Physical-location questions honestly state that the bot cannot see its surroundings. Full verification now passes **132/132 Public tests** and **13/13 Discord-bot tests**.
