# ChudGPT-Public V20 quality and reliability report

Date: 2026-08-17

## Evidence reviewed

- Fully parsed all 691 valid records in `D:\ChudGPT-Discord-Logs\discord-2026-08.jsonl` (zero JSON parse failures).
- 516 unique user prompts and 507 unique bot replies.
- The audit found 76 generic clarification/apology replies, 59 long replies to prompts of four words or fewer, 25 likely code-topic leaks, 15 unsolicited numbered/list responses, and five explicit recursive language-model loops.
- Representative failures included an unsolicited six-step cube/cow/bicycle list after a casual insult, Python code after “look at this,” arithmetic after nonsense, repeated self-definitions, and a Python/C#/HTML list after “I hope thyself perish.”

## Root causes

1. The neural selector rejected bad candidates, but a later non-list branch could return a candidate rejected for every other reason. The retry path likewise checked only for a generic uncertainty phrase.
2. The focused dataset selected 1,600 reviewed rows by hash order. OpenAssistant-style instructional prose dominated that sample.
3. Relevance checks used only the current message, weakening short reactions such as “what,” “why,” and “bro.”
4. Code validation did not verify that its language matched the request.
5. Discord retained recent user messages but not recent bot replies in its explicit metadata.

## Changes

- Removed the selector bypass: every returned neural retry must pass the complete quality gate.
- Added context-aware relevance for short follow-ups using the previous user and assistant turns.
- Added wrong/mixed language detection for Python, C#, Unity, JavaScript, TypeScript, Java, Rust, and C++.
- Added checks for exact item counts, one sentence, yes/no-only, code-only, repeated clauses, low-diversity loops, recursive self-definitions, unsolicited years/currency, and broader tutorial leads.
- Increased repetition penalties and strengthened the matching server/Discord system instruction.
- Discord now supplies recent bot replies and applies translated English intent to social/code handling before translating back.
- Rebuilt the broad corpus to 6,862 unique conversations. Removed 2,080 topic/template leaks and 151 malformed/low-quality rows.
- Rebalanced the 1,669-row focused curriculum: 669 authored, 700 reviewed prose, 120 legitimate structured requests, and 180 reviewed code tasks. Added reviewed conversational examples for Spanish, Portuguese, French, German, Russian, and Japanese.
- Fine-tuned a new checkpoint for 250 optimizer steps on CUDA. Validation loss improved from 3.6359 to 3.6326. The prior V20 checkpoint remains archived.

## Verification

- Public tests: 210 passed (20 model/serving plus 190 V20 behavior tests).
- Discord-bot tests: 32 passed; one third-party `audioop` deprecation warning remains.
- Standard live serving benchmark: 18/18 before and 18/18 after.
- Raw inspection still shows the expected limits of a 20,999,184-parameter model. The serving quality layer remains necessary.

## Remaining limitations

- The tiny neural checkpoint can still generate nonsense on novel open-ended input; rejection/resampling cannot add frontier-scale reasoning.
- Translation depends on the configured provider and network.
- The model has no live web knowledge and is unsuitable for high-stakes decisions.
