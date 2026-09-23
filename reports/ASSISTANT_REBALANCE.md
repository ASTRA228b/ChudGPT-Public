# Assistant rebalance — September 23, 2026

## Selected change

The main model now selects `checkpoints/public_assistant_rebalance/latest.pt`, step 1,400. Its base is the previous games-v2 step-800 checkpoint. All serving code, fallbacks, grounded handlers, casual sampling, and the separate LGBTQIA+ checkpoint remain unchanged. This changes learned weights rather than intercepting bad game answers with canned text. The previous main checkpoint remains archived for rollback.

The first training pass used 1,347 original/reviewed conversations: titles, extraction, transformations, basic facts, everyday help, reasoning, Python/C#, existing replay, and 77 game-retention examples (5.7%). No Gorilla Tag/Another Axiom answers appear outside the game-retention category. Every three-word title training answer has three words. Response-only CUDA fine-tuning completed 1,400 steps with final validation loss 0.6832.

A broader second candidate completed 1,200 more steps on 3,904 examples (loss 0.4817). It improved some title responses but regressed math-help and Python replies. It is retained locally for analysis, not selected. Lower validation loss was not treated as proof of better assistance.

## Actual generation checks

`evaluate_assistant_rebalance.py` compares 22 prompts using normal serving generation at temperature 0.6 and 120 new tokens, with all existing handlers present. The evaluated task replies used neural generation. The old checkpoint produced unprompted Gorilla Tag material in two of the 20 non-game prompts; the selected candidate produced none in this sample. This narrow substring count measures observed topic drift, not correctness.

| Request | Selected candidate's observed reply |
|---|---|
| Invent a three-word title for a game about ducks. | Duck Moonlight Adventure |
| Suggest an original song title about a confused robot. | Robot Melody |
| Help me with math | Asked for the problem to work through |
| What is a verb? | Explained an action or state |
| Give a Python function that doubles a number. | `return number * 2` inside a function |
| Sort 8, 3, 6 | 3, 6, 8 |

Both Gorilla Tag and modding-community questions retained relevant answers. These are actual outputs, not runtime templates.

## Limits

The candidate is still a small, unreliable assistant. Unseen extraction returned Tokyo instead of the supplied Madrid; uppercase conversion, past-tense rewriting, and some creative constraints failed. One summary prompt exhausted the existing generation checks and raised a generation error. The robot title is generic and does not clearly express confusion. No handler was removed or added to hide these failures.

Five evaluation prompts exactly overlap training (math help, ice floating, verb definition, weather/climate, Gorilla Tag), and many other prompts are related paraphrases. Results informed iteration, so this is a development comparison, not an independent benchmark. Validation splits also contain related examples. Responses may vary between runs.

## Reproduction and evidence

Run `build_assistant_rebalance.py`, then `fine_tune.py --config configs/finetune_assistant_rebalance.yaml --device cuda`. The optional unselected continuation uses `build_assistant_transfer.py` and `configs/finetune_assistant_transfer.yaml`. Run the evaluator with `--checkpoint` and `--output` to inspect a checkpoint.

Local reports: `assistant_baseline.json`, `assistant_candidate.json`, `assistant_transfer_candidate.json`, `assistant_dataset_audit.json`, and training/build/test logs. Checkpoint binaries and detailed JSON/log files retain the repository's existing Git exclusions. No raw Discord data or personal identifiers were added.

## Activation checks

All 654 repository tests passed with the new checkpoint selection. After restarting Public, both screenshot prompts and math help returned neural task-related answers. Live math, geography, emoji, greetings, and project-identity checks retained their respective assistance reasons; the trans disclosure used the unchanged LGBTQIA+ specialist at step 350. The server and handler files were verified unchanged from commit `fd531a1`. The landing site's model details were updated without changing the chat layout.
