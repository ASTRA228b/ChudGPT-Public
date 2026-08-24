# ChudGPT-Public changelog

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
