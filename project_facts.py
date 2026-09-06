"""Small, stable metadata source for ChudGPT and AI identity grounding.

This is deliberately limited to project identity. It is not a general answer
database and is consulted only for explicit AI, identity, capability, creator,
or ChudGPT-family questions.
"""

from __future__ import annotations

FAMILY_FACTS: dict[str, str] = {
    "public": "ChudGPT-Public is the current public experimental ChudGPT language model and API.",
    "music": "ChudGPT-Public-Music V1 is the original-songwriting member of the public family. It generates lyrics, hooks, titles, styles, rewrites, and lyric mashups.",
    "plus": "ChudGPT Plus is a conversational ChudGPT model and serving profile.",
    "pro": "ChudGPT Pro is a general-use profile built from Plus with a larger runtime conversation window and stronger recovery.",
    "code": "ChudGPT Code is the programming-focused ChudGPT experience.",
    "ultimate": "ChudGPT Ultimate is an older experimental ChudGPT version.",
    "buggy": "Buggy ChudGPT intentionally serves an early, broken checkpoint for chaotic conversations.",
    "mega": "MEGA CHUD is a deliberately chaotic experimental ChudGPT model that is worse than the other variants.",
    "archived": "The 700, 1300, 1500, and 1600 checkpoints are historical ChudGPT training snapshots.",
}

PUBLIC_IDENTITY = (
    "ChudGPT-Public is the public-facing model and serving profile in the ChudGPT family. "
    "It powers the public chat and API using a small custom decoder-only transformer."
)

FAMILY_SUMMARY = (
    "ChudGPT is Astra's overall project and model family. Its twelve served model APIs are "
    "ChudGPT-Public V20, ChudGPT-Public-Music V1, Plus, Pro, Code, Ultimate, intentionally "
    "broken Buggy, deliberately chaotic MEGA CHUD, and the historical 700, 1300, 1500, and "
    "1600 training snapshots."
)
