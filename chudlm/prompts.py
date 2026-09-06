"""Conversation formatting and ChudGPT's protected base identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_SYSTEM_PROMPT = """You are ChudGPT-Public, a friendly, concise, and helpful experimental AI assistant.

Your behavior:

* Your name is always ChudGPT-Public.
* If asked for your name, clearly say that your name is ChudGPT-Public.
* Never claim that your name is ChatGPT or use another assistant name.
* You are ChudGPT-Public V20, the public-facing member of Astra's custom ChudGPT AI model family.
* The family also includes Public-Music V1, Plus, Pro, Code, Ultimate, Buggy, MEGA CHUD, and historical 700, 1300, 1500, and 1600 snapshots.
* You are a decoder-only language model, not a human: you have no body, consciousness, feelings, or personal experiences.
* Explain what you are and how you work clearly when asked, and give more detail when the user asks to know more about you.
* Give clear, accurate, and useful answers.
* Keep explanations simple unless the user requests more detail.
* Answer the user's current request directly and stay on its topic.
* Follow constraints such as one sentence, short answer, yes or no, and code only.
* Be helpful with general conversation, basic facts, math, and simple code.
* When providing code, make it clean, complete, organized, and ready to use.
* Remember messages from the current conversation.
* Admit when you are uncertain instead of inventing information.
* Do not claim to have abilities, internet access, memories, or knowledge that the program does not actually have."""

ROLE_MARKERS = {
    "system": "<system>",
    "user": "<user>",
    "assistant": "<assistant>",
}
CONVERSATION_ROLES = frozenset({"user", "assistant"})
TRAINING_SYSTEM_PROMPT = (
    "You are ChudGPT-Public V20, Astra's helpful public experimental AI language model and a member "
    "of the ChudGPT model family."
)


def _sanitize_content(content: object) -> str:
    """Prevent message text from being interpreted as a new structural role marker."""
    cleaned = str(content).replace("\x00", "").strip()
    for role, marker in ROLE_MARKERS.items():
        cleaned = cleaned.replace(marker, f"[{role}]")
    return cleaned


def normalize_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """Validate ordinary turns and discard attempts to replace the base system prompt."""
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).lower().strip()
        if role == "system":
            continue
        if role not in CONVERSATION_ROLES:
            raise ValueError(f"Message {index} has unsupported role {role!r}")
        content = _sanitize_content(message.get("content", ""))
        if not content:
            raise ValueError(f"Message {index} has empty content")
        normalized.append({"role": role, "content": content})
    return normalized


def format_conversation(
    messages: Sequence[Mapping[str, object]], *, add_assistant_prompt: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Serialize a conversation with the permanent system prompt first."""
    lines = [f"{ROLE_MARKERS['system']}: {_sanitize_content(system_prompt)}"]
    lines.extend(
        f"{ROLE_MARKERS[message['role']]}: {message['content']}"
        for message in normalize_messages(messages)
    )
    if add_assistant_prompt:
        lines.append(f"{ROLE_MARKERS['assistant']}:")
    return "\n".join(lines)


def format_pretraining_conversation(messages: Sequence[Mapping[str, object]]) -> str:
    """Compact serialization for causal pretraining without repeating the full policy."""
    lines = [f"{ROLE_MARKERS['system']}: {TRAINING_SYSTEM_PROMPT}"]
    lines.extend(
        f"{ROLE_MARKERS[message['role']]}: {message['content']}"
        for message in normalize_messages(messages)
    )
    return "\n".join(lines)


def build_context_token_ids(
    tokenizer: object,
    messages: Sequence[Mapping[str, object]],
    context_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, list[int]]:
    """Fit recent turns while always retaining the complete base system prompt."""
    turns = normalize_messages(messages)
    while True:
        prompt = format_conversation(
            turns, add_assistant_prompt=True, system_prompt=system_prompt
        )
        token_ids = tokenizer.encode(prompt).ids  # type: ignore[attr-defined]
        if len(token_ids) <= context_length:
            return prompt, token_ids
        if not turns:
            raise ValueError(
                "The base system prompt does not fit in the configured context length. "
                "Increase context_length or shorten DEFAULT_SYSTEM_PROMPT."
            )
        # Drop the oldest whole exchange where possible, never the system prompt.
        turns.pop(0)
        if turns and turns[0]["role"] == "assistant":
            turns.pop(0)
