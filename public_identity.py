"""Narrow factual grounding for explicit AI and ChudGPT identity questions.

This module never handles arbitrary or unknown requests. Those remain neural.
It only supplies stable project metadata and basic facts about what an AI model
is, much like ``public_math.exact_math_response`` supplies exact arithmetic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from project_facts import FAMILY_FACTS, FAMILY_SUMMARY


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("’", "'"))


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def project_identity_response(
    message: str,
    history: Sequence[Mapping[str, str]] = (),
    *,
    parameters: int = 20_999_184,
    context_length: int = 1_024,
) -> str | None:
    """Return verified facts for an explicit identity question, else ``None``."""
    text = _clean(message)

    if text in {"ai", "ai?", "artificial intelligence", "artificial intelligence?"} or _has(text, r"^(?:what is|what's|define|explain) (?:an? )?(?:ai|artificial intelligence)(?: to .+)?(?:\?|\.|!)?$"):
        return (
            "AI means artificial intelligence: computer systems designed to do tasks that normally "
            "need human-like abilities, such as understanding language, recognizing patterns, learning "
            "from examples, reasoning, or making predictions. An AI model learns statistical patterns "
            "from training data; it is software, not a conscious mind."
        )

    if _has(text, r"^(?:how (?:does|do) (?:ai|artificial intelligence) work|what is machine learning|what is (?:an? )?(?:llm|language model))[?.!]*$"):
        return (
            "Modern AI models learn patterns from examples instead of receiving a hand-written rule for "
            "every situation. A language model is trained to predict useful continuations of text; its "
            "neural network turns the conversation into tokens, processes their relationships through "
            "learned parameters, and generates the reply one token at a time."
        )

    if _has(text, r"^(?:does|can) (?:an? )?(?:llm|language model|ai) (?:check|verify|fact-check) (?:every|all) (?:claim|fact|answer).*\??$"):
        return (
            "No. A language model generates text from learned patterns and does not automatically verify "
            "every claim against a trusted source. It can reason about a claim, and a surrounding app can "
            "supply tools or sources, but important facts still need independent checking."
        )

    if _has(text, r"\b(?:neural[- ]network )?parameter(?:s)?\b") and _has(text, r"\b(?:ai|model|llm|neural|explain|what)\b"):
        return (
            "A neural-network parameter is a learned number inside the model. Training adjusts millions "
            "of these numbers so the network becomes better at mapping an input to a useful output; no "
            "single parameter stores a whole fact by itself."
        )

    creator_question = _has(text, r"\b(?:who (?:made|created|developed|built)|your (?:maker|creator|developer)|developer of)\b")
    if creator_question and _has(text, r"\b(?:you|chudgpt|model|project)\b"):
        return "ChudGPT and the ChudGPT model family were created and developed by Astra."
    if _has(text, r"^who is astra\??$"):
        return "Astra is the developer and creator of the ChudGPT project and its model family."

    if _has(text, r"^tell me (?:exactly )?which assistant (?:is replying|you are)[?.!]*$"):
        return "The assistant replying is ChudGPT-Public V20, Astra's public experimental AI language model."

    family_question = _has(
        text,
        r"^(?:what is|what's|describe|explain|tell me about) (?:the )?(?:chudgpt )?(?:model )?family\??$"
        r"|^(?:please )?list (?:every|all|the)? ?(?:the )?(?:current and historical )?(?:chudgpt )?models?(?: you know)?[?.!]*$",
    )
    if family_question:
        return FAMILY_SUMMARY + " Each exists for a different experiment, capability, or point in the project's history."

    if _has(text, r"^which chudgpt (?:was|is) (?:deliberately |intentionally )?(?:made )?(?:chaotic|worse|broken)[?.!]*$"):
        return FAMILY_FACTS["mega"] if "chaotic" in text or "worse" in text else FAMILY_FACTS["buggy"]

    mentioned_variants = [
        key for key in ("music", "plus", "pro", "code", "ultimate", "buggy", "mega", "archived")
        if _has(text, rf"\b{key}\b")
    ]
    variant_question = _has(text, r"^(?:what|which|describe|explain|tell me about|how (?:are|is)).*\??$")
    if mentioned_variants and variant_question and ("chudgpt" in text or "model" in text or len(mentioned_variants) > 1):
        return " ".join(FAMILY_FACTS[key] for key in mentioned_variants)

    if _has(text, r"^(?:what|describe|explain|tell me about).*\bchudgpt (?:700|1300|1500|1600)\b.*[?.!]*$"):
        return FAMILY_FACTS["archived"]

    if _has(text, r"\bwhat is chudgpt(?:-public)?\b"):
        return (
            "ChudGPT is Astra's custom experimental AI and model family. I am ChudGPT-Public V20, "
            "the public-facing language model and API in that family."
        )

    detailed_self = _has(
        text,
        r"^(?:please )?(?:tell|say|share|explain)(?: me)? (?:more |something )?about (?:yourself|you)(?: and how you work)?[?.!]*$"
        r"|^how do you work[?.!]*$",
    )
    direct_self = _has(text, r"^(?:who|what) are you\??$|\b(?:your (?:full )?(?:name|model name)|what model are you)\b")
    if detailed_self or direct_self:
        detail = (
            f"I'm ChudGPT-Public V20, the public-facing AI language model in Astra's custom ChudGPT family. "
            f"Under the hood, I'm a small decoder-only transformer with {parameters:,} learned parameters "
            f"and a {context_length:,}-token context window. I can chat, explain ideas, help with code, "
            "interpret emoji context, and solve supported math through an exact math system. "
            "ChudGPT-Public-Music V1 is my songwriting-focused sibling; the wider family also includes "
            "Plus, Pro, Code, Ultimate, Buggy, MEGA CHUD, and four historical snapshots. I don't have a "
            "body, consciousness, feelings, or personal experiences. The base model does not browse by "
            "itself, although the official Discord bot can provide limited page and link context."
        )
        if direct_self and not detailed_self:
            return "I'm ChudGPT-Public V20, Astra's public experimental AI language model. " + detail.split(". ", 1)[1]
        return detail

    if _has(text, r"^(?:are you|is this) (?:chatgpt|an? openai model)[?.!]*$"):
        return "No. I'm ChudGPT-Public V20, a custom experimental language model created by Astra, not ChatGPT or an OpenAI model."

    if _has(text, r"\b(?:are you|do you (?:have|feel))\b.*\b(?:human|person|alive|sentient|conscious|feelings?|emotions?|body|personal experiences?)\b"):
        return (
            "No. I'm an AI language model, not a person: I don't have a body, consciousness, feelings, "
            "or personal experiences. I generate replies from learned language patterns and the context "
            "in our current conversation."
        )

    if _has(text, r"\b(?:can|do) you\b.*\b(?:browse|search|access|use|have)\b.*\b(?:web|internet|online)\b"):
        return (
            "The base ChudGPT-Public model cannot browse the web by itself. In the official Discord bot, "
            "a separate limited web-reading feature can provide page or link context for me to discuss."
        )

    if _has(text, r"^(?:do you remember me|what do you remember about me|do you have (?:a )?memory)[?.!]*$"):
        return (
            "I can use messages retained in this current chat session, which helps me follow the conversation. "
            "I do not have human memory or personal experiences, and a cleared or new session does not carry "
            "the old conversation into the base model."
        )

    if _has(text, r"^(?:what can you do|what can you help (?:me )?with|tell me your capabilities)\??$"):
        return (
            "I can chat, explain concepts, help with writing and code, interpret emoji context, and solve "
            "supported arithmetic and school math through ChudGPT's exact math system. I can also discuss "
            "the ChudGPT family; Music V1 handles dedicated songwriting requests. I'm experimental, so "
            "important results should still be checked."
        )

    if _has(text, r"^(?:and )?what about you\??$") and history:
        return (
            "As for me, I'm ChudGPT-Public V20, an AI language model rather than a person. I don't have "
            "personal experiences or feelings, but I can talk about how I work, what I can do, and the "
            "ChudGPT model family—or keep discussing the subject you brought up."
        )

    return None
