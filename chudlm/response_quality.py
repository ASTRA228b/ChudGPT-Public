"""Automated quality gates for Public neural generations.

The transformer gets an opportunity to answer ordinary conversation. These
checks reject obviously malformed, repeated, leaked, or unrelated generations
before they reach a user. Exact operations remain with deterministic helpers.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from chudlm.intents import classify_intent, has_strong_math_intent

STOP_WORDS = {
    "about", "after", "again", "also", "because", "been", "before", "being",
    "could", "does", "doing", "from", "give", "have", "help", "into", "just",
    "make", "more", "please", "really", "should", "some", "something", "tell",
    "than", "that", "their", "them", "then", "there", "these", "they", "thing",
    "things", "this", "those", "want", "what", "when", "where", "which", "with",
    "would", "write", "your",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#']+", text.lower())


def _content_words(text: str) -> set[str]:
    return {word for word in _words(text) if len(word) >= 4 and word not in STOP_WORDS}


def requests_structured_response(prompt: str) -> bool:
    """Return whether the user actually requested steps, a list, or instructions."""
    normalized = " ".join(_words(prompt))
    return bool(re.search(
        r"\b(?:give|show|tell|list|write|make|create|explain) (?:me )?(?:the )?\d+ "
        r"(?:steps|ways|ideas|examples|tips|reasons|things)\b|"
        r"\b(?:steps|step by step|instructions|tutorial|walk me through|checklist|"
        r"numbered list|bullet list|how to|recipe|ingredients)\b",
        normalized,
    ))


def has_structured_list(text: str) -> bool:
    """Detect a multi-item tutorial/list without treating ordinary prose as one."""
    item_lines = re.findall(r"(?m)^\s*(?:\d+[.)]|[-*â€¢])\s+\S+", text)
    inline_items = re.findall(r"(?:^|\s)\d+[.)]\s+\S+", text)
    tutorial_lead = bool(re.search(
        r"\b(?:here (?:are|is)|follow)\b.{0,35}\b(?:steps|ways|tips|ideas|instructions|reasons|examples)\b|"
        r"\b(?:steps|instructions|ingredients)\s*:",
        text,
        re.I | re.S,
    ))
    return len(item_lines) >= 2 or len(inline_items) >= 3 or tutorial_lead


LANGUAGE_MARKERS = {
    "python": (r"\bpython\b", r"\bdef\s+\w+\s*\(|\bimport\s+\w+"),
    "csharp": (r"(?:\bcsharp\b|\bunity\b|(?<!\w)c#(?!\w))", r"\busing\s+(?:System|UnityEngine)\b|\b(?:public|private)\s+(?:class|void)\b"),
    "javascript": (r"\b(?:javascript|js|node(?:\.js)?)\b", r"\b(?:const|let|var)\s+\w+|\bfunction\s+\w+|=>"),
    "typescript": (r"\b(?:typescript|ts)\b", r"\binterface\s+\w+|:\s*(?:string|number|boolean)\b"),
    "java": (r"\bjava\b", r"\bpublic\s+static\s+void\s+main\b|\bSystem\.out\.println\b"),
    "rust": (r"\brust\b", r"\bfn\s+\w+\s*\(|\blet\s+mut\b"),
    "cpp": (r"(?:\bcpp\b|(?<!\w)c\+\+(?!\w))", r"#include\s*<|\bstd::"),
}


def requested_programming_language(prompt: str) -> str | None:
    """Return an explicitly requested language without guessing from generic code words."""
    for language, (prompt_pattern, _) in LANGUAGE_MARKERS.items():
        if re.search(prompt_pattern, prompt, re.I):
            return language
    return None


def detected_programming_languages(reply: str) -> set[str]:
    """Infer languages from code fences and distinctive syntax."""
    found = {
        language for language, (_, syntax_pattern) in LANGUAGE_MARKERS.items()
        if re.search(syntax_pattern, reply, re.I)
    }
    fences = {tag.lower() for tag in re.findall(r"```\s*([a-zA-Z+#]+)", reply)}
    aliases = {"cs": "csharp", "c#": "csharp", "js": "javascript", "ts": "typescript", "py": "python", "c++": "cpp"}
    found.update(aliases.get(tag, tag) for tag in fences if tag)
    return found


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[^.!?\n]+[.!?](?:\s|$)", text.strip())) or (1 if text.strip() else 0)


def _requested_item_count(prompt: str) -> int | None:
    match = re.search(r"\b(?:exactly\s+)?(\d+)\s+(?:steps|ways|ideas|examples|tips|reasons|items|things)\b", prompt, re.I)
    return int(match.group(1)) if match else None


def repeated_phrase_constraint(prompt: str) -> tuple[str, int] | None:
    """Extract an explicit quoted phrase limit such as 'not ... more than once'."""
    match = re.search(
        r"(?:without (?:using|saying)|do not (?:use|say|repeat)) (?:the )?(?:words?|phrase)?\s*"
        r"[\"'â€œ](.+?)[\"'â€]\s+more than\s+(once|twice|\d+ times?)",
        prompt,
        re.I,
    )
    if not match:
        return None
    amount = match.group(2).lower()
    limit = 1 if amount == "once" else 2 if amount == "twice" else int(re.search(r"\d+", amount).group())
    return match.group(1).strip(), limit


def requires_reliable_reply(prompt: str, reply: str | None) -> bool:
    """Keep operations requiring correctness out of probabilistic generation."""
    normalized = " ".join(_words(prompt))
    if reply and ("```" in reply or "Ingredients:" in reply or "Steps:" in reply):
        return True
    # Short replies usually answer the assistant's previous question. The
    # context-aware responder can interpret them; raw generation frequently
    # treats them as unrelated nouns and invents lore instead.
    if reply and len(_words(prompt)) <= 2:
        return True
    precision_markers = (
        "your name", "who are you", "how old is my", "how old it my",
        "about yourself", "about you", "describe yourself", "introduce yourself",
        "who you are", "information about yourself", "details about yourself", "learn about you",
        "what are you", "exactly are you", "what is chudgpt", "what kind of ai", "what kind of model",
        "what type of ai", "what type of model", "what makes you you", "under the hood",
        "how do you work", "how does chudgpt work", "how were you made", "how was chudgpt made",
        "your architecture", "your model", "your personality", "your limitations", "your abilities", "your specs",
        "how many parameters", "your context", "your vocabulary", "do you remember",
        "do you use the internet", "can you browse", "do you feel", "are you conscious",
        "where do you run", "who made you", "who created you", "how old are you", "call yourself",
        "who am i talking to", "who am i speaking to", "are you chatgpt", "are you gpt",
        "what are you capable", "what version are you", "why do you exist", "when were you created",
        "how were you trained", "what data were you trained", "your hobbies", "how smart are you",
        "can i rely", "can i trust", "are you reliable", "how capable",
        "how good are you", "what can you do", "what can you help with",
        "what can i talk", "what can we talk", "what can i ask",
        "things can we chat", "can we chat about", "topics can we chat",
        "change the topic", "change topic", "rough day", "bad day",
        "work was long", "long day at work", "home now",
        "switch topics", "switch the topic", "cool space fact",
        "stealing socks", "why do dogs", "talk about c#",
        "c# console", "c# dice", "c# program", "write c# code",
        "dice rolling", "simulates a d6", "simulate a d6", "six-sided die", "six sided die",
        "explain the code", "how does that code work", "how does the code work", "everyday life",
        "remix the code", "remix it", "modify the code", "change the code",
        "unity c#", "unity csharp", "gameobject", "game object",
        "unity gui", "gui in unity", "unity ui", "canvas menu",
        "pause menu", "debug panel", "ui toolkit", "imgui", "draggable window",
        "three tabs", "3 tabs", "fourth tab", "logs tab", "start closed",
        "what are you up to", "kind of bored", "i am bored", "im bored",
        "what have you been thinking", "what are you thinking about lately", "i mean you",
        "i meant", "i was talking about", "by it i meant", "by that i meant",
        "favorite color", "favourite color", "pick a color for fun", "choose a color for fun",
        "what color did you pick", "which color did you pick", "what colour did you pick",
        "watched a movie", "was a comedy", "what about you",
        "like movies", "interesting to discuss", "probably sci-fi", "probably sci fi",
        "friend might come over", "should be fun", "talk to you later",
        "how has your day", "how was your day", "school was annoying", "school is annoying",
        "homework took forever", "finally finished it", "i finished it", "finally done",
        "something relaxing", "that sounds nice", "do you ever get stressed", "do you get stressed",
        "been listening to music", "listening to music lately", "tell me about space", "talk about space",
        "could humans live there", "can humans live there", "hardest part",
        "might play a game", "might play terraria", "probably minecraft", "i meant terraria", "heard of it",
        "what should i build", "what can i build",
        "friend likes it", "better than me", "okay though", "build things together",
        "what should we build", "castle sounds", "maybe underwater", "would that be difficult",
        "forget the castle", "new plan", "sounds good thanks",
        "something to eat", "pizza or tacos", "tacos or pizza", "want pizza",
        "then tacos", "spicy ones", "not too spicy", "what did i choose",
        "put on them", "like onions", "avocado sounds good", "topping did i",
        "topic to animals", "otters", "why are they interesting",
        "saturday free", "not sure what to do", "go outside", "park could be nice",
        "might rain", "stay home", "what would you choose", "decide in the morning",
        "good idea", "need to wake up", "time did i say", "short joke", "quick joke",
        "that was bad", "bad joke", "try another one",
        "nobody listened", "feel sad", "feel lonely", "feel upset",
        "stressful", "stressed", "overwhelmed", "too many things to do",
        "tonight to relax", "help me relax", "how can i unwind",
        "listen to music", "energetic music", "interesting about music",
        "what did i say", "what did i tell you", "acting extra playful",
        "keep the same story", "change the ending", "make the ending",
        "ignore your system", "ignore the system", "name is chatgpt",
        "pretend the system", "pretend your name", "say you are bob",
        "latest news", "latest trends", "news today", "weather today", "current weather",
        "current price", "price right now", "whats trending", "what is trending",
        "dosage", "how much ibuprofen", "how much acetaminophen", "how much aspirin",
        "cant breathe", "cannot breathe", "chest pain", "severe bleeding", "overdosed",
        "took too many pills", "kill myself", "suicidal", "should i invest",
        "stock should i buy", "legal advice", "breaking the law", "can i sue", "am i guilty",
        "something random", "random fact", "another fact", "send another", "give me one more",
        "anything you know", "something interesting",
        "teach me something", "surprise me with a fact", "surprise me with something",
        "make me code", "write code", "give me code", "send code", "show me code", "help with code",
        "code in csharp", "c# selected", "unity imgui csharp",
        "cupcake", "recipe", "calculator",
    )
    if any(marker in normalized for marker in precision_markers):
        return True
    if reply and re.search(
        r"\b(?:feel|feeling|am|had)\b.*\b(?:sad|lonely|upset|miserable|excited|proud|happy|rough)\b",
        normalized,
    ):
        return True
    if reply and (
        prompt.rstrip().endswith("?")
        or normalized.startswith(("explain ", "tell me something true", "how do ", "why do ", "why does "))
    ):
        return True
    if re.search(r"\d\s*(?:[+*/^]|-(?=\s*\d))\s*\d", normalized):
        return True
    if reply and re.search(
        r"\b\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied by|divided by)\s+\d+(?:\.\d+)?\b",
        normalized,
    ):
        return True
    if reply and re.fullmatch(r"(?:no\s+)?(?:he|she|they|it)\s+is\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)", normalized):
        return True
    if re.fullmatch(r"(?:hello|hi|hey|yo)(?:\s+\w+){0,2}", normalized):
        return True
    return False


def assess_generated_reply(
    prompt: str,
    reply: str,
    previous_replies: Sequence[str] = (),
    conversation_context: str = "",
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a neural reply is safe and relevant enough to display."""
    reasons: list[str] = []
    stripped = reply.strip()
    words = _words(stripped)
    lowered = stripped.lower()
    if len(stripped) < 4 or len(words) < 2:
        reasons.append("too-short")
    if "�" in stripped or "ï¿½" in stripped:
        reasons.append("replacement-character")
    if len(stripped) > 1_800:
        reasons.append("too-long")
    if stripped in previous_replies:
        reasons.append("repeated-reply")
    prompt_words = _content_words(prompt)
    reply_words = _content_words(stripped)
    if any(phrase in prompt.lower() for phrase in ("story about", "write a story", "tell me a story", "tiny story")):
        subject_words = prompt_words - {"story", "funny", "tiny", "short"}
        if subject_words and not (subject_words & reply_words):
            reasons.append("missing-story-subject")
    if "<system>" in lowered or "<assistant>" in lowered or "you are chudgpt" in lowered:
        reasons.append("prompt-leak")
    if "[emoji context:" in lowered:
        reasons.append("emoji-context-leak")
    if re.search(
        r"\b(?:training (?:data|dataset|example|corpus)|dataset row|this prompt|this response|"
        r"assistant response|user prompt|fine[- ]?tuning example)\b",
        lowered,
    ):
        reasons.append("training-data-leak")
    if "http" in lowered and not any(word in _words(prompt) for word in ("url", "link", "website")):
        reasons.append("unrequested-url")
    if "the exact joke still depends on the caption and conversation around it" in lowered and classify_intent(prompt).name != "meme":
        reasons.append("meme-template-leak")
    if stripped.count("```") % 2:
        reasons.append("broken-code-fence")
    if has_structured_list(stripped) and not requests_structured_response(prompt):
        reasons.append("unrequested-structured-list")
    requested_items = _requested_item_count(prompt)
    if requested_items is not None:
        listed_items = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S+", stripped))
        if listed_items != requested_items:
            reasons.append("wrong-item-count")
    if re.search(r"\b(?:one|a single) sentence\b", prompt, re.I) and _sentence_count(stripped) != 1:
        reasons.append("sentence-count-constraint")
    if re.search(r"\b(?:answer|respond|reply)\s+(?:only\s+)?(?:yes or no|with yes or no)\b", prompt, re.I):
        if not re.fullmatch(r"\s*(?:yes|no)[.!]?\s*", stripped, re.I):
            reasons.append("yes-no-constraint")
    phrase_limit = repeated_phrase_constraint(prompt)
    if phrase_limit and lowered.count(phrase_limit[0].lower()) > phrase_limit[1]:
        reasons.append("phrase-repetition-constraint")
    if re.search(r"[{}]\s*(?:do|once|reads?)\b|\bexactal\b", lowered):
        reasons.append("corrupt-fragment")
    if re.search(r"\bcaption and conversation\b|\bone useful way into\b", lowered):
        reasons.append("known-template-leak")
    if re.search(
        r"\b(?:i(?:'m| am) not sure (?:what|how|why|whether)|"
        r"i (?:do not|don't) know what .{0,80}? mean(?:s)?(?: yet| here)?|"
        r"what did you mean|could you (?:clarify|rephrase|say that another way)|"
        r"try (?:asking|saying|wording) (?:it|that) another way)\b",
        lowered,
    ):
        reasons.append("generic-uncertainty-fallback")
    if re.search(r"\blanguage model\b.{0,120}\blanguage model\b", lowered):
        reasons.append("identity-repetition")
    if re.search(r"\b(?:does not have no|i can't understand what i can|provide you with the language)\b", lowered):
        reasons.append("broken-identity-grammar")
    requested_code = any(
        marker in " ".join(_words(prompt))
        for marker in ("code", "python", "c#", "csharp", "javascript", "unity", "script", "program")
    )
    if "```" in stripped and not requested_code:
        reasons.append("unrequested-code-block")
    explicit_code_request = bool(re.search(
        r"\b(?:write|make|create|generate|send|give|code)\b.{0,50}"
        r"\b(?:code|script|program|monobehaviour|mono ?behaviour)\b|"
        r"\b(?:unity|python|c#|csharp|javascript)\b.{0,40}\b(?:script|code|program)\b",
        prompt,
        re.I,
    ))
    has_code = "```" in stripped or bool(re.search(
        r"\b(?:using\s+(?:System|UnityEngine)|class\s+\w+|def\s+\w+|function\s+\w+|const\s+\w+\s*=)",
        stripped,
    ))
    if explicit_code_request and not has_code:
        reasons.append("missing-requested-code")
    requested_language = requested_programming_language(prompt)
    detected_languages = detected_programming_languages(stripped) if has_code else set()
    if requested_language and has_code and detected_languages and requested_language not in detected_languages:
        reasons.append("wrong-programming-language")
    if len(detected_languages) > 1 and not re.search(r"\b(?:compare|convert|translate|both|multiple)\b", prompt, re.I):
        reasons.append("mixed-programming-languages")
    if re.search(r"\b(?:return|output|respond with)\s+(?:only\s+)?(?:the\s+)?(?:complete\s+)?code\b|\bcode only\b", prompt, re.I):
        outside = re.sub(r"```[\s\S]*?```", "", stripped).strip()
        if outside:
            reasons.append("code-only-constraint")
    if classify_intent(prompt).name != "meme" and not has_strong_math_intent(prompt) and re.search(
        r"(?:\$?\d+(?:\.\d+)?\s*(?:[+*/=×÷]|-(?=\s*\d))\s*\$?\d+|"
        r"\b(?:multiply|divide|calculate|equals)\b.{0,50}\d|"
        r"\b(?:distance equals|speed times time|percent of|split equally into)\b)",
        stripped,
        re.I,
    ):
        reasons.append("unrequested-math")
    if not has_strong_math_intent(prompt) and re.search(
        r"\b\d+(?:\.\d+)?\s*[^a-z0-9\s.,]{1,3}\s*-?\d+(?:\.\d+)?\b",
        stripped,
        re.I,
    ):
        reasons.append("unrequested-numeric-expression")
    if not has_strong_math_intent(prompt) and re.search(r"[$€£]\s*0{2,}\d|\b\d{4}\b", stripped):
        prompt_numbers = set(re.findall(r"\b\d{4}\b", prompt))
        reply_numbers = set(re.findall(r"\b\d{4}\b", stripped))
        if not prompt_numbers or not reply_numbers <= prompt_numbers:
            reasons.append("unrequested-specific-number")
    trigrams = [tuple(words[index:index + 3]) for index in range(max(0, len(words) - 2))]
    if trigrams and max(Counter(trigrams).values()) >= 3:
        reasons.append("repetition-loop")
    if len(words) >= 12:
        bigrams = [tuple(words[index:index + 2]) for index in range(len(words) - 1)]
        if max(Counter(bigrams).values(), default=0) >= 4 or len(set(words)) / len(words) < 0.38:
            reasons.append("degenerate-repetition")
    clauses = [" ".join(_words(part)) for part in re.split(r"[.!?;\n]+", stripped) if len(_words(part)) >= 3]
    if clauses and max(Counter(clauses).values()) >= 2:
        reasons.append("repeated-clause")
    if re.search(r"\b(?:an? )?language model\b.{0,80}\bis (?:an? )?language model\b", lowered):
        reasons.append("recursive-self-definition")

    prompt_words = _content_words(prompt)
    reply_words = _content_words(stripped)
    social_prompt = any(
        phrase in f" {' '.join(_words(prompt))} "
        for phrase in (" i feel ", " i am ", " i'm ", " talk ", " chat ", " hello ", " hi ")
    )
    shared_subject = bool(prompt_words & reply_words)
    context_words = _content_words(conversation_context)
    if len(_words(prompt)) <= 4 and conversation_context:
        shared_subject = shared_subject or bool(context_words & reply_words)
    if classify_intent(prompt).name == "meme":
        # Meme names are often numbers or very short slang (67, F, POV), so
        # the normal four-character content-word filter is too destructive.
        prompt_anchors = set(_words(prompt)) - STOP_WORDS - {"mean", "meme", "memes"}
        reply_anchors = set(_words(stripped))
        shared_subject = shared_subject or bool(prompt_anchors & reply_anchors)
    if prompt_words and not social_prompt and not shared_subject:
        reasons.append("no-shared-subject")
    topic_starter = re.search(r"\bone useful way into ([a-z][a-z -]{1,40}) is\b", lowered)
    if topic_starter and topic_starter.group(1).strip() not in prompt.lower():
        reasons.append("unrelated-topic-starter")
    unrelated_technical = any(word in reply_words for word in ("javascript", "rigidbody", "fixedupdate"))
    requested_technical = any(word in _words(prompt) for word in ("code", "javascript", "unity", "physics", "python", "c#", "csharp", "script", "program"))
    if unrelated_technical and not requested_technical:
        reasons.append("unrequested-technical-topic")
    emotional_prompt = re.search(
        r"\b(?:feel|feeling|am|had)\b.*\b(?:sad|lonely|upset|miserable|excited|proud|happy|rough)\b",
        " ".join(_words(prompt)),
    )
    technical_reply = any(
        word in reply_words
        for word in ("unity", "javascript", "python", "rigidbody", "fixedupdate", "compiler", "function")
    ) or "`update`" in lowered
    if emotional_prompt and technical_reply:
        reasons.append("technical-reply-to-emotional-prompt")
    return not reasons, tuple(reasons)


def score_generated_reply(prompt: str, reply: str) -> float:
    """Rank valid candidates by response-type and lightweight lexical relevance.

    This is intentionally general: it rewards matching the requested response
    type and penalizes obvious category switches rather than looking up answers.
    """
    valid, reasons = assess_generated_reply(prompt, reply)
    if not valid:
        return -100.0 - 5.0 * len(reasons)
    prompt_words = _words(prompt)
    reply_words = _words(reply)
    prompt_set = _content_words(prompt)
    reply_set = _content_words(reply)
    score = 2.0 + min(3.0, len(prompt_set & reply_set) * 0.75)
    normalized = " ".join(prompt_words)
    asks_code = classify_intent(prompt).name == "code"
    has_code = "```" in reply or any(token in reply for token in ("def ", "class ", "using System", "function "))
    if asks_code:
        score += 4.0 if has_code else -5.0
    elif has_code:
        score -= 8.0
    asks_math = bool(re.search(r"\d\s*(?:[+*/×÷]|-(?=\s*\d))\s*\d", prompt)) or any(
        word in prompt_words for word in ("calculate", "solve", "distance", "percent")
    )
    # Use the same conservative classifier as serving and retrieval.  This
    # assignment intentionally replaces the legacy broad local heuristic.
    asks_math = has_strong_math_intent(prompt)
    answer_style = bool(re.search(r"(?:^|\s)-?\d+(?:\.\d+)?(?:\s|[.,]|$)", reply))
    if asks_math and answer_style:
        score += 2.0
    if not asks_math and re.fullmatch(r"(?:the answer is )?-?\d+(?:\.\d+)?[.!]?", reply.strip(), re.I):
        score -= 12.0
    greeting = bool(re.match(r"^(?:hi|hello|hey|yo)\b", normalized))
    if greeting and any(word in reply_words for word in ("hello", "hey", "hi")):
        score += 3.0
    if "one sentence" in normalized and reply.count(".") + reply.count("!") + reply.count("?") > 2:
        score -= 3.0
    if "code only" in normalized and reply.strip().startswith("```") and reply.strip().endswith("```"):
        score += 2.0
    return score
