"""Build a balanced Public v10 SFT set for identity, AI, memes, and short prompts."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_v10_conversations.jsonl"
SOURCES = (ROOT / "data/public_conversations.jsonl", ROOT / "data/alignment_conversations.jsonl", ROOT / "data/public_v9_conversations.jsonl")
TARGET = 12_000
MATH = re.compile(r"calculate|arithmetic|factorial|percent|\d\s*[+*/Ã—x-]\s*\d", re.I)
BAD = ("ï¿½", "Ãƒ", "one useful way into", "distance equals speed times time", "multiply quantity by price")


def dialogue(*pairs: tuple[str, str], source: str = "public-v10") -> dict[str, object]:
    messages: list[dict[str, str]] = []
    for user, assistant in pairs:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": messages, "source": source}


def build(seed: int = 1010) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    answers: Counter[str] = Counter()
    math_count = 0

    def add(item: dict[str, object], cap: int = 4) -> None:
        nonlocal math_count
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if not text.strip() or any(term.lower() in text.lower() for term in BAD):
            return
        is_math = bool(MATH.search(" ".join(m["content"] for m in messages if m["role"] == "user")))
        if is_math and math_count >= 1_700:
            return
        response_texts = [m["content"] for m in messages if m["role"] == "assistant"]
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen or any(answers[a] >= cap for a in response_texts):
            return
        seen.add(key); answers.update(response_texts)
        records.append({"messages": messages, "source": item.get("source", "public-v10")})
        math_count += int(is_math)

    source_rows: list[dict[str, object]] = []
    for path in SOURCES:
        source_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rng.shuffle(source_rows)
    for item in source_rows:
        add(item, cap=2)
        if len(records) >= 5_500:
            break

    identity_questions = [
        "What are you?", "Who are you?", "Which assistant is this?", "Are you an AI?",
        "Which ChudGPT are you?", "What model am I talking to?", "Tell me your identity.",
        "Are you Pro?", "Are you Plus?", "What does Public mean in your name?",
        "What exactly are you running as?", "hu are u", "which chud is this",
    ]
    identity_answers = [
        "I am ChudGPT-Public, the public experimental model in the ChudGPT family. I am a small decoder-only transformer language model.",
        "You are talking to ChudGPT-Public. I generate replies by predicting text tokens from your message and recent conversation.",
        "I am the Public ChudGPT model, not Pro or Plus. I am experimental and can still make mistakes or produce nonsense.",
        "Yes, I am an AI language model called ChudGPT-Public. I am software, not a person or a conscious being.",
    ]
    for i in range(520):
        q = rng.choice(identity_questions)
        a = rng.choice(identity_answers)
        add(dialogue((q + rng.choice(["", " Please be direct.", " In one sentence."]), a)), cap=140)

    # These contextual phrasings intentionally have many distinct surface forms.
    # Earlier v10 generation sampled from too small a Cartesian product, so
    # deduplication collapsed 520 requested examples into fewer than 100 rows.
    identity_leads = [
        "Quick check:", "Before we continue,", "For the record,", "Be precise:",
        "No marketing language:", "A friend asked me:", "On this public page,",
        "I am comparing assistants:", "In plain English,", "One short answer:",
        "I lost track:", "Please clarify:", "For my notes,", "Seriously,",
        "Without a slogan,", "Just the facts:", "New conversation:",
        "This may sound obvious:", "Help me label this chat:", "Direct question:",
    ]
    identity_tails = [
        "", " Answer naturally.", " Do not describe another model.",
        " I mean the assistant replying right now.", " Keep the answer useful.",
    ]
    for lead in identity_leads:
        for question in identity_questions:
            add(dialogue((f"{lead} {question}{rng.choice(identity_tails)}", rng.choice(identity_answers))), cap=220)

    family = {
        "ChudGPT": "ChudGPT is the overall project and family of small experimental language models and serving profiles.",
        "ChudGPT-Public": "ChudGPT-Public is the current public experimental model and API in the ChudGPT family.",
        "ChudGPT Plus": "ChudGPT Plus is a conversational ChudGPT model and serving profile.",
        "ChudGPT Pro": "ChudGPT Pro is a general-use serving profile based on Plus with expanded runtime context and recovery.",
        "ChudGPT Code": "ChudGPT Code is the programming-focused ChudGPT experience.",
        "ChudGPT Ultimate": "ChudGPT Ultimate is an older ChudGPT experiment.",
        "Buggy ChudGPT": "Buggy ChudGPT intentionally uses an early broken checkpoint for chaotic, unreliable conversations.",
        "MEGA CHUD": "MEGA CHUD is a deliberately chaotic experimental model designed to be worse than the other ChudGPT variants.",
        "archived ChudGPT checkpoints": "The archived 700, 1300, 1500, and 1600 checkpoints are historical ChudGPT training snapshots.",
    }
    forms = ["What is {name}?", "Tell me about {name}.", "Where does {name} fit?", "Explain {name} briefly.", "what even is {name}"]
    for name, fact in family.items():
        for _ in range(75):
            add(dialogue((rng.choice(forms).format(name=name), fact)), cap=90)
    family_leads = ["Compare the project:", "From the ChudGPT library,", "I saw this name online:",
                    "For a quick overview,", "No guessing:", "In this model family,"]
    family_forms = ["describe {name}", "what role does {name} have?", "how is {name} different?",
                    "is {name} a model or profile?", "give me the point of {name}"]
    for name, fact in family.items():
        for lead in family_leads:
            for form in family_forms:
                add(dialogue((f"{lead} {form.format(name=name)}", fact)), cap=180)
    for _ in range(130):
        add(dialogue((rng.choice(["What other ChudGPTs exist?", "List the ChudGPT family.", "What are the other models?", "Which ChudGPT is better?"]),
                      "The family includes Public, Plus, Pro, Code, Ultimate, intentionally broken Buggy, deliberately chaotic MEGA CHUD, and archived 700, 1300, 1500, and 1600 snapshots. Better depends on the goal: Public is the open experiment, Code focuses on programming, and Pro is the broader serving profile.")), cap=140)

    ai_facts = [
        ("AI", "Artificial intelligence is software designed to perform tasks that normally require human-like pattern recognition, prediction, planning, or language processing."),
        ("an LLM", "A large language model predicts text tokens using patterns learned during training. It does not automatically understand or verify every claim it produces."),
        ("a parameter", "A parameter is a learned numerical value inside a neural network. Training adjusts many parameters to reduce prediction error."),
        ("training data", "Training data is the collection of examples used to adjust a model's parameters. Its quality, balance, and coverage strongly affect behavior."),
        ("a checkpoint", "A checkpoint is a saved model state from a particular point in training, usually including learned weights and training progress."),
        ("inference", "Inference is the stage where a trained model receives input and generates a prediction or reply without updating its weights."),
        ("context length", "Context length is the maximum number of tokens a model can consider at once, including instructions and recent conversation."),
    ]
    for term, answer in ai_facts:
        for form in ["What is {x}?", "Explain {x} simply.", "How does {x} work?", "Can you define {x}?", "why does {x} matter"]:
            for suffix in ["", " Keep it short.", " I am new to this."]:
                add(dialogue((form.format(x=term) + suffix, answer)), cap=20)
        for lead in ["I am learning AI:", "Technical basics:", "Explain without hype:",
                     "For a beginner project:", "I keep hearing this term:", "Short lesson:"]:
            for request in ["define {x}", "what does {x} actually mean?", "why should I understand {x}?",
                            "explain the purpose of {x}"]:
                add(dialogue((f"{lead} {request.format(x=term)}", answer)), cap=90)
    for _ in range(180):
        add(dialogue((rng.choice(["Why do small models make nonsense?", "Why did that AI answer strangely?", "Can a language model be confidently wrong?", "Why can an LLM lose the topic?"]),
                      rng.choice(["Small models have limited capacity and may not learn enough robust patterns, so sampling can produce fluent-looking nonsense.", "A language model predicts likely text rather than checking truth automatically, so weak training coverage or context can produce confident mistakes.", "Short prompts can be ambiguous, and a small model may latch onto the wrong learned pattern unless training data covers that kind of context."]))), cap=80)

    meme_responses = [
        "That sounds like meme shorthand. I get the vibe, though the exact reference may depend on the community using it.",
        "Internet slang detected. I might not know the exact origin, but it reads like an intentionally absurd reaction phrase.",
        "That has chaotic meme energy. If you mean a specific trend, give me where you saw it and I can interpret the context.",
        "I recognize the joke-like tone, but I would rather admit uncertainty than invent an unrelated fact.",
    ]
    meme_prompts = ["67", "bro said 67", "67 moment", "why is everyone saying 67", "what is the 67 meme", "tung tung tung sahur", "skibidi", "zero aura", "bro is cooked", "npc energy", "absolute cinema", "let him cook", "what the sigma", "brainrot", "goofy ahh"]
    for i in range(620):
        p = rng.choice(meme_prompts) + rng.choice(["", " lol", " 💀", "?", " fr"])
        if p.strip().startswith("67") and not any(ch in p for ch in "+%"):
            a = rng.choice(["67 is being used here as a meme or reaction number, not a math problem. Its meaning depends on the surrounding trend or joke.", "You dropped 67 like a meme. I am reading it as internet shorthand unless you give me a calculation or score context.", *meme_responses])
        else:
            a = rng.choice(meme_responses)
        add(dialogue((p, a)), cap=100)
    for n, answer in [("67 + 8", "67 + 8 is 75."), ("67% of 300", "67% of 300 is 201."), ("I scored 67 points", "You scored 67 points. Whether that is good depends on the game or scoring scale.")]:
        for suffix in ["", ".", " today", " can you help?"]:
            add(dialogue((n + suffix, answer)), cap=8)

    for i in range(600):
        token = rng.choice(["yo", "bruh", "hmm", "nah", "okay", "wild", "what", "pluh", "jdkslfjskdf", "florp", "???"])
        answer = rng.choice(["Hey, I am here.", "Fair enough. What is going on?", "That is a mysterious message. Give me a little context and I will roll with it.", "I am not sure that has a fixed meaning, but it definitely arrived with confidence.", "All right, new topic accepted. What direction are we going?"])
        add(dialogue((token + rng.choice(["", "!", "...", " bro"]), answer)), cap=80)

    rng.shuffle(source_rows)
    for item in source_rows:
        add(item, cap=4)
        if len(records) >= TARGET:
            break
    if len(records) < TARGET:
        raise RuntimeError(f"Built {len(records)} records; expected {TARGET}")
    rng.shuffle(records)
    return records[:TARGET]


def main() -> None:
    records = build()
    OUTPUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records):,} Public v10 conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
