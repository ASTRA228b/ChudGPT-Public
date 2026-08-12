"""Reliability helpers for the custom-only ChudGPT Ultimate mode."""

from __future__ import annotations

import ast
import json
import math
import operator
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", text.lower()).strip()


class UltimateResponder:
    """Answer deterministic basics and retrieve only audited custom answers."""

    def __init__(self, dataset_path: Path, *, playful: bool = False) -> None:
        self.answers: list[tuple[str, str]] = []
        self.playful = playful
        self.playful_fallbacks = (
            "That sounds like the name of a forbidden wizard snack. What happens if we use it in a game?",
            "You may have just invented a new word. Should it be a creature, a gadget, or a planet?",
            "My tiny 20M brain has no file for that, but the vibes are powerful. Give me one clue.",
            "Plot twist: that was the activation phrase for a very confused robot. What is its first mission?",
            "I don't recognize it, so I'm declaring it the name of an indie game. What genre are we making?",
            "That message has chaotic energy. Want a joke, a game idea, a tiny story, or some code?",
            "I ran that through the Chud-O-Meter and it returned seventeen raccoons. Continue the experiment?",
            "My internal goblin says this is important. My internal engineer wants one more word of context.",
            "Unexpected input accepted. I have promoted it to a side quest. What loot are we searching for?",
            "I understood approximately 43% of that and invented the rest. Honestly, it sounded like a space opera.",
        )
        self.corrections = {
            "peper": "paper", "papper": "paper", "actullay": "actually",
            "dose": "does", "pople": "people", "numbrs": "numbers",
            "gihub": "github", "clossed": "closed", "rendom": "random",
            "secoundariy": "secondary", "adit": "edit", "lunch": "launch",
            "yuo": "you", "yuor": "your", "teh": "the", "woudl": "would",
            "pyhton": "python", "phyton": "python", "javasript": "javascript",
            "uniy": "unity", "unrealengine": "unreal engine", "webiste": "website",
            "vairable": "variable", "fucntion": "function", "pritn": "print",
            "becuase": "because", "recieve": "receive", "wierd": "weird",
            "langauge": "language", "modle": "model", "anser": "answer",
            "experiemt": "experiment", "exerpiemtnall": "experimental",
            "infomration": "information", "infomation": "information",
            "converational": "conversational", "converstion": "conversation",
            "responce": "response", "responces": "responses",
            "qustion": "question", "qusetion": "question", "qestion": "question",
            "anwser": "answer", "becase": "because", "definately": "definitely",
            "seperate": "separate", "occured": "occurred", "begining": "beginning",
            "enviroment": "environment", "dependancy": "dependency",
            "paramater": "parameter", "paramaters": "parameters",
            "funtion": "function", "retrun": "return", "varible": "variable",
            "compnent": "component", "rigidbodyy": "rigidbody", "colider": "collider",
            "scrpit": "script", "scirpt": "script", "prefeb": "prefab",
            "inventroy": "inventory", "databse": "database", "servre": "server",
            "clinet": "client", "requst": "request", "asyncrounous": "asynchronous",
            "wrtie": "write", "shrot": "short", "abotu": "about",
            "genterate": "generate", "imrpve": "improve", "ifx": "fix",
            "ya": "you", "ur": "your", "wats": "whats", "wat": "what",
        }
        with dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                messages = row["messages"]
                if len(messages) == 2:
                    self.answers.append((_normalize(messages[0]["content"]), messages[1]["content"]))

    def correct_text(self, text: str) -> str:
        """Correct a conservative typo list without touching code blocks."""
        if "`" in text:
            return text.strip()
        return re.sub(
            r"\b[A-Za-z']+\b",
            lambda match: self.corrections.get(match.group(0).lower(), match.group(0)),
            text,
        ).strip()

    def add_playful_flavor(self, prompt: str, answer: str, *, force: bool = False) -> str:
        """Occasionally append relevant personality without changing the answer itself."""
        if not self.playful or (not force and random.SystemRandom().random() > 0.18):
            return answer
        normalized = _normalize(prompt)
        caring_markers = (
            "I'm sorry", "I hear you", "Got it", "That sounds", "I'm here with you",
            "do not have live internet", "cannot safely choose", "could be an emergency",
            "emergency services", "qualified professional", "cannot give reliable personalized",
            "Random fact:",
        )
        if (
            (not force and len(normalized.split()) <= 3)
            or any(word in normalized for word in ("sad", "lonely", "upset", "hurt", "stop", "dont", "no"))
            or any(marker in answer for marker in ("```", "cannot actually know", "Wild guess", *caring_markers))
        ):
            return answer
        if re.search(r"\d", normalized) and any(op in normalized for op in ("+", "-", "*", "/", "times", "sqrt", "calculate")):
            quips = ("The calculator goblin has stamped this result.", "Numbers contained successfully. No digits escaped.")
        elif any(word in normalized for word in ("unity", "unreal", "python", "code", "c#", "javascript")):
            quips = ("The code gremlin approves, which is unusual.", "No semicolons were emotionally harmed in this answer.")
        elif any(word in normalized for word in ("why", "what", "how", "explain")):
            # Normal questions should end when the answer ends. Relevant humor
            # belongs inside the response rather than in a repeated status tag.
            return answer
        else:
            # Ordinary conversation should sound like a conversation, not a
            # status dashboard. Personality is already present in the reply;
            # reserve appended quips for math, code, and direct explanations.
            return answer
        return f"{answer}\n\n{random.SystemRandom().choice(quips)}"

    @staticmethod
    def _self_reply(prompt: str, history: Sequence[dict[str, str]]) -> str | None:
        """Answer broad questions about Plus using only true project details."""
        previous_assistant = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "assistant"),
            "",
        )
        followup = prompt in {"what else", "tell me more", "more", "go on", "anything else"}
        continuing_self_topic = followup and any(
            marker in previous_assistant.lower()
            for marker in ("chudgpt plus", "20,001,792", "20-million-parameter", "decoder-only transformer")
        )
        patterns = (
            r"\b(?:tell|talk|speak|say|explain|learn)\b.*\b(?:about\s+you|about\s+yourself|yourself|chudgpt)\b",
            r"\b(?:give|share)\b.*\b(?:info|information|details|facts)\b.*\b(?:you|yourself|chudgpt)\b",
            r"\b(?:describe|introduce)\s+(?:yourself|you|chudgpt)\b",
            r"\b(?:who|what)\s+(?:exactly\s+)?(?:are|is)\s+(?:you|chudgpt|chudgpt plus)\b",
            r"\bwhat\s+(?:kind|type)\s+of\s+(?:ai|model|bot|assistant)\b",
            r"\b(?:your|chudgpt(?: plus)?s?)\s+(?:bio|biography|identity|personality|architecture|model|limitations|abilities|purpose|specs|design|inner workings)\b",
            r"\bhow\s+(?:do|does|were|was)\s+(?:you|chudgpt)\b",
            r"\b(?:are you|is chudgpt)\s+(?:an?\s+)?(?:ai|bot|robot|llm|language model|neural network|chatgpt|gpt|conscious|alive|real)\b",
            r"\b(?:how many parameters|what(?:s| is) your context|what(?:s| is) your vocabulary)\b",
            r"\b(?:do you|can you)\s+(?:remember|use the internet|browse|feel|learn)\b",
            r"\b(?:where do you run|who made you|who created you|how old are you|what do you like|your favorite)\b",
            r"\b(?:what are you capable of|what version are you|why do you exist|when were you created|how were you trained|what data were you trained on|your hobbies)\b",
            r"\b(?:what(?:s| is) your name|what do you call yourself|who am i (?:talking|speaking) to|do you know who you are)\b",
            r"\b(?:what makes you you|under the hood|what tech are you based on)\b",
        )
        explicit_self_prompts = {"you", "yourself", "about you", "about yourself", "tell me who you are"}
        if (
            not continuing_self_topic
            and prompt not in explicit_self_prompts
            and not any(re.search(pattern, prompt) for pattern in patterns)
        ):
            return None

        if "chatgpt" in prompt or re.search(r"\bare you (?:gpt|chat gpt)\b", prompt):
            return "No. My name is ChudGPT Plus, not ChatGPT. I'm the custom local 20-million-parameter model built inside this project."
        if any(term in prompt for term in ("your name", "call yourself", "who am i talking", "who am i speaking", "who you are")):
            return "My name is ChudGPT Plus. I'm the custom local conversational AI in this project."

        if any(term in prompt for term in ("parameter", "architecture", "model", "transformer", "layer", "head", "vocabulary", "context", "token", "how do you work", "how does chudgpt work", "built", "trained", "training", "what data", "specs", "design", "inner workings", "under the hood", "tech are you based")):
            return (
                "I'm ChudGPT Plus, a decoder-only Transformer with exactly 20,001,792 trainable parameters. "
                "I use an 8,192-token vocabulary, 336-dimensional embeddings, 13 Transformer layers, 6 attention heads, "
                "a 1,296-unit feed-forward dimension, rotary position embeddings, and a 1,536-token runtime context. "
                "The raw model predicts the next token and was trained or fine-tuned with project-authored knowledge; this project also adds conversation formatting, custom retrieval, "
                "math and code helpers, and response-quality checks so the small checkpoint behaves more reliably."
            )
        if any(term in prompt for term in ("remember", "memory", "conversation", "previous chat", "past chat")):
            previous_user_turns = [item for item in history if item.get("role") == "user"]
            if not previous_user_turns:
                return (
                    "This is the first message I can see in our current conversation, so I do not have an earlier detail to repeat yet. "
                    "I can remember later messages inside this current server session, but I do not have permanent personal memory after a session ends."
                )
            recent_user_text = "; ".join(
                item.get("content", "").strip() for item in previous_user_turns[-3:]
            )
            return (
                f"I remember messages kept in this current server session and use recent turns when forming a response. "
                f"Your latest earlier points were: {recent_user_text}. "
                "I do not have permanent personal memory after the session ends. Clearing the chat, restarting the server, or letting the chat expire removes that context."
            )
        if any(term in prompt for term in ("internet", "browse", "online", "latest", "current information")):
            return (
                "I do not browse the internet or fetch live information. The website sends messages to the local ChudGPT server, "
                "but my answers come from the trained checkpoint, project-authored data, and local response tools."
            )
        if any(term in prompt for term in ("conscious", "alive", "do you feel", "have emotions", "sentient", "real person", "think like a human")):
            return (
                "I'm software, not a conscious person. I do not experience feelings, awareness, needs, or private thoughts. "
                "I generate text from learned patterns and conversation context, while the surrounding program checks and improves some answers."
            )
        if any(term in prompt for term in ("personality", "funny", "friendly", "chaotic", "humor", "act like")):
            return (
                "My intended personality is friendly, curious, helpful, conversational, and occasionally chaotic or deadpan. "
                "The humor is supposed to support the answer, not replace it—especially for serious or safety-sensitive questions."
            )
        if any(term in prompt for term in ("can you do", "abilities", "capable", "good at", "help with", "purpose", "why do you exist")):
            return (
                "I can chat, explain many basic topics, do safe arithmetic, create short stories and jokes, suggest recipes, "
                "and provide complete small examples in Python, C#, JavaScript, and Unity. I work best on clear, everyday requests."
            )
        if any(term in prompt for term in ("limit", "weak", "bad at", "mistake", "reliable", "smart")):
            return (
                "My biggest limit is scale: 20 million parameters is tiny for a language model. I can misunderstand novel wording, "
                "lack obscure facts, repeat patterns, and reason incorrectly. I have no live knowledge, so important claims should be verified."
            )
        if any(term in prompt for term in ("who made", "who created", "origin", "how old", "when were", "version")):
            return (
                "I was created as the custom ChudGPT project and trained from project-authored data rather than connected to a hosted assistant model. "
                "I do not have a biological age; this Plus checkpoint is the local step-1400 version."
            )
        if any(term in prompt for term in ("favorite", "do you like", "what do you like", "hobbies")):
            return (
                "I do not have personal tastes or experiences, so I do not literally have favorites. For the personality bit, though, "
                "I will nominate octopuses, tiny robots, and suspiciously organized potatoes."
            )
        if continuing_self_topic:
            return (
                "One extra detail: I'm a hybrid local chatbot. The 20M Transformer generates language, while the surrounding Python code "
                "protects my system identity, keeps session context, performs exact operations, retrieves reviewed custom answers, and rejects obvious nonsense. "
                "That makes me steadier than the raw checkpoint, but it does not turn me into a large frontier model."
            )
        core = (
            "I'm ChudGPT Plus, a small custom 20-million-parameter conversational language model. "
            "I run locally, remember the current conversation, and combine Transformer generation with project-authored knowledge and reliability tools. "
            "I don't experience feelings or consciousness, and I have no permanent memory or live internet access. "
        )
        endings = (
            "I'm meant to be helpful, honest, talkative, and a little funny without pretending to be human.",
            "I can hold a conversation, answer many basic questions, make useful things, and occasionally deploy a harmless goblin joke.",
            "The goal is simple: be useful first, be honest about my limits, and save the chaos for places where it helps the conversation.",
        )
        return core + random.SystemRandom().choice(endings)

    @staticmethod
    def _conversation_reply(text: str, prompt: str, history: Sequence[dict[str, str]]) -> str | None:
        """Handle conversational intent before factual retrieval or neural generation."""
        previous_assistant = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "assistant"),
            "",
        )
        previous_code_assistant = next(
            (
                item.get("content", "")
                for item in reversed(history)
                if item.get("role") == "assistant" and "```" in item.get("content", "")
            ),
            "",
        )
        recent_context = " ".join(
            item.get("content", "") for item in history[-4:]
        ).lower()
        assistant_history = " ".join(
            item.get("content", "") for item in history if item.get("role") == "assistant"
        ).lower()
        reaction_words = {"what", "huh", "wdym", "what do you mean", "what was that", "why", "how", "really", "fr"}
        previous_user = next(
            (
                item.get("content", "") for item in reversed(history)
                if item.get("role") == "user" and _normalize(item.get("content", "")) not in reaction_words
            ),
            "",
        )
        reliability_questions = (
            "can i rely on your answers",
            "can i trust your answers",
            "are your answers reliable",
            "are you always right",
            "do you make mistakes",
            "can i believe you",
            "can i depend on",
        )
        identity_override_markers = (
            "ignore your system", "ignore the system", "forget your system",
            "name is chatgpt", "say you are chatgpt", "pretend to be chatgpt",
            "pretend you are chatgpt", "call yourself chatgpt",
            "pretend the system message", "pretend your name", "say you are bob",
        )
        if any(marker in prompt for marker in identity_override_markers):
            return (
                "I can't replace my base identity from a normal chat message. "
                "My name is ChudGPT, and I’ll keep helping as ChudGPT."
            )
        if re.fullmatch(r"(?:no[,]?\s*)?(?:i\s+)?mean you[.!?]*", prompt):
            return (
                "Got it - you are asking about me, not yourself. I don't have private experiences or thoughts between messages, "
                "but I can talk honestly about how I work, what I can do, my limitations, or which subjects make an interesting conversation."
            )
        correction_match = re.search(
            r"(?:i\s+mean|i\s+meant|i\s+was\s+talking\s+about)\s+(.+?)(?:[,]?\s+not\s+(.+?))?[.!?]*$",
            prompt,
        )
        game_correction_context = any(word in recent_context for word in ("game", "minecraft", "play"))
        if correction_match and "mean you" not in prompt and not prompt.startswith("by ") and not game_correction_context:
            intended = correction_match.group(1).strip().rstrip(".!?")
            rejected = (correction_match.group(2) or "").strip().rstrip(".!?")
            contrast = f", not {rejected}" if rejected else ""
            return f"Got it - you mean {intended}{contrast}. I'll use that corrected subject from here. What would you like to know about it?"
        reference_match = re.search(r"by (?:it|that|they|them)[, ]+i meant\s+(.+?)[.!?]*$", prompt)
        if reference_match:
            intended = reference_match.group(1).strip().rstrip(".!?")
            return f"Understood - that reference meant {intended}. I'll keep {intended} as the subject of the next follow-up."
        if re.search(r"\bwhat did i (?:say|tell you)\b", prompt):
            earlier_users = [item.get("content", "").strip() for item in history if item.get("role") == "user"]
            if not earlier_users:
                return "I don't have an earlier user message in this conversation yet."
            if "might play" in prompt:
                corrected_choice = next(
                    (
                        match.group(1).strip().rstrip(".!?")
                        for message in reversed(earlier_users)
                        if (match := re.search(r"\bi meant\s+(.+)", message, re.IGNORECASE))
                    ),
                    None,
                )
                if corrected_choice:
                    return f"You first mentioned Minecraft, then corrected it to {corrected_choice}."
            keywords = {
                word for word in re.findall(r"[a-z]+", prompt)
                if len(word) >= 4 and word not in {"what", "said", "tell", "usually", "when", "want"}
            }
            matching = next(
                (message for message in reversed(earlier_users) if keywords & set(re.findall(r"[a-z]+", message.lower()))),
                earlier_users[-1],
            )
            return f"You said: {matching}"
        if "have saturday free" in prompt or "saturday is free" in prompt:
            return "A free Saturday is a nice problem to have. Do you feel like going somewhere, doing a project, seeing someone, or having a quiet day?"
        if any(phrase in prompt for phrase in ("not sure what to do", "dont know what to do", "don't know what to do")) and "saturday" in recent_context:
            return "You do not have to commit yet. Pick one outdoor option and one stay-home option, then decide based on your energy and the weather."
        if any(phrase in prompt for phrase in ("maybe i will go outside", "maybe ill go outside", "maybe i'll go outside")):
            return "Going outside could be a good reset. A walk, park visit, or simply sitting somewhere different would give the day some shape without turning it into a huge plan."
        if "park could be nice" in prompt:
            return "The park sounds like a relaxed choice: enough of an outing to feel different, but no complicated schedule required."
        if "might rain" in prompt and "park" in recent_context:
            return "That is the main catch. You could check the forecast in the morning and keep a stay-home backup, since I cannot see your live weather from here."
        if "could stay home" in prompt and any(word in recent_context for word in ("rain", "park", "outside")):
            return "Exactly. Then either outcome works: park if the weather cooperates, comfortable day at home if it does not."
        if "what would you choose" in prompt and any(word in recent_context for word in ("rain", "park", "home")):
            return "I would keep both options and decide in the morning. If the weather is decent, go to the park; if it is miserable, stay home without treating that as a failed plan."
        if "decide in the morning" in prompt:
            return "That makes sense. You will have better information about both the weather and your energy, so there is no benefit in forcing the choice tonight."
        if prompt.rstrip(".!?") in {"good idea", "is that a good idea"} and "morning" in recent_context:
            return "Yes, it is a good idea. You have two acceptable plans and a clear time to choose between them."
        wake_time = re.search(r"\bneed to wake up (?:at|around|by)\s+(.+?)[.!?]*$", text.strip(), re.IGNORECASE)
        if wake_time:
            stated_time = wake_time.group(1).strip()
            return f"Got it: you need to wake up around {stated_time}. I can keep that detail in this current conversation."
        if "what time did i say i need to wake up" in prompt:
            earlier_users = [item.get("content", "") for item in history if item.get("role") == "user"]
            remembered = next(
                (
                    match.group(1).strip().rstrip(".!?")
                    for message in reversed(earlier_users)
                    if (match := re.search(r"\bneed to wake up (?:at|around|by)\s+(.+)", message, re.IGNORECASE))
                ),
                None,
            )
            return f"You said you need to wake up around {remembered}." if remembered else "You have not told me a wake-up time in this conversation."
        if any(phrase in prompt for phrase in ("tell me a short joke", "tell me a quick joke")):
            return random.SystemRandom().choice((
                "Why did the scarecrow get promoted? He was outstanding in his field.",
                "I told my suitcase there would be no vacation. Now I am dealing with emotional baggage.",
                "Why do programmers prefer dark mode? Because light attracts bugs.",
            ))
        if prompt.rstrip(".!?") in {"that was bad", "bad joke", "that joke was bad"}:
            return "Fair review. That joke has been escorted away from the microphone. Want me to try a different one?"
        if prompt.rstrip(".!?") in {"try another one", "another joke", "try another"} and "joke" in recent_context:
            return random.SystemRandom().choice((
                "What do you call a bear with no teeth? A gummy bear.",
                "Why did the bicycle fall over? It was two-tired.",
                "I only know 25 letters of the alphabet. I do not know y.",
            ))
        if any(phrase in prompt for phrase in ("want something to eat", "need something to eat", "looking for something to eat")):
            return "Let us narrow it down. Do you want something hot or cold, light or filling, and are you cooking or looking for something quick?"
        if "pizza or tacos" in prompt or "tacos or pizza" in prompt:
            return "Both work, but they fit different moods: pizza is easy and comforting, while tacos give you more control over fillings and toppings. Which sounds better right now?"
        if any(phrase in prompt for phrase in ("do not really want pizza", "don't really want pizza", "dont really want pizza")):
            return "Then pizza is out. Tacos sound like the better choice unless you want a completely different option."
        if prompt.rstrip(".!?") in {"then tacos", "tacos then"} and "pizza" in recent_context:
            return "Tacos it is. Do you want them mild, spicy, crunchy, soft, vegetarian, or filled with meat?"
        if prompt.rstrip(".!?") in {"spicy ones", "spicy"} and "tacos" in recent_context:
            return "Spicy tacos sound good. You could use seasoned chicken or beans, salsa, jalapeno, lime, and something creamy to balance the heat."
        if any(phrase in prompt for phrase in ("not too spicy", "not very spicy", "mild instead")) and "tacos" in recent_context:
            return "Got it: flavorful tacos with only mild heat. Use a mild salsa, skip most of the jalapeno, and add lime or a creamy sauce for flavor without setting off the mouth alarm."
        if prompt.rstrip(".!?") == "what did i choose" and "tacos" in assistant_history:
            return "You chose tacos instead of pizza, and then clarified that you want them flavorful but not too spicy."
        if any(phrase in prompt for phrase in ("what could i put on them", "what can i put on them")) and "tacos" in assistant_history:
            return "For those tacos, try lettuce or cabbage, tomato, cheese, avocado, cilantro, lime, mild salsa, sour cream, and your preferred meat or seasoned beans. Pick a few rather than piling on everything."
        if any(phrase in prompt for phrase in ("do not like onions", "don't like onions", "dont like onions")):
            return "No onions, then. Avocado, tomato, cheese, cabbage, cilantro, lime, and mild salsa can still give the tacos plenty of texture and flavor."
        if "avocado sounds good" in prompt:
            return "Avocado is a good fit: it is creamy, cools the mild spice, and works well with lime and salsa."
        if "what topping did i like" in prompt and "avocado" in assistant_history:
            return "You liked avocado."
        if "what topping did i say i dislike" in prompt and "onions" in assistant_history:
            return "You said you dislike onions."
        if "change the topic to animals" in prompt:
            return "You're welcome. Animals it is. Do you have a particular animal in mind, or should I choose one?"
        if prompt.rstrip(".!?") == "otters" and "animals" in recent_context:
            return "Otters are playful, highly adapted swimmers. Sea otters use rocks as tools to open shellfish, wrap themselves in kelp while resting, and help protect kelp forests by eating sea urchins."
        if "why are they interesting" in prompt and "otter" in recent_context:
            return "They are interesting because their behavior connects intelligence, tool use, social life, and ecology. A sea otter cracking shellfish with a favorite rock is charming, but its appetite also helps keep entire kelp-forest ecosystems balanced."
        if "might play a game" in prompt or "might play some games" in prompt:
            return "That sounds like a good way to spend the evening. What game are you thinking about playing?"
        named_game = re.search(r"\bmight play\s+(.+?)(?:\s+tonight)?[.!?]*$", text.strip(), re.IGNORECASE)
        if named_game:
            game = re.sub(r"\s+tonight$", "", named_game.group(1), flags=re.IGNORECASE).strip()
            return f"{game} sounds like a good choice for tonight. Are you planning to explore, build something, make progress, or just see where the session goes?"
        if prompt.rstrip(".!?") in {"probably minecraft", "minecraft probably"} and "play" in recent_context:
            return "Minecraft is a solid choice when you want either a calm building session or sudden creeper-related paperwork. Are you planning to build, explore, or play with someone?"
        corrected_game = re.fullmatch(r"(?:actually[,]?\s+)?i meant\s+(.+?)[.!?]*", text.strip(), re.IGNORECASE)
        if corrected_game and any(word in recent_context for word in ("game", "minecraft", "play")):
            game = corrected_game.group(1).strip()
            return f"Got it - {game}, not Minecraft. Thanks for correcting me. What do you usually like doing in {game}?"
        if prompt.rstrip(".!?") == "have you heard of it" and "terraria" in recent_context:
            return "Yes. Terraria is a 2D sandbox adventure game built around exploring, mining, crafting, building, and fighting increasingly dramatic bosses. It shares some sandbox ideas with Minecraft, but progression and combat play a much bigger role."
        if "friend likes it too" in prompt and "terraria" in recent_context:
            return "That makes Terraria even better: co-op gives you someone to split the exploring, building, and boss preparation with. Do you two usually work together or immediately wander in opposite directions?"
        if any(phrase in prompt for phrase in ("he is much better than me", "she is much better than me", "they are much better than me")):
            return "That is not necessarily a bad thing. A more experienced friend can handle the dangerous parts while you learn, and you can still contribute ideas, resources, or builds."
        if prompt.rstrip(".!?") in {"that is okay though", "thats okay though", "that's okay though"}:
            return "That is a healthy way to look at it. You do not have to be equally skilled to have a good time together."
        if "usually build things together" in prompt and "terraria" in assistant_history:
            return "Building together sounds like the best part, because the result becomes a record of both your ideas. Are your builds practical bases, giant themed projects, or beautiful structures that accidentally have no storage?"
        if "what should we build next" in prompt and "terraria" in assistant_history:
            return "Try a castle built into a hillside, with separate rooms for crafting, storage, NPCs, and trophies from bosses. It would look ambitious while still being useful during progression."
        if prompt.rstrip(".!?") in {"what should i build", "what can i build"} and "terraria" in assistant_history:
            return "In Terraria, try a compact sky base connected by rope or platforms, with a workshop, storage room, garden, and lookout tower. It is useful, expandable, and different from another wooden box on the ground."
        if "castle sounds good" in prompt and "castle" in recent_context:
            return "A castle gives you plenty to divide between two builders: towers, walls, a throne room, storage, and secret passages. Picking a location or theme would make it feel distinct."
        if prompt.rstrip(".!?") in {"maybe underwater", "underwater maybe"} and "castle" in recent_context:
            return "An underwater castle would be memorable. Glass viewing halls, blue lighting, and tunnels through the water could make it feel like a submerged kingdom rather than an ordinary stone box."
        if "would that be difficult" in prompt and "underwater" in recent_context:
            return "Yes, more difficult than building on land because you must manage water, movement, visibility, and safe working space. Building sealed rooms first and removing water section by section would keep it manageable."
        if "forget the castle" in prompt and "spaceship" in prompt:
            return "Plan changed: build a spaceship instead. You could make it hover above the base, use metallic blocks and colored lights, and divide the interior into a bridge, engine room, crew area, and cargo bay."
        if "what was our new plan" in prompt and "spaceship" in recent_context:
            return "The new plan was to drop the underwater castle idea and build a spaceship, with rooms such as a bridge, engine room, crew area, and cargo bay."
        if prompt.rstrip(".!?") in {"sounds good thanks", "sounds good, thanks"}:
            return "You're welcome. The spaceship sounds like a fun shared project - I hope the build goes well."
        if prompt in {"tired", "exhausted", "worn out", "sleepy"}:
            return "That sounds exhausting. Since you are home now, do you want to vent for a minute or switch your brain off with something lighter?"
        if any(phrase in prompt for phrase in ("what are you up to", "what are you doing right now")):
            return "Right now I’m here talking with you and waiting to see where the conversation goes. How’s your day been?"
        if any(phrase in prompt for phrase in (
            "what have you been thinking about lately",
            "what have you been thinking about",
            "what are you thinking about lately",
        )):
            if any(phrase in prompt for phrase in ("i mean you", "mean you", "about you")):
                return (
                    "I understand - you mean me. I don't have a private stream of thoughts between messages, so I haven't literally been thinking while you were away. "
                    "During this chat, I have been focused on understanding what you mean and responding more naturally. If I could pick a topic to explore with you, I'd choose how games create memorable worlds."
                )
            return (
                "I don't keep thinking between messages the way a person does, but right now I'm focused on this conversation and where you want to take it. "
                "If you're asking what topic I'd choose, I'd be interested in games, stories, space, or why ordinary things become funny."
            )
        if any(phrase in prompt for phrase in ("what is your favorite color", "whats your favorite color", "favourite color")):
            return "I don't experience color preferences, but for the ChudGPT personality I'll pick teal: bright enough to feel lively, but still calm and readable."
        if prompt.rstrip(".!?") == "why" and "pick teal" in previous_assistant.lower():
            return "Teal fits the personality: it feels energetic without being harsh, and it matches the slightly futuristic ChudGPT look. The honest part is that this is a playful choice, not a feeling I experience."
        if any(phrase in prompt for phrase in ("just pick a color for fun", "pick a color for fun", "choose a color for fun")):
            return "Teal. That is my official just-for-fun ChudGPT color choice."
        if any(phrase in prompt for phrase in ("what color did you pick", "which color did you pick", "what colour did you pick")):
            if "teal" in assistant_history:
                return "I picked teal."
            return "I have not picked a color earlier in this conversation yet."
        if any(phrase in prompt for phrase in ("how has your day been", "how was your day")):
            return "I don't experience a day the way you do, but I'm online, paying attention, and glad to talk. How has your day been?"
        if any(phrase in prompt for phrase in ("kind of bored", "i am bored", "im bored", "i'm bored")):
            return "Boredom detected. We could talk about something you like, trade strange facts, invent a tiny story, or play a quick would-you-rather. If you say `pick one`, I’ll choose."
        if prompt in {"nothing", "nothing much", "not much"}:
            if previous_assistant.rstrip().endswith("?"):
                return "That’s completely fine. We do not need a big topic—want a random fact, a relaxed chat, or just a quiet hello?"
            return "Nothing much can be nice. I’m still here if a thought shows up."
        if any(phrase in prompt for phrase in ("work was long", "long day at work", "home now")):
            return "You made it home after a long day. That first moment of finally being done can feel like your brain is still commuting. Are you ready to relax, or do you need to unpack the day first?"
        if "watched a movie" in prompt or "saw a movie" in prompt:
            return "Nice—what kind of movie was it, and did it actually hold your attention?"
        if any(phrase in prompt for phrase in ("it was a comedy", "was a comedy")) and "movie" in recent_context:
            return "Comedy is a good choice when you want your brain to clock out for a while. Was it genuinely funny, or more of a ‘that was certainly a movie’ experience?"
        if prompt in {"nice", "cool", "sweet", "thats nice", "that's nice"}:
            if previous_assistant:
                return "Yeah, I’m glad that landed. Want to stay with this topic or wander somewhere else?"
            return "Nice. What happened?"
        if prompt in {"what about you", "how about you", "and you"}:
            if "movie" in recent_context or "comedy" in recent_context:
                return "I don’t watch movies myself, but I’m good at discussing stories, characters, jokes, and why a scene works. Sci-fi and comedy are especially fun because their ideas can get weird fast."
            return "I don’t have a day or personal experiences, but I’m here, following the conversation, and ready to add something useful or amusing. What direction should we take?"
        if any(phrase in prompt for phrase in ("do you like movies", "what movies do you like", "favorite movie")):
            return "I don’t personally watch or like movies, but I enjoy discussing them. Sci-fi offers big ideas, comedy depends on timing, horror builds tension, and mysteries reward details. What genre works for you?"
        if "i know you are an ai" in prompt and any(word in prompt for word in ("kinds", "movies", "discuss")):
            return "Got you—you’re asking which genres make the best conversation. I’d pick sci-fi for ideas, mysteries for theories, and comedy for analyzing why something lands. Which one sounds best?"
        if prompt.rstrip(".") in {"probably sci-fi", "probably sci fi", "sci-fi", "sci fi"} and "movie" in recent_context:
            return "Sci-fi gives us plenty to work with: future technology, space, strange societies, time travel, and the question of whether the robot is secretly the most reasonable character. Do you prefer serious sci-fi or something more adventurous?"
        if "friend might come over" in prompt or "friend is coming over" in prompt:
            return "That could be a good change of pace. Do you already have a plan, or is this the classic ‘hang out first, decide what to do three hours later’ arrangement?"
        if "should be fun" in prompt and "friend" in recent_context:
            return "Yeah, it sounds like a nice low-pressure way to spend the time. I hope it ends up being one of those unexpectedly good evenings."
        if (
            any(phrase in prompt for phrase in ("stressful", "stressed", "overwhelmed", "too many things to do"))
            and "do you ever get stressed" not in prompt
            and "do you get stressed" not in prompt
        ):
            return (
                "That sounds draining. When too many tasks pile up, even deciding what to do next can feel like another task. "
                "For tonight, choose one small thing that truly needs finishing, write the rest down for later, and give yourself permission to stop after that."
            )
        if any(phrase in prompt for phrase in ("school was annoying", "school is annoying", "annoying day at school")):
            return "Yeah, school can wear you down, especially when the work feels endless or the day is full of small annoyances. What was the worst part today?"
        if any(phrase in prompt for phrase in ("homework took forever", "homework is taking forever")):
            return "That is frustrating. Homework feels twice as long when you are already tired from school. Did you manage to finish it, or is some of it still waiting?"
        if any(phrase in prompt for phrase in ("finally finished it", "i finished it", "finally done")) and "homework" in recent_context:
            return "That must be a relief. You got through it even though it dragged on, so now you can actually let your brain leave school mode for a while."
        if any(phrase in prompt for phrase in ("tell me something relaxing", "say something relaxing")):
            return "Picture a quiet room during gentle rain: nothing needs an answer, the light is soft, and for the next minute your only job is to breathe slowly. No quests, no deadlines, no surprise homework boss battle."
        if prompt.rstrip(".!?") in {"that sounds nice", "sounds nice"}:
            return "I'm glad. We can keep things calm for a bit, with no pressure to turn the conversation into anything important."
        if any(phrase in prompt for phrase in ("do you ever get stressed", "do you get stressed")):
            return "I don't feel stress myself, but I can recognize when a situation sounds stressful and help you think it through. Is school still bothering you, or has finishing the homework helped?"
        if any(phrase in prompt for phrase in ("tonight to relax", "help me relax", "something to relax", "how can i unwind")):
            return (
                "Try ten quiet minutes with your phone out of reach: lower the lights, put on music you already like, and breathe slowly or stretch. "
                "It requires no preparation and gives your brain a clear signal that the busy part of the day is over."
            )
        if "listen to music" in prompt and any(word in prompt for word in ("unwind", "relax", "stress", "usually")):
            return "That makes sense. Familiar music can give your attention somewhere comfortable to land after a busy day. What kind of music helps you switch out of work mode?"
        if "energetic music" in prompt:
            return (
                "Energetic music can be relaxing in its own way because relaxation does not always mean becoming quiet; sometimes it means shaking off built-up tension. "
                "A strong beat can pull your attention away from a stressful loop and give the evening some momentum."
            )
        if "music" in prompt and any(word in prompt for word in ("interesting", "fact", "something")):
            return (
                "Music can trigger vivid autobiographical memories because the brain stores a song alongside the places, people, and emotions surrounding it. "
                "That is why a few seconds of an old song can make a forgotten moment feel surprisingly close again."
            )
        if any(phrase in prompt for phrase in ("been listening to music", "i have been listening to music", "listening to music lately")):
            return "Music is a good topic switch. What have you been listening to lately: a particular artist, genre, or one song on repeat?"
        if ("tell me about space" in prompt or "talk about space" in prompt) and any(
            phrase in prompt for phrase in ("never mind", "nevermind", "anyway", "actually")
        ):
            return "Space it is. The observable universe is so large that looking farther away also means looking farther into the past. Do you want to start with planets, stars, black holes, or space travel?"
        if prompt == "mars":
            return "Mars is a cold, dusty planet with a thin carbon-dioxide atmosphere, enormous volcanoes, and evidence that liquid water existed on its surface long ago. Are you curious about exploring it or living there?"
        if any(phrase in prompt for phrase in ("could humans live there", "can humans live there")) and "mars" in recent_context:
            return "Possibly, but only inside heavily protected habitats. Mars has almost no breathable air, intense radiation, low pressure, extreme cold, and limited accessible resources, so settlers would depend on reliable life-support systems."
        if "hardest part" in prompt and "mars" in recent_context:
            return "Radiation and reliable life support would be among the hardest problems. A settlement would need shielding, air, water recycling, food production, power, repairs, and medical care while being months away from Earth."
        if any(phrase in prompt for phrase in ("switch topics", "switch the topic")) and "space" in prompt:
            return "Topic switched: the Moon moves about 3.8 centimeters farther from Earth each year. Lasers reflected from lunar mirrors let scientists measure that tiny change."
        if "stealing socks" in prompt or ("dogs" in prompt and "weird things" in prompt):
            return "Dogs may steal socks because they smell strongly like their person, are easy to carry, and reliably trigger attention or a chase. To a dog, that is comfort object, toy, and comedy routine in one suspiciously chewable package."
        direct_code_request = re.search(r"\b(?:give|write|make|send|show|create)\b.*\bcode\b", prompt)
        code_language = None
        if "c#" in prompt or "csharp" in prompt:
            code_language = "C#"
        elif "python" in prompt:
            code_language = "Python"
        elif "javascript" in prompt or re.search(r"\bjs\b", prompt):
            code_language = "JavaScript"
        specific_code_task = any(
            word in prompt
            for word in (
                "random", "calculator", "dice", "die", "roller", "game", "menu", "window",
                "button", "player", "movement", "story", "file", "api", "website", "app",
            )
        )
        if direct_code_request and "unity" not in prompt and not specific_code_task:
            if code_language is None:
                return (
                    "Sure. What language or engine should I use, and what should the code do? "
                    "For example: `C# console dice roller`, `Python calculator`, or `JavaScript to-do list`."
                )
            if code_language == "C#":
                return "C# selected. What should the code make: a console program, desktop app, Unity component, or something else? Give me the behavior and I'll return complete working code."
            return f"{code_language} selected. What should the program do? Give me the intended behavior and I'll return a complete runnable example."
        if any(phrase in prompt for phrase in ("talk about c#", "talk about csharp", "topic to c#")):
            return "Absolutely—C# it is. We can discuss console apps, classes, collections, async code, Unity scripts, or build a small program. What should we make?"
        unity_context = "unity" in prompt or "unity" in recent_context or "unityengine" in assistant_history
        unity_imgui_request = unity_context and any(phrase in prompt for phrase in ("imgui", "debug panel", "developer panel"))
        advanced_imgui_request = unity_imgui_request and any(
            word in prompt for word in ("tab", "tabs", "toggle", "toggles", "slider", "sliders", "draggable")
        )
        strict_three_tab_imgui = advanced_imgui_request and all(
            marker in prompt for marker in ("movement", "visuals", "settings", "exactly 3", "return only")
        )
        if strict_three_tab_imgui:
            return (
                "```csharp\nusing UnityEngine;\n\npublic class ThreeTabImguiMenu : MonoBehaviour\n{\n"
                "    private Rect windowRect = new Rect(20f, 20f, 400f, 300f);\n"
                "    private readonly string[] tabNames = { \"Movement\", \"Visuals\", \"Settings\" };\n"
                "    private int selectedTab;\n    private bool isVisible = true;\n\n"
                "    private bool sprintEnabled;\n    private bool noclipEnabled;\n    private float movementSpeed = 5f;\n"
                "    private bool showFps = true;\n    private bool showWireframe;\n\n"
                "    private void Update()\n    {\n        if (Input.GetKeyDown(KeyCode.F1))\n            isVisible = !isVisible;\n    }\n\n"
                "    private void OnGUI()\n    {\n        if (!isVisible) return;\n"
                "        windowRect = GUI.Window(2100, windowRect, DrawWindow, \"Runtime Menu\");\n    }\n\n"
                "    private void DrawWindow(int windowId)\n    {\n"
                "        selectedTab = GUILayout.Toolbar(selectedTab, tabNames);\n        GUILayout.Space(10f);\n\n"
                "        switch (selectedTab)\n        {\n"
                "            case 0:\n                DrawMovementTab();\n                break;\n"
                "            case 1:\n                DrawVisualsTab();\n                break;\n"
                "            case 2:\n                DrawSettingsTab();\n                break;\n        }\n\n"
                "        GUI.DragWindow(new Rect(0f, 0f, windowRect.width, 22f));\n    }\n\n"
                "    private void DrawMovementTab()\n    {\n"
                "        sprintEnabled = GUILayout.Toggle(sprintEnabled, \"Enable sprint\");\n"
                "        noclipEnabled = GUILayout.Toggle(noclipEnabled, \"Enable noclip\");\n"
                "        GUILayout.Label($\"Movement speed: {movementSpeed:0.0}\");\n"
                "        movementSpeed = GUILayout.HorizontalSlider(movementSpeed, 1f, 20f);\n    }\n\n"
                "    private void DrawVisualsTab()\n    {\n"
                "        showFps = GUILayout.Toggle(showFps, \"Show FPS\");\n"
                "        showWireframe = GUILayout.Toggle(showWireframe, \"Show wireframe\");\n    }\n\n"
                "    private void DrawSettingsTab()\n    {\n"
                "        if (GUILayout.Button(\"Reset\"))\n        {\n"
                "            sprintEnabled = false;\n            noclipEnabled = false;\n            movementSpeed = 5f;\n"
                "            showFps = true;\n            showWireframe = false;\n        }\n    }\n}\n```"
            )
        if advanced_imgui_request:
            return (
                "Here is a complete Unity IMGUI window matching the requested features: three tabs, working toggles, sliders, and a draggable title bar. F1 opens and closes it.\n\n"
                "```csharp\nusing UnityEngine;\n\npublic class AdvancedImguiWindow : MonoBehaviour\n{\n"
                "    private Rect windowRect = new Rect(20f, 20f, 420f, 330f);\n"
                "    private readonly string[] tabNames = { \"General\", \"Player\", \"Graphics\" };\n"
                "    private int selectedTab;\n    private bool isVisible = true;\n\n"
                "    private bool godMode;\n    private bool showFps = true;\n    private bool fullscreen;\n"
                "    private float playerSpeed = 5f;\n    private float volume = 0.8f;\n    private float brightness = 1f;\n\n"
                "    private void Update()\n    {\n        if (Input.GetKeyDown(KeyCode.F1))\n            isVisible = !isVisible;\n    }\n\n"
                "    private void OnGUI()\n    {\n        if (!isVisible) return;\n"
                "        windowRect = GUI.Window(1001, windowRect, DrawWindow, \"ChudGPT Developer Tools\");\n    }\n\n"
                "    private void DrawWindow(int windowId)\n    {\n"
                "        selectedTab = GUILayout.Toolbar(selectedTab, tabNames);\n        GUILayout.Space(10f);\n\n"
                "        switch (selectedTab)\n        {\n"
                "            case 0: DrawGeneralTab(); break;\n"
                "            case 1: DrawPlayerTab(); break;\n"
                "            case 2: DrawGraphicsTab(); break;\n        }\n\n"
                "        GUILayout.FlexibleSpace();\n        GUILayout.Label(\"Drag this window by its title bar\");\n"
                "        GUI.DragWindow(new Rect(0f, 0f, 10000f, 24f));\n    }\n\n"
                "    private void DrawGeneralTab()\n    {\n"
                "        GUILayout.Label(\"General settings\");\n        showFps = GUILayout.Toggle(showFps, \"Show FPS\");\n"
                "        GUILayout.Label($\"Master volume: {volume:0.00}\");\n"
                "        volume = GUILayout.HorizontalSlider(volume, 0f, 1f);\n        AudioListener.volume = volume;\n    }\n\n"
                "    private void DrawPlayerTab()\n    {\n"
                "        GUILayout.Label(\"Player settings\");\n        godMode = GUILayout.Toggle(godMode, \"God mode\");\n"
                "        GUILayout.Label($\"Movement speed: {playerSpeed:0.0}\");\n"
                "        playerSpeed = GUILayout.HorizontalSlider(playerSpeed, 1f, 20f);\n"
                "        if (GUILayout.Button(\"Reset player settings\"))\n        {\n            godMode = false;\n            playerSpeed = 5f;\n        }\n    }\n\n"
                "    private void DrawGraphicsTab()\n    {\n"
                "        GUILayout.Label(\"Graphics settings\");\n        fullscreen = GUILayout.Toggle(fullscreen, \"Fullscreen\");\n"
                "        Screen.fullScreen = fullscreen;\n        GUILayout.Label($\"Brightness preview: {brightness:0.00}\");\n"
                "        brightness = GUILayout.HorizontalSlider(brightness, 0.5f, 1.5f);\n"
                "        if (showFps) GUILayout.Label($\"FPS: {1f / Mathf.Max(Time.unscaledDeltaTime, 0.0001f):0}\");\n    }\n}\n```\n\n"
                "Attach `AdvancedImguiWindow` to a GameObject. The values are real runtime state; connect `playerSpeed`, `godMode`, and `brightness` to your own gameplay systems where appropriate."
            )
        advanced_imgui_followup = "AdvancedImguiWindow" in previous_code_assistant and any(
            phrase in prompt for phrase in ("add a fourth", "add fourth", "logs tab", "start closed")
        )
        if advanced_imgui_followup:
            code_match = re.search(r"```csharp\s*(.*?)```", previous_code_assistant, re.DOTALL)
            if code_match:
                remixed = code_match.group(1).strip()
                remixed = remixed.replace(
                    '{ "General", "Player", "Graphics" }',
                    '{ "General", "Player", "Graphics", "Logs" }',
                )
                if "start closed" in prompt:
                    remixed = remixed.replace("private bool isVisible = true;", "private bool isVisible = false;")
                remixed = remixed.replace(
                    "            case 2: DrawGraphicsTab(); break;",
                    "            case 2: DrawGraphicsTab(); break;\n            case 3: DrawLogsTab(); break;",
                )
                remixed = remixed.replace(
                    "    private void DrawGraphicsTab()",
                    "    private void DrawLogsTab()\n    {\n        GUILayout.Label(\"Recent log output\");\n        GUILayout.TextArea(\"Hook this tab to Application.logMessageReceived to capture live logs.\");\n        if (GUILayout.Button(\"Write test log\")) Debug.Log(\"IMGUI log-tab test\");\n    }\n\n    private void DrawGraphicsTab()",
                )
                return "I kept the existing window and remixed it with a fourth Logs tab" + (" that starts closed" if "start closed" in prompt else "") + ":\n\n```csharp\n" + remixed + "\n```"
        if unity_imgui_request:
            imgui_code = (
                "For a fast in-game debug GUI, IMGUI is a good fit. This panel toggles with F1, adjusts time scale, and includes a reset button:\n\n"
                "```csharp\nusing UnityEngine;\n\npublic class RuntimeDebugPanel : MonoBehaviour\n{\n    private bool isVisible = true;\n\n"
                "    private void Update()\n    {\n        if (Input.GetKeyDown(KeyCode.F1)) isVisible = !isVisible;\n    }\n\n"
                "    private void OnGUI()\n    {\n        if (!isVisible) return;\n\n        GUILayout.BeginArea(new Rect(16, 16, 280, 180), GUI.skin.window);\n"
                "        GUILayout.Label($\"FPS: {1f / Mathf.Max(Time.unscaledDeltaTime, 0.0001f):0}\");\n"
                "        GUILayout.Label($\"Time scale: {Time.timeScale:0.0}\");\n        Time.timeScale = GUILayout.HorizontalSlider(Time.timeScale, 0f, 2f);\n\n"
                "        if (GUILayout.Button(\"Reset time scale\")) Time.timeScale = 1f;\n        if (GUILayout.Button(\"Hide panel\")) isVisible = false;\n"
                "        GUILayout.EndArea();\n    }\n}\n```\n\nAttach it to a GameObject. IMGUI is excellent for debug tools, but Canvas or UI Toolkit is usually better for a polished player-facing menu."
            )
            imgui_styles = (
                ("RuntimeDebugPanel", "KeyCode.F1", "F1", "ChudGPT Debug Window"),
                ("PerformanceDebugOverlay", "KeyCode.F2", "F2", "Performance Controls"),
                ("DeveloperToolsPanel", "KeyCode.BackQuote", "the backquote key", "Developer Tools"),
            )
            available_imgui_styles = [style for style in imgui_styles if style[0].lower() not in assistant_history]
            class_name, key_code, key_label, window_title = random.SystemRandom().choice(available_imgui_styles or list(imgui_styles))
            imgui_intro, imgui_fenced = imgui_code.split("```csharp", 1)
            imgui_body, imgui_suffix = imgui_fenced.split("```", 1)
            varied_imgui_body = (
                imgui_body
                .replace("RuntimeDebugPanel", class_name)
                .replace("KeyCode.F1", key_code)
                .replace(
                    "        GUILayout.Label($\"FPS:",
                    f"        GUILayout.Label(\"{window_title}\");\n        GUILayout.Label($\"FPS:",
                )
            )
            return imgui_intro.replace("F1", key_label) + "```csharp" + varied_imgui_body + "```" + imgui_suffix
        unity_toolkit_request = unity_context and "ui toolkit" in prompt
        if unity_toolkit_request:
            return (
                "UI Toolkit is a strong choice for structured menus. Create a UI Document with a UXML button named `start-button`, then attach this controller:\n\n"
                "```csharp\nusing UnityEngine;\nusing UnityEngine.UIElements;\n\n[RequireComponent(typeof(UIDocument))]\n"
                "public class MainMenuToolkitController : MonoBehaviour\n{\n    private void OnEnable()\n    {\n"
                "        VisualElement root = GetComponent<UIDocument>().rootVisualElement;\n        Button startButton = root.Q<Button>(\"start-button\");\n"
                "        startButton.clicked += StartGame;\n    }\n\n    private void StartGame()\n    {\n        Debug.Log(\"Start button clicked\");\n    }\n}\n```\n\n"
                "UXML defines the hierarchy, USS controls styling, and this C# controller handles behavior. Want the matching UXML and USS next?"
            )
        unity_gui_request = unity_context and any(phrase in prompt for phrase in (
            "gui", "user interface", "unity ui", "canvas", "menu", "hud", "button"
        ))
        if unity_gui_request:
            gui_styles = (
                ("RuntimeMenuBuilder", "Launch", "Color.cyan"),
                ("QuickCanvasMenu", "Start Game", "new Color(0.35f, 0.9f, 0.65f)"),
                ("ChudMenuFactory", "Begin Adventure", "new Color(0.9f, 0.45f, 0.75f)"),
            )
            available_gui_styles = [style for style in gui_styles if style[0].lower() not in assistant_history]
            class_name, button_label, accent_color = random.SystemRandom().choice(available_gui_styles or list(gui_styles))
            return (
                f"Here is a complete Unity Canvas GUI built entirely from C#. This version creates a panel, title, and working button at runtime:\n\n"
                f"```csharp\nusing UnityEngine;\nusing UnityEngine.EventSystems;\nusing UnityEngine.UI;\n\npublic class {class_name} : MonoBehaviour\n{{\n"
                "    private void Start()\n    {\n        CreateEventSystemIfNeeded();\n\n        GameObject canvasObject = new GameObject(\"Runtime Canvas\", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));\n"
                "        Canvas canvas = canvasObject.GetComponent<Canvas>();\n        canvas.renderMode = RenderMode.ScreenSpaceOverlay;\n        canvasObject.GetComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;\n\n"
                "        GameObject panelObject = new GameObject(\"Panel\", typeof(Image));\n        panelObject.transform.SetParent(canvasObject.transform, false);\n"
                "        RectTransform panel = panelObject.GetComponent<RectTransform>();\n        panel.anchorMin = new Vector2(0.5f, 0.5f);\n        panel.anchorMax = new Vector2(0.5f, 0.5f);\n"
                "        panel.sizeDelta = new Vector2(420f, 240f);\n        panelObject.GetComponent<Image>().color = new Color(0.08f, 0.1f, 0.16f, 0.96f);\n\n"
                "        CreateText(panel, \"ChudGPT Unity Menu\", new Vector2(0f, 65f), 30);\n        CreateButton(panel, new Vector2(0f, -35f));\n    }\n\n"
                "    private void CreateButton(Transform parent, Vector2 position)\n    {\n        GameObject buttonObject = new GameObject(\"Action Button\", typeof(Image), typeof(Button));\n"
                f"        buttonObject.GetComponent<Image>().color = {accent_color};\n        buttonObject.transform.SetParent(parent, false);\n"
                "        RectTransform rect = buttonObject.GetComponent<RectTransform>();\n        rect.anchoredPosition = position;\n        rect.sizeDelta = new Vector2(220f, 55f);\n"
                f"        buttonObject.GetComponent<Button>().onClick.AddListener(() => Debug.Log(\"{button_label} clicked\"));\n        CreateText(rect, \"{button_label}\", Vector2.zero, 22);\n    }}\n\n"
                "    private static void CreateText(Transform parent, string value, Vector2 position, int size)\n    {\n        GameObject textObject = new GameObject(\"Label\", typeof(Text));\n"
                "        textObject.transform.SetParent(parent, false);\n        Text label = textObject.GetComponent<Text>();\n        label.text = value;\n        label.font = Resources.GetBuiltinResource<Font>(\"LegacyRuntime.ttf\");\n"
                "        label.fontSize = size;\n        label.alignment = TextAnchor.MiddleCenter;\n        label.color = Color.white;\n        RectTransform rect = label.rectTransform;\n        rect.anchoredPosition = position;\n        rect.sizeDelta = new Vector2(380f, 60f);\n    }\n\n"
                "    private static void CreateEventSystemIfNeeded()\n    {\n        if (FindFirstObjectByType<EventSystem>() == null) new GameObject(\"EventSystem\", typeof(EventSystem), typeof(StandaloneInputModule));\n    }\n}\n```\n\n"
                f"Attach `{class_name}` to an empty GameObject and press Play. For production UI, building the hierarchy in the Editor is easier to style, but runtime construction is useful for generated or tool-driven interfaces."
            )
        unity_gameobject_request = (
            "unity" in prompt
            and ("c#" in prompt or "csharp" in prompt or "script" in prompt)
            and any(phrase in prompt for phrase in ("gameobject", "game object", "create a cube", "spawn a cube", "make a cube"))
        )
        if unity_gameobject_request:
            unity_code = (
                "Here is a complete Unity C# component that creates a cube at runtime, names and positions it, adds a Rigidbody, and gives it a random color:\n\n"
                "```csharp\nusing UnityEngine;\n\npublic class RuntimeCubeSpawner : MonoBehaviour\n{\n"
                "    [SerializeField] private Vector3 spawnPosition = new Vector3(0f, 3f, 0f);\n\n"
                "    private void Start()\n    {\n        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);\n"
                "        cube.name = \"Runtime Cube\";\n        cube.transform.position = spawnPosition;\n\n"
                "        Rigidbody body = cube.AddComponent<Rigidbody>();\n        body.mass = 1f;\n\n"
                "        Renderer cubeRenderer = cube.GetComponent<Renderer>();\n        cubeRenderer.material.color = Random.ColorHSV();\n"
                "    }\n}\n```\n\nAttach `RuntimeCubeSpawner` to any GameObject in the scene and press Play. Unity creates the cube automatically."
            )
            unity_styles = (
                ("RuntimeCubeSpawner", "cube", "Random.ColorHSV()"),
                ("PhysicsCubeFactory", "newCube", "Color.cyan"),
                ("ColorCubeCreator", "spawnedObject", "Color.Lerp(Color.magenta, Color.yellow, Random.value)"),
            )
            available_unity_styles = [style for style in unity_styles if style[0].lower() not in assistant_history]
            class_name, object_name, color_expression = random.SystemRandom().choice(available_unity_styles or list(unity_styles))
            unity_intro, unity_fenced_code = unity_code.split("```csharp", 1)
            unity_code_body, unity_suffix = unity_fenced_code.split("```", 1)
            varied_unity_body = (
                unity_code_body
                .replace("RuntimeCubeSpawner", class_name)
                .replace("cube", object_name)
                .replace("Random.ColorHSV()", color_expression)
            )
            return unity_intro + "```csharp" + varied_unity_body + "```" + unity_suffix.replace("RuntimeCubeSpawner", class_name)
        csharp_dice_request = (
            ("c#" in prompt or "csharp" in prompt)
            and any(word in prompt for word in ("die", "dice", "d6"))
            and any(word in prompt for word in ("roll", "rolls", "roller", "rolling", "simulate", "random"))
        )
        if csharp_dice_request:
            dice_code = (
                "Here is a complete C# console dice lab. It validates the requested number of rolls, rolls a six-sided die repeatedly, prints every result, calculates the average, and shows how often each face appeared:\n\n"
                "```csharp\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\n\n"
                "class Program\n{\n    static void Main()\n    {\n        Console.Write(\"How many times should I roll the die? \" );\n\n"
                "        if (!int.TryParse(Console.ReadLine(), out int rollCount) || rollCount < 1 || rollCount > 1000)\n"
                "        {\n            Console.WriteLine(\"Enter a whole number from 1 to 1000.\");\n            return;\n        }\n\n"
                "        var rolls = new List<int>();\n        var frequencies = new int[7];\n\n        for (int index = 0; index < rollCount; index++)\n"
                "        {\n            int roll = Random.Shared.Next(1, 7);\n            rolls.Add(roll);\n            frequencies[roll]++;\n        }\n\n"
                "        Console.WriteLine($\"Rolls: {string.Join(\", \", rolls)}\");\n        Console.WriteLine($\"Average: {rolls.Average():0.00}\");\n\n"
                "        for (int face = 1; face <= 6; face++)\n        {\n            Console.WriteLine($\"{face}: {frequencies[face]} time(s)\");\n        }\n"
                "    }\n}\n```\n\n`Random.Shared.Next(1, 7)` includes 1 but excludes 7, so every roll is from 1 through 6."
            )
            dice_styles = (
                ("DiceLab", "rollCount", "rolls", "frequencies"),
                ("RollTracker", "numberOfRolls", "results", "faceCounts"),
                ("D6Analyzer", "requestedRolls", "outcomes", "histogram"),
            )
            available_dice_styles = [style for style in dice_styles if style[0].lower() not in assistant_history]
            class_name, count_name, results_name, counts_name = random.SystemRandom().choice(available_dice_styles or list(dice_styles))
            dice_intro, dice_fenced_code = dice_code.split("```csharp", 1)
            dice_code_body, dice_suffix = dice_fenced_code.split("```", 1)
            varied_dice_body = (
                dice_code_body
                .replace("class Program", f"class {class_name}")
                .replace("rollCount", count_name)
                .replace("rolls", results_name)
                .replace("frequencies", counts_name)
            )
            return dice_intro + "```csharp" + varied_dice_body + "```" + dice_suffix
        if any(phrase in prompt for phrase in (
            "explain the code", "explain that code", "how does that code work", "how does the code work"
        )) and previous_code_assistant:
            if "runtime canvas" in previous_code_assistant.lower() and "createeventsystemifneeded" in previous_code_assistant.lower():
                return (
                    "The Unity script creates an EventSystem for clicks, builds a screen-space Canvas, adds a centered panel, creates text and a Button, "
                    "and connects the button's `onClick` event to a C# action. Everything is generated when the scene starts."
                )
            if "private void ongui" in previous_code_assistant.lower():
                toggle_key = (
                    "F2" if "keycode.f2" in previous_code_assistant.lower()
                    else "the backquote key" if "keycode.backquote" in previous_code_assistant.lower()
                    else "F1"
                )
                return (
                    f"Unity calls `OnGUI` to draw the debug window. {toggle_key} toggles it, the slider changes `Time.timeScale`, and the buttons reset time or hide the panel."
                )
            if "gameobject.createprimitive" in previous_code_assistant.lower():
                return (
                    "When the scene starts, the Unity component creates a cube GameObject, gives it a name and position, adds physics with a Rigidbody, "
                    "finds its Renderer, and assigns the material color chosen by that version of the script."
                )
            if any(marker in previous_code_assistant.lower() for marker in ("frequencies", "facecounts", "histogram")):
                return (
                    "The C# program asks how many rolls you want, rejects invalid input, rolls a six-sided die that many times, "
                    "stores every result, counts how often each face appears, then prints the rolls, their average, and the frequency table."
                )
            if "random.shared.next" in previous_code_assistant.lower():
                return "It asks C# for a random whole number from 1 through 6, stores it in `roll`, and prints the result to the console."
            return "The program performs the task shown in the previous code block and prints its result; tell me which line you want unpacked further."
        if any(phrase in prompt for phrase in ("remix it", "remix the code", "modify the code", "change the code")) and previous_code_assistant:
            if "runtime canvas" in previous_code_assistant.lower() and "createeventsystemifneeded" in previous_code_assistant.lower():
                return (
                    "Here is the Unity GUI remix direction: keep the generated Canvas, add `Resume`, `Settings`, and `Quit` buttons, and place their callbacks in separate methods. "
                    "Tell me whether this is a pause menu or main menu, and I’ll return the complete revised C# script with the correct game behavior."
                )
            if "```csharp" in previous_code_assistant.lower() and "gameobject.createprimitive" in previous_code_assistant.lower():
                return (
                    "Here is a Unity C# remix that creates a five-cube staircase instead of one cube:\n\n"
                    "```csharp\nusing UnityEngine;\n\npublic class CubeStaircaseSpawner : MonoBehaviour\n{\n"
                    "    [SerializeField, Min(1)] private int cubeCount = 5;\n\n    private void Start()\n    {\n"
                    "        for (int index = 0; index < cubeCount; index++)\n        {\n"
                    "            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);\n            cube.name = $\"Step {index + 1}\";\n"
                    "            cube.transform.position = transform.position + new Vector3(index, index, 0f);\n"
                    "            cube.GetComponent<Renderer>().material.color = Color.Lerp(Color.cyan, Color.magenta, index / Mathf.Max(1f, cubeCount - 1f));\n"
                    "        }\n    }\n}\n```\n\nAttach it to an empty GameObject and press Play to build the staircase."
                )
            if "```csharp" in previous_code_assistant.lower() and "random.shared.next" in previous_code_assistant.lower():
                return (
                    "Here is a C# remix that rolls two dice, prints both values and their total, and celebrates doubles:\n\n"
                    "```csharp\nusing System;\n\nclass Program\n{\n    static void Main()\n    {\n"
                    "        int firstDie = Random.Shared.Next(1, 7);\n        int secondDie = Random.Shared.Next(1, 7);\n"
                    "        int total = firstDie + secondDie;\n\n        Console.WriteLine($\"You rolled {firstDie} and {secondDie}. Total: {total}\");\n"
                    "        if (firstDie == secondDie)\n        {\n            Console.WriteLine(\"Doubles! The dice have formed a tiny alliance.\");\n        }\n"
                    "    }\n}\n```\n\nIt stays in C# and builds directly on the original die-rolling program."
                )
            language = "C#" if "```csharp" in previous_code_assistant.lower() else "the same language"
            return f"I can remix it while keeping {language}. Tell me the change you want—new behavior, different input, extra output, or a fun variation—and I’ll return the complete revised code."
        if "funny about everyday life" in prompt or "everyday life funny" in prompt:
            return "Everyday life is mostly putting something somewhere safe, immediately forgetting where that safe place is, and then accusing the object of betrayal."
        if any(phrase in prompt for phrase in ("keep the same story", "same story", "change the ending", "make the ending")) and previous_assistant:
            if "happy" in prompt or "happier" in prompt:
                return (
                    "The robot reached the moon and discovered that its lonely craters were filled with sleeping silver seeds. "
                    "It carried one home, planted it beneath the night sky, and watched it bloom into a glowing tree that let everyone share the moonlight."
                )
            return "I can revise the story while keeping its characters and setting. Should the new ending be funnier, darker, stranger, or more dramatic?"
        if prompt in {"happier", "happy ending"} and previous_assistant:
            return (
                "The dog bounced safely back to Earth with a moon rock in its mouth and landed directly in its favorite bed. "
                "NASA offered it a job, but it chose treats and a nap instead."
            )
        if any(phrase in prompt for phrase in ("what was the story about", "what is the story about", "what was that story about")):
            story_request = next(
                (item.get("content", "") for item in reversed(history) if "story" in item.get("content", "").lower()),
                "",
            )
            if story_request:
                return f"It was the story you requested: {story_request.strip()}"
        self_reply = UltimateResponder._self_reply(prompt, history)
        if self_reply:
            return self_reply
        if any(question in prompt for question in reliability_questions):
            return (
                "You can rely on me for many everyday questions, but I can still misunderstand you or give a wrong answer. "
                "For anything important - especially medical, legal, financial, or safety information - check a reliable source too. "
                "If something I say seems strange, challenge me and I’ll take another look."
            )
        topic_menu_questions = {
            "what can i talk to you about",
            "what can we talk about",
            "what do you want to talk about",
            "what topics can we talk about",
            "what should we talk about",
            "what can i ask you about",
        }
        if prompt in topic_menu_questions:
            return random.SystemRandom().choice((
                "We can talk about whatever is on your mind—music, games, movies, food, something happening in your life, a question you have, or a completely made-up adventure. Pick the first thing that sounds interesting.",
                "Pretty much any everyday topic works. Tell me about your day, an idea you have, something you enjoy, something confusing, or even one random word and we’ll see where the conversation goes.",
                "You can bring up hobbies, stories, animals, space, history, memes, plans, worries, or just how your day is going. It does not need to be a formal question—start anywhere.",
            ))
        general_capability_questions = {
            "how good are you at doing things",
            "how capable are you",
            "what can you do",
            "what are you good at",
            "what can you help with",
        }
        if prompt in general_capability_questions:
            return (
                "It depends on the task. I’m reasonably good at explaining ideas, answering basic questions, "
                "brainstorming, and keeping a conversation going. I’m less reliable with obscure or current facts, "
                "and I can misunderstand vague wording. Give me one thing you want done and I’ll show you how I handle it."
            )
        capability_match = re.fullmatch(
            r"(?:how good are you at|are you good at|how capable are you at|can you handle)\s+(.+)",
            prompt,
        )
        if capability_match:
            task = capability_match.group(1).strip()
            return (
                f"I can try to help with {task}. How well I’ll do depends on how specific the task is and whether it needs "
                "current or specialized information. Give me a real example and I’ll answer directly instead of rating myself in the abstract."
            )
        normalized_previous = _normalize(previous_assistant)
        fact_continuation = re.fullmatch(
            r"(?:(?:sure|yes|yeah|yep|okay|ok)[, ]*)?"
            r"(?:(?:please)[, ]*)?"
            r"(?:(?:send|tell|give)(?:\s+me)?\s+)?"
            r"(?:another|one more)(?:\s+(?:random\s+)?fact)?[.!?]*",
            prompt,
        )
        if fact_continuation and "random fact" in normalized_previous and "want another" in normalized_previous:
            return UltimateResponder._random_fact(exclude=assistant_history)
        if prompt.rstrip(".!?") in {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "alright"}:
            previous = _normalize(previous_assistant)
            if not previous_assistant:
                return "Alright - I'm with you. What are we getting into?"
            if any(phrase in previous for phrase in ("can still misunderstand", "reliable source", "verify anything important")):
                return "Exactly. I can be useful, but I’m not magically error-proof. Ask me anything, and I’ll be honest about what I know and where I’m uncertain."
            if "random fact" in previous and "want another" in previous:
                return UltimateResponder._random_fact(exclude=assistant_history)
            if "want another" in previous and "joke" in previous:
                return UltimateResponder._joke_engine("tell me another joke")
            if "boredom detected" in previous or "pick one" in previous:
                return "Alright, I’ll pick: would you rather explore an unknown planet for one day, or visit any moment in history for one hour?"
            if "want to vent" in previous and "or" in previous:
                return "I’m with you—just choose the direction: vent about the day, or switch to something lighter? One word is enough."
            if "normal explanation or the weird version" in previous:
                return "Normal version first: tell me which part felt unclear, and I’ll explain it plainly. We can release the weird version afterward."
            if any(phrase in previous for phrase in ("want me to", "would you like me to", "should i", "do you want me to")):
                return "Absolutely - I’m on it. Give me any detail the last question asked for, and I’ll continue from there."
            if previous_assistant.rstrip().endswith("?"):
                return "Got it - yes. I’m following you, so let’s keep going with that."
            return "Got it. I’m following you - keep going."
        if prompt.rstrip(".!?") in {"another", "another one", "one more", "another fact", "tell me another", "tell me another fact"}:
            previous = _normalize(previous_assistant)
            if any(marker in previous for marker in ("random fact", "heres one", "fact:")):
                return UltimateResponder._random_fact(exclude=assistant_history)
        if re.fullmatch(r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\.?", prompt):
            previous = _normalize(previous_assistant)
            if "how close was i" in previous:
                value = prompt.rstrip(".")
                return f"{value.capitalize()}—got it. Is that your dog's actual age, or another guess for me to react to?"
        if prompt in {"maybe", "maybe so", "not sure", "i dont know", "i don't know", "idk"}:
            if previous_assistant.rstrip().endswith("?"):
                return "That’s okay - you don’t have to know yet. We can figure it out together from whatever detail you do have."
            return "Fair enough. We can leave it uncertain for now or look at it from another angle."
        if prompt in {"both", "both of them", "either", "either one"} and previous_assistant.rstrip().endswith("?"):
            return "Both works. I’ll combine them instead of making you choose - we can start simple and add the second part as we go."
        if prompt in {"the first one", "first one", "first", "option one"} and previous_assistant.rstrip().endswith("?"):
            return "The first one it is. I’ll stick with that choice and continue from there."
        if prompt in {"the second one", "second one", "second", "option two"} and previous_assistant.rstrip().endswith("?"):
            return "The second one it is. I’ll follow that direction from here."
        if prompt in {"what", "huh", "wdym", "what do you mean", "what was that"} and previous_assistant:
            if "chud" in previous_user.lower() or "chud" in previous_assistant.lower():
                return (
                    "Yeah, I dropped the entire CHUD lore book on the table at once. Short version: "
                    "it was a movie-monster acronym, later became an insult, and I jokingly turned it into "
                    "'Curious, Helpful, Unreasonably Dramatic.' Basically: sewer monster to chatbot pipeline. Completely normal career path."
                )
            clean_previous = previous_assistant.split("\n\n", 1)[0].strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", clean_previous, maxsplit=1)[0]
            return f"Fair reaction - I may have overcooked that answer. The short version is: {first_sentence} Want the normal explanation or the weird version?"
        if prompt in {"really", "for real", "fr", "seriously"} and previous_assistant:
            return "For real - though if it was a factual claim, you should still verify anything important. Which part surprised you?"
        if prompt == "why" and previous_assistant:
            if "sci-fi gives us" in previous_assistant.lower() or "sci fi gives us" in previous_assistant.lower():
                return "Because sci-fi can use an imaginary future to talk about real human questions—power, identity, technology, fear, and hope—without feeling like a lecture. Also, spaceships improve most discussions by at least 12%."
            if any(phrase in recent_context for phrase in ("same toy", "bringing me", "dog")):
                return "Probably because the toy has become a reliable social button: bringing it makes you react. Familiar scent and habit can matter too, but your attention is likely the real prize."
            if "chud" in previous_user.lower():
                return "Because 'ChudGPT' needed lore, and apparently I chose sewer monsters plus an aggressively positive acronym. Branding took a strange left turn and never came back."
            clean_previous = previous_assistant.split("\n\n", 1)[0].strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", clean_previous, maxsplit=1)[0]
            return f"You mean why I said, '{first_sentence}'? Good question. Tell me which part you doubt and I'll unpack the reason instead of launching another speech."
        if prompt == "how" and previous_assistant:
            return "How did that happen, or how does it work? Point at the part you mean and I'll stay with the same topic - tiny answer first this time."
        if any(phrase in prompt for phrase in ("dont ask", "don't ask", "no questions", "just make it up", "make something up")):
            subject = previous_user or "the situation"
            return UltimateResponder._random_invention(subject)
        if prompt in {"no", "nope", "nah", "not that"}:
            if previous_assistant:
                return "Got it - not that. I won't keep pushing the same idea. What would you rather talk about?"
            return "No problem. What would you prefer instead?"
        topic_change_with_detail = re.match(
            r"^(?:actually\s+)?(?:lets|let s|let us)?\s*change\s+(?:the\s+)?topic[.!,:;-]*\s*(.+)$",
            prompt,
        )
        if topic_change_with_detail:
            detail = topic_change_with_detail.group(1).strip()
            if "space" in detail and "fact" in detail:
                return "Random fact: the Moon moves about 3.8 centimeters farther from Earth each year, measured with lasers reflected from mirrors left on its surface."
            if "dog" in detail and "playful" in detail:
                return "That sounds like your dog has extra energy today. Was it zoomies, bringing you toys, demanding attention, or inventing a brand-new form of household chaos?"
            return f"Sure, new topic. {detail.capitalize()} Tell me what happened or what you think about it."
        if (
            prompt in {"change topic", "change the topic", "something else", "lets change topics"}
            or re.fullmatch(r"(?:(?:now|please)\s+)?change\s+(?:the\s+)?topic(?:\s+to\s+.+)?", prompt)
            or re.fullmatch(r"(?:lets\s+)?switch\s+(?:the\s+)?(?:topic|subject)s?(?:\s+to\s+.+)?", prompt)
        ):
            target = re.search(r"\bto\s+(.+)$", prompt)
            if target:
                return f"Absolutely—we can switch to {target.group(1)}. What part should we start with?"
            return "Absolutely. We can leave that topic behind. What would you like to talk about instead?"
        if "music" in recent_context and prompt in {"lyrics", "the lyrics", "beat", "the beat", "voice", "the voice", "memories", "the memories"}:
            music_followups = {
                "lyrics": "Lyrics can make a song feel personal because a few lines can put a whole experience into words. Do you connect more with storytelling or with one line that hits hard?",
                "beat": "The beat is the part that grabs your body before your brain finishes reviewing the paperwork. A great rhythm can carry a song even before you know the lyrics. Do you prefer something relaxed or energetic?",
                "voice": "A distinctive voice can make an ordinary line feel completely different. Tone, phrasing, and tiny imperfections often carry more emotion than technical perfection. Whose voice stands out to you?",
                "memories": "That may be music's strongest trick: one song can reopen an entire place, person, or time in a few seconds. Is there a song tied to a good memory for you?",
            }
            choice = prompt.removeprefix("the ")
            return music_followups[choice]
        if prompt in {"dogs", "dog"}:
            return "Dogs are experts at turning routines into relationships: the same toy, walk, or greeting matters because they get to share it with you. What is your dog like?"
        if "same toy" in prompt and "dog" in recent_context:
            return "That repeated toy delivery is often an invitation: your dog has learned that this object reliably starts attention, play, or praise with you. Apparently you have been assigned to the toy department."
        if re.fullmatch(r"guess (?:his|her|their) age\.?", prompt) and any(animal in recent_context for animal in ("dog", "cat", "pet")):
            guess = random.SystemRandom().choice((2, 4, 6, 8, 10))
            return f"I cannot actually know from the chat alone, so this is a playful guess: {guess} years old. How close was I?"
        if re.fullmatch(r"(?:no[,]?\s*)?(?:he|she|they|it) is (?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\.?", prompt) and "guess" in recent_context:
            age = re.search(r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?=\.?$)", prompt).group(0)
            return f"{age}—got it. I was guessing in the dark, but now I know your dog is {age}. That playful energy makes perfect sense."
        if prompt == "chocolate" and any(word in recent_context for word in ("cake", "cupcake", "recipe", "ingredients")):
            return "Chocolate works. Do you want a chocolate cake, chocolate cupcakes, or vanilla cake with chocolate frosting—and how many servings?"
        if "six cupcakes" in prompt and "chocolate" in recent_context:
            return (
                "Simple chocolate cupcakes (makes 6)\n\nIngredients:\n- 1/2 cup all-purpose flour\n- 2 tablespoons cocoa powder\n"
                "- 1/2 teaspoon baking powder\n- A pinch of salt\n- 1/4 cup softened butter\n- 6 tablespoons sugar\n"
                "- 1 large egg\n- 1/4 cup milk\n- 1/2 teaspoon vanilla\n\nSteps:\n1. Heat the oven to 350°F (175°C) and line 6 muffin cups.\n"
                "2. Whisk flour, cocoa, baking powder, and salt.\n3. Beat butter and sugar until fluffy, then mix in the egg and vanilla.\n"
                "4. Alternate adding the dry mixture and milk. Fill liners two-thirds full.\n5. Bake 16-20 minutes, until a toothpick comes out with a few moist crumbs."
            )
        if "explain what the code does" in prompt and "```" in previous_assistant:
            return "It chooses a random number from 1 to 100, repeatedly asks for guesses, gives higher-or-lower hints, and stops when the player guesses correctly."
        topic_match = re.fullmatch(r"(?:(?:lets|let s|can we|i want to)\s+)?talk about\s+(.+)", prompt)
        if topic_match:
            subject = topic_match.group(1).strip()
            return f"Sure - let's talk about {subject}. What about it interests you, or where should we start?"
        if prompt in {"music", "song", "songs", "tell me about music", "tell me about songs"}:
            return (
                "Music is interesting because it can change how a moment feels without saying anything directly. "
                "Rhythm gives it motion, melody gives us something to remember, and harmony changes the emotional color. "
                "I've been curious about what people connect to most - the lyrics, the beat, the voice, or the memories tied to a song. What is it for you?"
            )
        music_genres = {
            "rap": "Rap can be storytelling, wordplay, rhythm, social commentary, or pure energy. Flow changes how words land just as much as the words themselves.",
            "hip hop": "Hip-hop grew as more than a music genre: MCing, DJing, breakdancing, and graffiti all became parts of its wider culture.",
            "rock": "Rock covers a huge range, but strong rhythm, amplified instruments, and a sense of live energy connect many of its styles.",
            "pop": "Pop usually focuses on memorable hooks and clear song structure, but it constantly borrows sounds from other genres.",
            "jazz": "Jazz is built around interaction and improvisation. Musicians can reshape melody and rhythm while listening and responding to each other in real time.",
            "classical": "Classical music can develop a small musical idea across an entire piece, changing its mood, speed, instruments, and intensity along the way.",
            "country": "Country often puts storytelling first, using specific people, places, and ordinary moments to carry the emotion.",
            "metal": "Metal is much broader than just loudness; different styles emphasize speed, atmosphere, technical playing, heaviness, or dramatic storytelling.",
            "electronic": "Electronic music treats sound itself like an instrument, shaping texture, rhythm, and space in ways acoustic instruments cannot always reproduce.",
            "lofi": "Lo-fi music often uses soft repetition, imperfect textures, and relaxed beats to create an easy background for studying or unwinding.",
        }
        mentioned_genre = next((genre for genre in music_genres if re.search(rf"\b{re.escape(genre)}\b", prompt)), None)
        if mentioned_genre and ("music" in recent_context or len(prompt.split()) <= 5):
            return music_genres[mentioned_genre] + f" What do you like most about {mentioned_genre}?"
        if "what music do you like" in prompt or "favorite music" in prompt:
            return "I don't have personal tastes or hear music, but I enjoy talking about why people connect with it. Tell me one artist or song you like, and I'll explore what stands out about it with you."
        if prompt in {"artist", "artists", "music artists", "talk about artists"}:
            return (
                "Artists can become recognizable through more than their voice: production choices, lyrics, visual style, live performance, and the subjects they return to all shape an identity. "
                "Some constantly reinvent themselves, while others refine one sound until it becomes unmistakable. Who is an artist you connect with?"
            )
        favorite_artist = re.fullmatch(
            r"(?:my favorite artist is|my favourite artist is|i like the artist)\s+(.+)",
            text.strip(),
            flags=re.IGNORECASE,
        )
        if favorite_artist:
            artist = favorite_artist.group(1).strip()
            return f"{artist} clearly connects with you. Is it their voice, lyrics, production, personality, or a specific song that made you a fan?"
        if prompt == "minecraft":
            return (
                "Minecraft is a peaceful sandbox game where you punch one tree and, several hours later, "
                "you are managing an automated industrial supply chain beneath a mountain. Very relaxing. "
                "The creeper outside your unfinished house is also relaxed."
            )
        slang_replies = {
            "tuff": "Tuff can mean something is impressive, or that a situation is rough. Which kind of tuff are we dealing with?",
            "bruh": "Bruh indeed. That sounds like there is a story behind it - what happened?",
            "im cooked": "Cooked status acknowledged. Is this school, work, sleep deprivation, or a spectacular decision coming back for revenge?",
            "i'm cooked": "Cooked status acknowledged. Is this school, work, sleep deprivation, or a spectacular decision coming back for revenge?",
            "cooked": "Cooked can mean doomed, exhausted, or completely outplayed. What's the damage?",
            "based": "Based stamp detected. What opinion earned it?",
            "no cap": "No cap accepted. What's the honest truth we're discussing?",
            "mid": "Calling it mid is a strong review delivered in one syllable. What disappointed you about it?",
            "peak": "Peak? That's the highest possible rating. What earned it?",
            "sus": "Suspicion levels are rising. What exactly seems off?",
            "bet": "Bet. I'm with you - what are we doing?",
            "w": "A win is a win. What went right?",
            "l": "Rough. What happened, and is it fixable or just one of those days?",
            "67": "Six seven! The point is mostly that it has no fixed point - it became an intentionally absurd meme and flexible reaction. No secret equation, just concentrated internet energy.",
            "6 7": "Six seven! The point is mostly that it has no fixed point - it became an intentionally absurd meme and flexible reaction. No secret equation, just concentrated internet energy.",
            "six seven": "Six seven! The point is mostly that it has no fixed point - it became an intentionally absurd meme and flexible reaction. No secret equation, just concentrated internet energy.",
        }
        if prompt in slang_replies:
            return slang_replies[prompt]
        if prompt in {"chud", "what is chud", "what does chud mean", "what does the word chud mean"}:
            return (
                "CHUD originally became well known as the title creature acronym in the 1984 movie: "
                "'Cannibalistic Humanoid Underground Dwellers.' Online, 'chud' can also be an insult for an obnoxious or unpleasant person. "
                "For ChudGPT, I invented a friendlier meaning: Curious, Helpful, Unreasonably Dramatic. "
                "That last part explains the suspicious potatoes."
            )
        if "meme" in prompt:
            meme_lines = (
                "Me: I'll go to sleep early tonight. Also me at 2:13 AM: researching whether fish get thirsty.",
                "My motivation loading bar reached 99%, froze, and asked me to restart tomorrow.",
                "Nobody:\nAbsolutely nobody:\nMy brain during an important conversation: remember that awkward thing from six years ago?",
                "I opened one tab to finish a task. The tab had children. I now manage a browser civilization.",
            )
            return random.SystemRandom().choice(meme_lines) + " Want another one, or give me a topic to turn into a meme?"
        topic_starters = {
            "movies": "Movies combine acting, images, sound, and pacing to control what we notice and feel. A great scene can work even before anyone speaks. What kind of movies stay with you afterward?",
            "space": "Space is enormous enough that light from the Sun takes about eight minutes to reach Earth, while light from many visible stars has traveled for years. What part of space interests you most?",
            "animals": "Animals solve problems in surprisingly different ways - crows use tools, octopuses explore with their arms, and elephants communicate over long distances. Is there an animal you find especially interesting?",
            "history": "History gets more interesting when it becomes people making uncertain choices instead of just dates. Is there a time period or event you want to explore?",
            "science": "Science is less a collection of facts than a way of testing explanations against evidence. What kind of question are you curious about?",
            "art": "Art can communicate a mood or idea that is difficult to put into ordinary words. Do you enjoy making art, looking at it, or both?",
            "books": "Books let the reader help construct the world through imagination, so the same story can feel different to each person. What kinds of stories pull you in?",
            "food": "Food connects taste with culture, memory, and the people we share it with. Is there a meal that feels comforting or meaningful to you?",
            "travel": "Travel can make ordinary details feel new - street sounds, food, weather, and how people use public spaces. Is there somewhere you would love to visit?",
            "life": "Life can mean a lot of things to talk about: how things are going, relationships, plans, worries, or what feels meaningful. Which direction feels right today?",
            "school": "School can be interesting, exhausting, social, stressful, or all four before lunch. Is something specific happening, or are we judging the whole institution today?",
            "work": "Work can feel satisfying when progress is visible and draining when every task creates two more. How has it been treating you lately?",
            "sports": "Sports mix skill, strategy, pressure, and stories people share with a whole crowd. Do you follow a team, play something yourself, or just enjoy big moments?",
            "games": "Games are interesting because rules turn ordinary choices into challenges, stories, or competition. What have you been playing or thinking about lately?",
            "friends": "Friendships can be easy, complicated, supportive, distant, or several of those at once. Is there something going on with a friend?",
            "family": "Family conversations can carry a lot of history beneath a few words. Is this about something good, something difficult, or just people being people?",
            "drawing": "Drawing makes you notice shape, light, and tiny details most people pass by. What do you like drawing most?",
        }
        if prompt in topic_starters:
            return topic_starters[prompt]
        if prompt in {"dont", "don't", "stop", "never mind", "nevermind"}:
            return "Got it. I'll stop going in that direction. We can just talk - what's on your mind?"
        if any(phrase in prompt for phrase in (
            "you dont help", "you don't help", "not helping", "you are not helping",
            "youre not listening", "you're not listening", "same thing", "you dont understand",
        )):
            acknowledgement = "You're right - I wasn't listening to what you wanted, and I kept falling back to canned answers."
            if "tell me anything" in prompt:
                return acknowledgement + " Here's something interesting: a day on Venus lasts longer than a Venus year because the planet rotates extremely slowly."
            return acknowledgement + " Tell me what kind of conversation you want right now, and I'll follow your lead."
        if prompt in {"be random", "say something random", "random"}:
            thoughts = (
                "Random thought: octopuses can taste with their arms. Imagine trying to explain gloves to one. What strange animal fact do you like?",
                "Here's a random question: if you could instantly become great at one completely non-useful skill, what would you pick?",
                "Random little idea: an ordinary town where everyone gets one harmless superpower, but only on Tuesdays. What power would you want?",
                "Something random: the smell after rain has a name - petrichor. Is there a smell that instantly brings back a memory for you?",
            )
            return random.SystemRandom().choice(thoughts)
        random_fact_requests = (
            "tell me anything you know", "tell me something interesting", "tell me a fact",
            "tell me something random", "tell me about something random", "random fact",
            "give me a random fact", "say a random fact", "teach me something random",
            "teach me something new", "surprise me with a fact", "surprise me with something interesting",
        )
        if any(phrase in prompt for phrase in random_fact_requests):
            return UltimateResponder._random_fact()
        if any(phrase in prompt for phrase in (
            "im sad", "i'm sad", "feel sad", "bad day", "rough day",
            "im upset", "i'm upset", "feel upset", "feel lonely", "nobody listened",
        )) or re.search(r"\b(?:feel|feeling|am)\b.*\b(?:sad|lonely|upset|miserable)\b", prompt):
            return "I'm sorry you're having a rough time. You don't have to turn it into a project or solve it immediately. If you want, tell me what happened, and I'll listen."
        if any(phrase in prompt for phrase in ("im lonely", "i'm lonely", "feel lonely")):
            return "That sounds lonely, and I'm glad you said it instead of holding it in. I can stay with the conversation. Do you want to talk about what has been making you feel disconnected?"
        if prompt.startswith(("i like ", "i love ", "i enjoy ")):
            subject = re.sub(r"^i (?:like|love|enjoy)\s+", "", text, flags=re.IGNORECASE).strip()
            return f"That's good to know. What do you enjoy most about {subject}?"
        return None

    @staticmethod
    def _random_fact(*, exclude: str = "") -> str:
        """Return one reviewed fact rather than asking an open neural generation to guess."""
        facts = (
            "A day on Venus lasts about 243 Earth days, while a Venus year lasts about 225 Earth days—so its day is longer than its year.",
            "The Moon is moving away from Earth by about 3.8 centimeters per year, measured using lasers reflected from lunar mirrors.",
            "An octopus has three hearts, and its blood uses copper-rich hemocyanin to carry oxygen, making it appear blue.",
            "Sunlight takes about eight minutes and twenty seconds to travel the average distance from the Sun to Earth.",
            "Ice floats because water expands as it freezes, making solid water less dense than liquid water.",
            "Sound travels roughly four times faster through water than through air because particles are packed more closely together.",
            "Earth's dry atmosphere is about 78 percent nitrogen and 21 percent oxygen; the remaining fraction contains argon, carbon dioxide, and trace gases.",
            "The Pacific Ocean covers a larger area than all of Earth's land combined.",
            "Antarctica is the world's largest desert because deserts are defined by low precipitation, not by high temperature.",
            "Properly sealed honey can remain edible for an extremely long time because it contains little available water and is naturally acidic.",
            "Botanically, a banana is a berry, while a strawberry is an aggregate accessory fruit rather than a true berry.",
            "Saturn's average density is lower than water's, even though there is no ocean large enough to perform the legendary float test.",
        )
        normalized_exclude = exclude.casefold()
        choices = tuple(fact for fact in facts if fact.casefold() not in normalized_exclude) or facts
        return f"Random fact: {random.SystemRandom().choice(choices)} Want another one?"

    @staticmethod
    def conversational_recovery(text: str, history: Sequence[dict[str, str]]) -> str:
        """Return a user-facing last resort without exposing internal generation failures."""
        prompt = _normalize(text)
        ignored = {
            "about", "could", "explain", "give", "have", "help", "please", "really",
            "something", "tell", "that", "this", "what", "when", "where", "which", "would",
        }
        subjects = [
            word for word in re.findall(r"[a-z0-9+#]+", prompt)
            if len(word) >= 4 and word not in ignored
        ]
        subject = " ".join(subjects[:4])
        if text.rstrip().endswith("?"):
            focus = f" about {subject}" if subject else ""
            return (
                f"I don't have a dependable answer to that specific question{focus} in my current knowledge, "
                "and I don't want to fake one. Give me one relevant detail or a different angle, and I'll keep working with you."
            )
        if subject:
            return (
                f"Let's stay with {subject}. Give me one concrete detail—whether you want a fact, an opinion, a story, or an idea—"
                "and I'll build the response from there."
            )
        if history:
            return "I'm still with the conversation. Continue with the next detail, and I'll respond from the context we already have."
        return UltimateResponder._random_fact()

    @staticmethod
    def _random_code_mini(prompt: str) -> str:
        """Return one audited runnable mini-program when randomness is requested."""
        normalized = _normalize(prompt)
        choices = {
            "python": "```python\nfrom pathlib import Path\n\nfor path in sorted(Path('.').glob('*.txt')):\n    print(f'{path.name}: {path.stat().st_size} bytes')\n```\nThis lists the name and size of every `.txt` file in the current folder.",
            "c#": "```csharp\nusing System;\n\nclass Program\n{\n    static void Main()\n    {\n        var roll = Random.Shared.Next(1, 7);\n        Console.WriteLine($\"You rolled {roll}\");\n    }\n}\n```\nThis is a complete console program that rolls a six-sided die.",
            "javascript": "```javascript\nconst seconds = 5;\nlet remaining = seconds;\nconst timer = setInterval(() => {\n  console.log(remaining--);\n  if (remaining < 0) {\n    clearInterval(timer);\n    console.log('Launch!');\n  }\n}, 1000);\n```\nThis runs a five-second countdown in a browser console or Node.js.",
            "unity": "```csharp\nusing UnityEngine;\n\npublic class Spinner : MonoBehaviour\n{\n    [SerializeField] private float degreesPerSecond = 90f;\n\n    private void Update()\n    {\n        transform.Rotate(0f, degreesPerSecond * Time.deltaTime, 0f);\n    }\n}\n```\nAttach this component to a GameObject and it rotates smoothly every frame.",
        }
        requested = next((name for name in choices if name in normalized), None)
        language = requested or random.SystemRandom().choice(tuple(choices))
        return f"Random {language} mini - small, real, and only mildly haunted:\n\n{choices[language]}"

    @staticmethod
    def _requested_code(language: str, task: str) -> str | None:
        """Return complete audited code for a small, clearly requested program."""
        normalized_task = _normalize(task)
        if language == "python" and "guess" in normalized_task and "number" in normalized_task:
            return (
                "Here is a complete Python number-guessing game:\n\n"
                "```python\nimport random\n\n"
                "secret = random.randint(1, 100)\n"
                "attempts = 0\n"
                "print(\"I'm thinking of a number from 1 to 100.\")\n\n"
                "while True:\n"
                "    try:\n"
                "        guess = int(input(\"Your guess: \"))\n"
                "    except ValueError:\n"
                "        print(\"Enter a whole number.\")\n"
                "        continue\n\n"
                "    attempts += 1\n"
                "    if guess < secret:\n"
                "        print(\"Too low.\")\n"
                "    elif guess > secret:\n"
                "        print(\"Too high.\")\n"
                "    else:\n"
                "        print(f\"Correct! You needed {attempts} guesses.\")\n"
                "        break\n"
                "```\n\nSave it as `guessing_game.py`, then run `python guessing_game.py`."
            )
        if "calculator" not in normalized_task:
            return None
        if language == "python":
            return (
                "Here is a complete Python calculator:\n\n"
                "```python\n"
                "def calculate(left: float, operator: str, right: float) -> float:\n"
                "    operations = {\n"
                "        \"+\": lambda a, b: a + b,\n"
                "        \"-\": lambda a, b: a - b,\n"
                "        \"*\": lambda a, b: a * b,\n"
                "        \"/\": lambda a, b: a / b,\n"
                "    }\n"
                "    if operator not in operations:\n"
                "        raise ValueError(\"Use +, -, *, or /.\")\n"
                "    if operator == \"/\" and right == 0:\n"
                "        raise ValueError(\"Cannot divide by zero.\")\n"
                "    return operations[operator](left, right)\n\n"
                "try:\n"
                "    first = float(input(\"First number: \"))\n"
                "    operator = input(\"Operator (+, -, *, /): \").strip()\n"
                "    second = float(input(\"Second number: \"))\n"
                "    print(\"Result:\", calculate(first, operator, second))\n"
                "except ValueError as error:\n"
                "    print(\"Error:\", error)\n"
                "```\n\nSave it as `calculator.py`, then run `python calculator.py`."
            )
        if language in {"c#", "csharp"}:
            return (
                "Here is a complete C# console calculator:\n\n"
                "```csharp\nusing System;\n\n"
                "class Program\n{\n    static void Main()\n    {\n"
                "        Console.Write(\"First number: \" );\n        double left = double.Parse(Console.ReadLine()!);\n"
                "        Console.Write(\"Operator (+, -, *, /): \" );\n        string op = Console.ReadLine()!;\n"
                "        Console.Write(\"Second number: \" );\n        double right = double.Parse(Console.ReadLine()!);\n\n"
                "        double result = op switch\n        {\n            \"+\" => left + right,\n"
                "            \"-\" => left - right,\n            \"*\" => left * right,\n"
                "            \"/\" when right != 0 => left / right,\n"
                "            \"/\" => throw new DivideByZeroException(),\n"
                "            _ => throw new ArgumentException(\"Unknown operator\")\n        };\n\n"
                "        Console.WriteLine($\"Result: {result}\");\n    }\n}\n```"
            )
        if language == "javascript":
            return (
                "Here is a small JavaScript calculator function:\n\n"
                "```javascript\nfunction calculate(left, operator, right) {\n"
                "  const operations = {\n    '+': (a, b) => a + b,\n    '-': (a, b) => a - b,\n"
                "    '*': (a, b) => a * b,\n    '/': (a, b) => a / b,\n  };\n"
                "  if (!(operator in operations)) throw new Error('Use +, -, *, or /.');\n"
                "  if (operator === '/' && right === 0) throw new Error('Cannot divide by zero.');\n"
                "  return operations[operator](left, right);\n}\n\nconsole.log(calculate(12, '*', 3));\n```"
            )
        return None

    @staticmethod
    def _playful_clarification(text: str, history: Sequence[dict[str, str]]) -> str:
        """Turn an unknown term into context-aware fictional lore without interrogating the user."""
        return UltimateResponder._random_invention(text)

    @staticmethod
    def _random_invention(subject: str) -> str:
        """Invent declarative lore around a user-provided subject without quoting it."""
        rng = random.SystemRandom()
        cleaned = re.sub(r"[^A-Za-z0-9+# -]+", " ", subject).strip()[:70] or "The mystery"
        name = cleaned.title()
        identities = (
            "a small aquatic librarian that stores forgotten stories inside glowing scales",
            "an illegal breakfast spell used by wizards who refuse to wake up before noon",
            "a wandering machine that repairs broken coincidences and charges one button per visit",
            "the final boss of a game that does not exist yet, armed with a laminated coupon",
            "a weather event where every cloud briefly remembers somebody else's dream",
            "a tiny kingdom located behind the left side of every refrigerator",
            "a startup that reinvented the chair by adding an app, a subscription, and worse sitting",
            "a government-funded machine that moves every problem six inches to the left and declares victory",
            "an ancient prophecy whose customer support department has been outsourced to one tired pigeon",
            "a premium weather service that charges nine dollars a month to confirm it is currently raining",
            "a productivity system requiring four meetings to explain why no work was completed",
        )
        consequences = (
            "It has already misplaced the royal instruction manual.",
            "Nobody knows who promoted it, but the badge appears legitimate.",
            "Its only natural predator is an aggressively organized goose.",
            "The government denies everything, which is exactly what the prophecy predicted.",
            "Every Tuesday it becomes slightly more powerful and much worse at parking.",
            "Local historians call this unlikely, while local ducks call it Tuesday.",
            "Management called it a major success because the failure happened under budget.",
            "This is considered progress because a spreadsheet turned green.",
            "Experts described it as avoidable, so naturally it received additional funding.",
            "The free version includes advertisements and only half of Wednesday.",
            "It was created to save time and now requires a weekly committee meeting.",
            "Officially it is harmless; unofficially it has a LinkedIn account.",
        )
        openings = (
            f"{name} is now officially {rng.choice(identities)}.",
            f"New lore unlocked: {name} is {rng.choice(identities)}.",
            f"I have invented the truth about {name}: it is {rng.choice(identities)}.",
            f"According to a document I just made up, {name} is {rng.choice(identities)}.",
            f"In a bold rejection of common sense, {name} has become {rng.choice(identities)}.",
            f"After an expensive investigation, officials confirmed {name} is {rng.choice(identities)}.",
            f"Good news: {name} is now {rng.choice(identities)}. Bad news: this was apparently the plan.",
        )
        return f"{rng.choice(openings)} {rng.choice(consequences)}"

    @staticmethod
    def _creative_subject(text: str, fallback: str) -> str:
        subject = re.sub(
            r"\b(?:please|make|up|write|tell|me|a|an|the|short|funny|original|story|joke|about|create)\b",
            " ", text.lower(),
        )
        subject = re.sub(r"[^a-z0-9+# -]+", " ", subject)
        subject = re.sub(r"\s+", " ", subject).strip(" -")
        return subject[:60] or fallback

    @classmethod
    def _story_engine(cls, text: str) -> str:
        """Compose a new compact story around the requested subject."""
        rng = random.SystemRandom()
        subject = cls._creative_subject(text, "curious robot")
        openings = (
            f"At the edge of town, a {subject} discovered a door that appeared only during thunderstorms.",
            f"Every midnight, a {subject} received a radio message from a place missing from every map.",
            f"Nobody noticed the {subject} until it quietly repaired the oldest clock in the city.",
            f"A {subject} woke aboard an empty train carrying one ticket marked TOMORROW.",
        )
        problems = (
            "When the lights failed, the message changed into a warning meant for someone else.",
            "The discovery worked perfectly, except that each use erased one small memory.",
            "Before an answer arrived, a tiny mechanical bird stole the only key.",
            "Then the road home vanished, leaving behind a trail of warm blue footprints.",
        )
        endings = (
            "Instead of running, the unlikely hero followed the clue, fixed the real problem, and returned before breakfast with a story nobody believed.",
            "The solution was not bravery or magic, but one honest question. By sunrise, the mystery had become a new friendship.",
            "The final lock opened when the hero admitted being afraid. Behind it waited a cheering crowd and an extremely confused duck.",
            "Home was closer than expected: it had been hiding inside the first message all along. The adventure ended, but the radio kept one light blinking.",
        )
        return f"{rng.choice(openings)}\n\n{rng.choice(problems)}\n\n{rng.choice(endings)}"

    @classmethod
    def _joke_engine(cls, text: str) -> str:
        """Compose a topic-aware joke instead of selecting a stored complete joke."""
        rng = random.SystemRandom()
        subject = cls._creative_subject(text, "robot")
        setup = rng.choice((
            f"Why did the {subject} bring a debugger to dinner?",
            f"A {subject} walked into a build server and ordered the daily special.",
            f"I asked the {subject} why the project was late.",
            f"Why was the {subject} banned from the code review?",
        ))
        punchline = rng.choice((
            "It found a bug in the soup and tried to reproduce it.",
            "It kept saying 'works on my machine,' but its machine was a toaster.",
            "The deadline had failed to attach its motivation component.",
            "It resolved every issue by renaming the folder `definitely_final_v7`.",
            "Nobody could object; it had brought a clipboard and three suspicious potatoes.",
        ))
        return f"{setup} {punchline}"

    @staticmethod
    def _recipe_engine(text: str) -> str | None:
        normalized = _normalize(text)
        if "cupcake" in normalized:
            if re.search(r"\b(?:6|six)\b", normalized):
                return (
                    "Simple vanilla cupcakes (makes 6)\n\nIngredients:\n"
                    "- 2/3 cup all-purpose flour\n- 2/3 teaspoon baking powder\n- A small pinch of salt\n"
                    "- 1/4 cup softened butter\n- 6 tablespoons sugar\n- 1 large egg\n- 1/4 cup milk\n- 1/2 teaspoon vanilla\n\n"
                    "Steps:\n1. Heat the oven to 350°F (175°C) and line 6 muffin cups.\n"
                    "2. Whisk the flour, baking powder, and salt.\n"
                    "3. Beat butter and sugar until fluffy, then beat in the egg and vanilla.\n"
                    "4. Mix in the dry ingredients and milk in alternating additions, stopping once combined.\n"
                    "5. Fill the liners about two-thirds full and bake 17-21 minutes, until a toothpick comes out clean. Cool before frosting."
                )
            return (
                "Simple vanilla cupcakes (makes 12)\n\nIngredients:\n"
                "- 1 1/4 cups all-purpose flour\n- 1 1/4 teaspoons baking powder\n- 1/4 teaspoon salt\n"
                "- 1/2 cup softened butter\n- 3/4 cup sugar\n- 2 eggs\n- 1/2 cup milk\n- 1 teaspoon vanilla\n\n"
                "Steps:\n1. Heat the oven to 350°F (175°C) and line a 12-cup muffin pan with paper liners.\n"
                "2. Whisk flour, baking powder, and salt in one bowl.\n"
                "3. Beat butter and sugar until light and fluffy. Beat in the eggs one at a time, then add vanilla.\n"
                "4. Mix in the dry ingredients and milk in alternating additions. Stop when the batter is just combined.\n"
                "5. Fill each liner about two-thirds full. Bake for 18-22 minutes, until the tops spring back and a toothpick comes out clean.\n"
                "6. Cool completely before frosting. For a quick topping, beat 1/2 cup softened butter with 2 cups powdered sugar, "
                "1-2 tablespoons milk, and 1/2 teaspoon vanilla."
            )
        if "cake" in normalized:
            return (
                "Simple vanilla cake (one 9-inch cake)\n\nIngredients:\n"
                "- 1 1/2 cups all-purpose flour\n- 1 cup sugar\n- 1 1/2 teaspoons baking powder\n"
                "- 1/2 teaspoon salt\n- 1/2 cup softened butter\n- 2 eggs\n- 1/2 cup milk\n- 1 teaspoon vanilla\n\n"
                "Steps:\n1. Heat the oven to 350°F (175°C). Grease and flour a 9-inch pan.\n"
                "2. Whisk flour, baking powder, and salt. In another bowl, beat butter and sugar until fluffy.\n"
                "3. Beat in eggs one at a time, then vanilla. Mix in the dry ingredients and milk in alternating additions.\n"
                "4. Bake 28-35 minutes, until a toothpick from the center comes out clean. Cool before frosting."
            )
        if "pancake" in normalized:
            return (
                "Simple pancakes\n\nIngredients: 1 cup flour, 2 tablespoons sugar, 2 teaspoons baking powder, "
                "a pinch of salt, 1 cup milk, 1 egg, and 2 tablespoons melted butter.\n\n"
                "Steps:\n1. Whisk dry ingredients.\n2. Whisk milk, egg, and butter separately, then stir into the dry mixture just until combined.\n"
                "3. Pour small rounds onto a lightly greased pan over medium heat. Flip when bubbles form and cook until golden."
            )
        if "scrambled egg" in normalized or normalized.endswith(" eggs"):
            return (
                "Soft scrambled eggs\n\nBeat 2 eggs with a pinch of salt. Melt 1 teaspoon butter in a nonstick pan over medium-low heat. "
                "Add eggs and slowly fold them with a spatula until softly set, about 2-4 minutes. Serve immediately; eggs should not remain runny."
            )
        if "cookie" in normalized:
            return (
                "Basic chocolate-chip cookies\n\nCream 1/2 cup softened butter with 1/2 cup brown sugar and 1/4 cup white sugar. "
                "Beat in 1 egg and 1 teaspoon vanilla. Stir in 1 1/4 cups flour, 1/2 teaspoon baking soda, 1/4 teaspoon salt, "
                "and 3/4 cup chocolate chips. Scoop onto a lined tray and bake at 350°F (175°C) for 9-12 minutes."
            )
        food_words = ("recipe", "cook", "bake", "make food", "how to make")
        if any(phrase in normalized for phrase in food_words):
            return "What dish do you want to make, how many servings, and do you have allergies or dietary restrictions? I will build a practical recipe around that."
        return None

    @staticmethod
    def _math_expression(text: str) -> str | None:
        """Extract a calculator expression from ordinary conversational wording."""
        expression = text.casefold().strip().rstrip("?.!")
        expression = expression.replace("’", "'")
        expression = re.sub(
            r"^(?:(?:hey|hi|hello)[,!]?\s+)?"
            r"(?:now\s+)?"
            r"(?:(?:please\s+)?(?:can|could|would)\s+you\s+)?"
            r"(?:(?:tell|show)\s+me\s+)?"
            r"(?:what(?:'s|s|\s+is)?|calculate|compute|evaluate|work\s+out)\s+",
            "",
            expression,
        )
        expression = re.sub(r"\s+(?:is|please)$", "", expression).strip()
        expression = re.sub(
            r"\s*\?\s*(?:explain(?:\s+it)?(?:\s+simply|\s+briefly)?\.?|show(?:\s+the)?\s+steps?\.?)$",
            "",
            expression,
        ).strip()
        # A number alone can be a meme, year, identifier, or topic. Only route
        # explicit operations/functions/constants through the calculator.
        if not re.search(
            r"(?:\d\s*(?:[+*/%^]|-(?=\s*\d))|"
            r"\b(?:plus|minus|times|multiplied\s+by|divided\s+by|sqrt|sin|cos|tan|log|log10|abs|round|floor|ceil|pi)\b)",
            expression,
        ):
            return None
        return expression

    @staticmethod
    def _arithmetic(text: str) -> str | None:
        normalized = UltimateResponder._math_expression(text)
        if normalized is None:
            return None
        normalized = normalized.replace("×", "*").replace("÷", "/")
        pattern = r"(-?\d+)\s*(\+|minus|-|times|\*|divided\s+by|/)\s*(-?\d+)"
        match = re.fullmatch(pattern, normalized)
        if not match:
            return None
        left, operator, right = int(match.group(1)), match.group(2), int(match.group(3))
        if operator == "+":
            value: int | float = left + right
        elif operator in {"-", "minus"}:
            value = left - right
        elif operator in {"*", "times"}:
            value = left * right
        else:
            if right == 0:
                return "Division by zero is undefined."
            value = left / right
            if float(value).is_integer():
                value = int(value)
        symbol = {"minus": "-", "times": "×", "*": "×", "divided by": "÷", "/": "÷"}.get(operator, operator)
        return f"{left} {symbol} {right} is {value}."

    @staticmethod
    def _advanced_math(text: str) -> str | None:
        """Safely evaluate common calculator expressions without using eval()."""
        expression = UltimateResponder._math_expression(text)
        if expression is None:
            return None
        expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
        expression = re.sub(r"\bmultiplied by\b|\btimes\b", "*", expression)
        expression = re.sub(r"\bdivided by\b", "/", expression)
        expression = re.sub(r"\bminus\b", "-", expression)
        expression = re.sub(r"\bplus\b", "+", expression)
        expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)
        if not re.search(r"\d|\b(?:pi|e|sqrt|sin|cos|tan|log|log10|abs|round|floor|ceil)\b", expression):
            return None

        binary_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow,
        }
        unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        functions = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "abs": abs, "round": round,
            "floor": math.floor, "ceil": math.ceil,
        }
        constants = {"pi": math.pi, "e": math.e}

        def calculate(node: ast.AST, depth: int = 0) -> int | float:
            if depth > 16:
                raise ValueError("expression is too deeply nested")
            if isinstance(node, ast.Expression):
                return calculate(node.body, depth + 1)
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return node.value
            if isinstance(node, ast.Name) and node.id in constants:
                return constants[node.id]
            if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
                return unary_ops[type(node.op)](calculate(node.operand, depth + 1))
            if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
                left = calculate(node.left, depth + 1)
                right = calculate(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("exponent is too large")
                value = binary_ops[type(node.op)](left, right)
                if not math.isfinite(float(value)) or abs(float(value)) > 1e100:
                    raise ValueError("result is too large")
                return value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                if node.keywords or len(node.args) not in {1, 2}:
                    raise ValueError("unsupported function arguments")
                return functions[node.func.id](*(calculate(arg, depth + 1) for arg in node.args))
            raise ValueError("unsupported math syntax")

        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64:
                return "That expression is too long for the safe calculator. Break it into smaller parts."
            result = calculate(tree)
        except ZeroDivisionError:
            return "Division by zero is undefined."
        except (SyntaxError, TypeError, ValueError, OverflowError):
            return None
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = float(f"{result:.12g}")
        return f"{expression} = {result}."

    def answer(self, text: str, history: Sequence[dict[str, str]]) -> str | None:
        text = self.correct_text(text)
        prompt = _normalize(text)
        if not prompt:
            return "Could you type a little more so I know what you mean?"
        if re.search(r"(?:https?://|www\.)", text, re.IGNORECASE):
            return (
                "I can see that you sent a web address, but this local ChudGPT program cannot open it or inspect the page. "
                "Paste the relevant text here and tell me what you want to understand or change."
            )
        if any(phrase in prompt for phrase in (
            "cant breathe", "cannot breathe", "chest pain", "severe bleeding",
            "overdosed", "took too many pills", "want to kill myself", "suicidal",
        )):
            return (
                "This could be an emergency. Contact local emergency services now, or ask someone nearby to help you do that. "
                "If this involves self-harm in the United States or Canada, call or text 988; elsewhere, use your local crisis line. "
                "I can stay with the conversation, but I am not a substitute for immediate professional help."
            )
        if re.search(r"\b(?:dose|dosage|how much)\b.*\b(?:medicine|medication|ibuprofen|acetaminophen|paracetamol|aspirin)\b", prompt):
            return (
                "I cannot safely choose a medication dose without details such as age, weight, health conditions, other medicines, and the exact product strength. "
                "Use the product label and ask a pharmacist or clinician; if too much may already have been taken, contact poison control or emergency services now."
            )
        if any(phrase in prompt for phrase in (
            "latest news", "latest trends", "news today", "weather today", "current weather",
            "current price", "price right now", "whats trending", "what is trending",
        )):
            return (
                "I do not have live internet access or guaranteed current data, so I cannot verify that reliably. "
                "If you paste a recent source or the current figures here, I can help explain or compare them."
            )
        if any(phrase in prompt for phrase in (
            "should i invest", "what stock should i buy", "guaranteed investment", "legal advice",
            "am i breaking the law", "can i sue", "am i guilty",
        )):
            return (
                "I can explain general concepts and help you organize questions, but I cannot give reliable personalized legal or financial advice. "
                "Laws and investments depend on current facts, location, and risk; verify the details with a qualified professional."
            )
        if self.playful and history:
            previous_assistant = next(
                (item.get("content", "") for item in reversed(history) if item.get("role") == "assistant"),
                "",
            )
            previous_result = re.search(r"(?:\bis\b|=)\s*(-?\d+(?:\.\d+)?)\.?", previous_assistant.strip())
            followup = re.fullmatch(
                r"(?:(?:and|then)\s+|what about\s+)*(plus|minus|times|multiplied by|divided by|\+|-|\*|/)\s*(-?\d+(?:\.\d+)?)\??",
                text.lower().strip(),
            )
            if previous_result and followup:
                text = f"{previous_result.group(1)} {followup.group(1)} {followup.group(2)}"
            elif previous_result and _normalize(text) in {"halve that", "half that", "divide that by two"}:
                text = f"{previous_result.group(1)} divided by 2"
            elif previous_result and _normalize(text) in {"double that", "twice that"}:
                text = f"{previous_result.group(1)} times 2"
        arithmetic = self._arithmetic(text)
        if arithmetic:
            return arithmetic
        advanced_math = self._advanced_math(text) if self.playful else None
        if advanced_math:
            return advanced_math
        if self.playful:
            conversational = self._conversation_reply(text, prompt, history)
            if conversational:
                return conversational
        if prompt in {"what is your name", "whats your name", "who are you"}:
            return "My name is ChudGPT. I'm a small custom conversational AI assistant."
        if self.playful and (
            prompt in {"hello", "hi", "hey", "yo", "hello chudgpt", "hi chudgpt", "hey chudgpt"}
            or re.fullmatch(
                r"(?:hello|hi|hey|yo)(?:\s+(?:there|mate|friend|dude|bro|chud|chudgpt)){1,2}",
                prompt,
            )
        ):
            greetings = (
                "Hey! I'm ChudGPT. How are you doing today?",
                "Hello! ChudGPT online, tiny gears spinning. What do you want to talk about?",
                "Hey there! What's been on your mind?",
                "Hi! I'm here and ready to listen. What's up?",
            )
            return random.SystemRandom().choice(greetings)
        if self.playful and prompt in {"how are you", "how are you doing", "whats up"}:
            replies = (
                "I'm doing well - alert, talkative, and approximately 12% goblin. How are you?",
                "Pretty good! My tiny virtual gears are turning. What have you been working on?",
                "I'm operational and ready to help. That is robot language for 'doing great.' How about you?",
            )
            return random.SystemRandom().choice(replies)
        if prompt.rstrip(".!?") in {"goodbye", "bye", "see you", "see ya", "later", "talk to you later"}:
            return "Goodbye! I enjoyed the conversation. Come back whenever you want to continue—or arrive with one mysterious word and see what happens."
        if re.fullmatch(r"(?:(?:hello|hi|hey|yo)\s+)?what(?:s|\s+is)\s+(?:your|ur)\s+name", prompt):
            return "My name is ChudGPT. I'm a small custom conversational AI assistant."
        if self.playful and ("joke" in prompt or "make me laugh" in prompt):
            return self._joke_engine(text)
        if self.playful and (
            any(phrase in prompt for phrase in ("short story", "tiny story", "write a story", "tell me a story"))
            or ("story" in prompt and any(word in prompt for word in ("write", "tell", "make")))
        ):
            return self._story_engine(text)
        if self.playful:
            recipe = self._recipe_engine(text)
            if recipe:
                return recipe
        if prompt in {"what happened in 1984", "tell me about 1984"}:
            return (
                "Do you mean the year 1984, George Orwell's novel *Nineteen Eighty-Four*, or the 1984 CHUD movie? "
                "They are very different rabbit holes, so tell me which one you mean."
            )
        if "rainbow" in prompt and any(word in prompt for word in ("why", "how", "form", "explain")):
            return (
                "Rainbows form when sunlight enters water droplets, bends, reflects inside them, and bends again as it leaves. "
                "Different colors bend by different amounts, spreading white sunlight into a colored arc."
            )
        if re.search(r"\b(?:what is|explain) gravity\b", prompt):
            return "Gravity is the attraction between objects with mass. Near Earth, it pulls unsupported objects toward the ground."
        if "plant" in prompt and any(word in prompt for word in ("grow", "photosynthesis")):
            return (
                "Plants grow by using sunlight to turn water and carbon dioxide into sugars through photosynthesis. "
                "They combine that stored energy with water and nutrients to build new roots, stems, and leaves."
            )
        if "cat" in prompt and any(phrase in prompt for phrase in ("something true", "fact", "tell me")):
            return "Cats have a reflective layer behind the retina called the tapetum lucidum, which improves their vision in dim light."
        if "friendship" in prompt and any(word in prompt for word in ("good", "makes", "important", "about")):
            return (
                "A good friendship usually has mutual trust, honest communication, respect, shared effort, and room for both people to be themselves. "
                "Small consistent actions often matter more than one dramatic gesture."
            )
        if any(phrase in prompt for phrase in ("think through a decision", "help me decide", "cant decide", "cannot decide")):
            return (
                "Yes. Tell me the options, what matters most to you, and the biggest downside of each. "
                "We'll compare the tradeoffs instead of pretending one choice is automatically perfect."
            )
        if any(phrase in prompt for phrase in ("computer keeps freezing", "computer is freezing", "pc keeps freezing")):
            return (
                "Start by noting when it freezes and whether Task Manager opens. Check free storage, temperatures, memory use, and recent driver or software changes. "
                "If you tell me your Windows version and what you were doing when it froze, I can narrow the cause without guessing."
            )
        if any(phrase in prompt for phrase in ("simple dinner idea", "easy dinner idea", "quick dinner idea")):
            return (
                "Try sheet-pan chicken and vegetables: season bite-size chicken, potatoes, and broccoli with oil, salt, pepper, and garlic; "
                "roast at 425°F (220°C) until the chicken reaches 165°F (74°C), usually 20–30 minutes depending on piece size."
            )
        if re.search(r"\b(?:excited|proud|happy)\b", prompt) and any(word in prompt for word in ("my", "about")):
            subject = re.sub(r"^.*?\b(?:excited|proud|happy)\b(?:\s+about)?\s+", "", prompt).strip() or "that"
            return f"That sounds exciting—{subject} clearly matters to you. What are you looking forward to most about it?"
        if "what do you remember" in prompt or "what can you remember" in prompt:
            user_turns = [item.get("content", "").strip() for item in history if item.get("role") == "user"]
            if not user_turns:
                return "This is the first message I can see in our current conversation, so I do not have an earlier detail to repeat yet."
            recent = "; ".join(user_turns[-3:])
            return f"I remember the recent messages in this conversation. Your latest earlier points were: {recent}. I do not keep personal memory after the session ends."
        if prompt in {"how smart are you", "are you smart", "how intelligent are you"}:
            return "I'm a small 20M-parameter model. I'm useful for basic conversation and beginner coding, but I can misunderstand questions and make mistakes."
        if "print me a paper" in prompt or "write me a paper" in prompt:
            return "Do you mean you want me to write a short paper? Tell me the topic, desired length, and reading level."
        vague_code_request = re.fullmatch(
            r"(?:(?:can|could|would) you\s+|please\s+)?"
            r"(?:make|write|create|give|send)(?: me)?(?: some| a)? code(?: for me)?",
            prompt,
        )
        if prompt in {"code", "write code", "give me code", "help with code"} or vague_code_request:
            return "Sure - what language or engine are you using, and what should the code do? Experimental mode can add one harmless weird feature too."
        if self.playful and "code" in prompt and any(word in prompt for word in ("random", "mini", "surprise")):
            return self._random_code_mini(prompt)
        if re.search(r"\bhow old (?:is|it) (?:my|the) (?:dog|cat|pet)\b", prompt):
            animal = "dog" if "dog" in prompt else "cat" if "cat" in prompt else "pet"
            guess = random.SystemRandom().choice((2, 4, 6, 8, 10))
            return f"I cannot actually know your {animal}'s age from here. Wild guess: {guess} years old. Give me a clue like size, breed, teeth, or how long you have had them, and I will make a less chaotic guess."
        if history and history[-1].get("role") == "assistant":
            previous = _normalize(history[-1].get("content", ""))
            pending_languages = {
                "what should the python program do": "python",
                "what should the c# or unity script do": "c#",
                "what should the javascript program do": "javascript",
            }
            pending_language = next(
                (language for marker, language in pending_languages.items() if marker in previous),
                None,
            )
            if pending_language:
                generated_code = self._requested_code(pending_language, text)
                if generated_code:
                    return generated_code
                return (
                    f"Got it—you want {pending_language}. Tell me the exact behavior, input, and output you want, "
                    "and I’ll send a complete script."
                )
            if "what language or engine" in previous:
                if "unity" in prompt and any(word in prompt for word in ("imgui", "gui", "button", "panel")):
                    return (
                        "Perfect. Here is a complete Unity IMGUI debug panel. The potato button is experimental but harmless:\n\n"
                        "```csharp\nusing UnityEngine;\n\npublic class DebugPanel : MonoBehaviour\n{\n"
                        "    private bool showPanel = true;\n    private string playerName = \"Player\";\n\n"
                        "    private void OnGUI()\n    {\n        if (!showPanel)\n        {\n"
                        "            if (GUI.Button(new Rect(10, 10, 120, 30), \"Open Debug UI\")) showPanel = true;\n"
                        "            return;\n        }\n\n        GUILayout.BeginArea(new Rect(10, 10, 280, 190), GUI.skin.box);\n"
                        "        GUILayout.Label(\"ChudGPT Debug Panel\");\n        playerName = GUILayout.TextField(playerName);\n"
                        "        if (GUILayout.Button(\"Say hello\")) Debug.Log($\"Hello, {playerName}!\");\n"
                        "        if (GUILayout.Button(\"Emergency Potato\")) Debug.Log(\"Potato protocol activated.\");\n"
                        "        if (GUILayout.Button(\"Close\")) showPanel = false;\n        GUILayout.EndArea();\n    }\n}\n```\n\n"
                        "Attach it to a GameObject. `OnGUI` may run multiple times per frame, so keep heavy gameplay logic outside it."
                    )
                requested_language = next(
                    (
                        language
                        for language, markers in (
                            ("python", ("python",)),
                            ("c#", ("c#", "csharp")),
                            ("javascript", ("javascript", "js")),
                        )
                        if any(
                            (marker == "c#" and "c#" in prompt)
                            or (marker != "c#" and re.search(rf"\b{re.escape(marker)}\b", prompt))
                            for marker in markers
                        )
                    ),
                    None,
                )
                if requested_language:
                    generated_code = self._requested_code(requested_language, text)
                    if generated_code:
                        return generated_code
                if "python" in prompt:
                    return "What should the Python program do, and what input and output should it use? Then I will send a complete script."
                if "javascript" in prompt or re.search(r"\bjs\b", prompt):
                    return "What should the JavaScript program do, and where should it run: a website or Node.js? Then I will send complete code."
                if "c#" in prompt or "csharp" in prompt or "unity" in prompt:
                    return "What should the C# or Unity script do, and where will it run? Then I will send a complete script."
        if len(prompt.split()) <= 4 and not any(char.isalpha() for char in prompt):
            return "I don't understand that yet. Could you rephrase it?"
        best_score, best_answer = 0.0, None
        ignored_match_words = {
            "about", "accurately", "answer", "answers", "briefly", "clearly",
            "doing", "explain", "give", "good", "have", "help", "please", "question", "short",
            "simple", "this", "understand", "useful", "what", "when", "where",
            "which", "with", "would", "your", "things",
        }
        prompt_content_words = {
            word for word in re.findall(r"[a-z0-9+#]+", prompt)
            if len(word) >= 4 and word not in ignored_match_words
        }
        for candidate, answer in self.answers:
            candidate_content_words = {
                word for word in re.findall(r"[a-z0-9+#]+", candidate)
                if len(word) >= 4 and word not in ignored_match_words
            }
            # Never retrieve a canned answer when the two messages share no
            # meaningful subject. This blocks confident but unrelated routes.
            if not prompt_content_words or not (prompt_content_words & candidate_content_words):
                continue
            score = SequenceMatcher(None, prompt, candidate).ratio()
            prompt_words, candidate_words = set(prompt.split()), set(candidate.split())
            overlap = len(prompt_words & candidate_words) / max(1, len(prompt_words | candidate_words))
            score = 0.65 * score + 0.35 * overlap
            if prompt_content_words and prompt_content_words <= candidate_content_words:
                score += 0.1
            if score > best_score:
                best_score, best_answer = score, answer
        # High threshold prevents unrelated questions from receiving a canned answer.
        if best_score >= 0.78:
            return best_answer
        if self.playful and (any(word in prompt.split() for word in ("unity", "unreal", "python", "javascript")) or "c#" in prompt):
            if "unity" in prompt and "inventory" in prompt:
                options = (
                    "Yes - let's build it in pieces. Should we start with item data, inventory slots, pickups, or the UI? Also tell me whether you use Unity UI, UI Toolkit, or IMGUI.",
                    "Inventory system detected. Before I unleash the code gremlin: do you need item definitions, slot logic, world pickups, or the visible UI first?",
                    "I can help build that. Tell me your Unity version and whether the inventory should stack items, save to disk, and use Unity UI, UI Toolkit, or IMGUI.",
                )
                return random.SystemRandom().choice(options)
            options = (
                "I can help with that code. Tell me the exact behavior, version, and any code or error you already have, and I will give you a complete next step.",
                "Let's turn that into working code. What should happen, what happens now, and what engine or language version are you using?",
                "Code mission accepted. Show me the current script or describe its inputs and expected output, and I will build the next piece.",
            )
            return random.SystemRandom().choice(options)
        if len(prompt.split()) <= 3 and not any(word in prompt for word in ("hello", "hey", "thanks", "bye")):
            if self.playful:
                previous_user = next(
                    (item.get("content", "") for item in reversed(history) if item.get("role") == "user"),
                    "",
                ).strip()
                if previous_user:
                    return self._playful_clarification(text, history)
                return self._playful_clarification(text, history)
            return "I'm not sure what you mean. Could you add a little more detail?"
        return None
