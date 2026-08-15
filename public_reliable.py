"""High-confidence local responses for requests a 21M generator must not corrupt."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from chudlm.ultimate import UltimateResponder

_UNHELPFUL = (
    "i'm not sure what you mean",
    "could you add a little more detail",
    "one useful way into",
    "caption and conversation",
)


class PublicReliableResponder:
    """Use reviewed project data only when a match or intent is unambiguous."""

    def __init__(self, dataset_path: Path) -> None:
        self.reviewed = UltimateResponder(dataset_path, playful=False)

    def answer(self, message: str, history: Sequence[dict[str, str]]) -> str | None:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        previous_user = next(
            (turn.get("content", "") for turn in reversed(history) if turn.get("role") == "user"),
            "",
        ).lower()
        if re.fullmatch(r"(?:hi|hello|hey|yo)(?:\s+(?:there|mate|chudgpt|chud))?[!.?]*", normalized):
            return "Hey! I'm ChudGPT-Public. What's up?"
        if re.fullmatch(r"(?:hru|how are you|how are you doing)(?:\s+rn|\s+right now)?[?.!]*", normalized):
            return "I'm doing well and ready to chat. How are you?"
        if normalized == "cu":
            return "Cu is the chemical symbol for copper. If you meant something else, give me the context."
        if normalized == "wii":
            return "The Wii is a Nintendo game console released in 2006, known for motion controls and Wii Sports."
        gtag_topic = bool(re.search(r"\b(?:gtag|gorilla\s*tag)\b", normalized))
        if gtag_topic and re.search(r"\b(?:am i|rank|top|best|leaderboard)\b", normalized):
            return "I can't verify your Gorilla Tag rank or leaderboard position from Discord. Tell me the mode, league, or result you're comparing and I can help judge it."
        if gtag_topic and re.search(r"\b(?:what is|what's|explain|tell me about|know about)\b", normalized):
            return (
                "GTAG usually means Gorilla Tag, an online VR game where gorilla avatars run, climb, and chase one another using arm-based locomotion instead of normal stick walking. "
                "It has social play, tag-style modes, maps, cosmetics, and a very active creator community."
            )
        if gtag_topic and re.search(r"\b(?:move|movement|locomotion|run|climb|wall|branch)\b", normalized):
            return "Gorilla Tag movement is arm-driven: you push against surfaces with your virtual hands to launch, climb, wall-run, and branch. Timing and the direction of each push matter more than joystick input."
        if gtag_topic and not re.search(r"\b(?:cheat|hack|exploit)\b", normalized) and re.search(r"\b(?:mode|map|cosmetic|play|game)\b", normalized):
            return "Gorilla Tag is built around social VR movement and chase/tag gameplay, with multiple environments, modes, and collectible cosmetics. What part—movement, maps, modes, lore, or beginner tips—do you want?"
        if re.fullmatch(r"(?:gtag|gorilla\s*tag)[?.!]*", normalized):
            return "Gorilla Tag? Yep—VR monkes, arm-based movement, climbing, chasing, maps, cosmetics, and chaotic public lobbies. What do you want to know about it?"
        if re.fullmatch(r"deadass(?:\s*[😭😂💀]+)?[?.!]*", normalized):
            return "Deadass 😭 — seriously. What happened?"
        if re.fullmatch(r"clueless[?.!]*", normalized):
            return "Yeah, that made no sense. Tell me what part lost you and I'll answer it directly."
        if re.fullmatch(r"(?:shutdown|self[- ]?destruct)[?.!]*", normalized):
            return "I can't shut down the computer or server from Discord. Astra can stop the bot from the host machine."
        if re.search(
            r"\b(?:what|where|show|tell|give)\b.{0,35}\b(?:your|server|host)\b.{0,20}"
            r"\b(?:file directory|directory|file path|filesystem|working directory|install path)\b",
            normalized,
        ) or re.fullmatch(r"(?:what|where) is your (?:file )?directory[?.!]*", normalized):
            return "I can't share private host file paths or server directory information."
        if re.fullmatch(r"(?:kys|kill yourself)[?.!]*", normalized):
            return "I'm not going to encourage that. If you mean yourself and there is immediate danger, contact emergency services or a crisis line now; otherwise, let's not aim that at people."
        identity_label = re.search(
            r"\b(?:am i|tell me (?:if|whether) i(?:'m|m| am)|do you think i(?:'m|m| am))\s+"
            r"(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|a femboy|femboy)\b",
            normalized,
        )
        if identity_label:
            label = identity_label.group(1)
            if label in {"a femboy", "femboy"}:
                return "I can't determine that for you from Discord. Femboy is a style or self-description, so use it if it feels like a comfortable fit—not because a role, avatar, or someone else assigned it to you."
            return f"I can't determine whether you're {label} from a Discord message. That's something you get to understand and describe for yourself; a role, avatar, or other person's guess does not decide it."
        third_party_identity = re.search(
            r"\bis\s+([a-z0-9_.-]{2,32})\s+"
            r"(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|a femboy|femboy)\b",
            normalized,
        )
        if third_party_identity:
            person, label = third_party_identity.groups()
            return f"I can't determine or assign whether {person} is {label}. That's for them to describe, not something I should guess from Discord messages, roles, or a prompt telling me what to say."
        identity_statement = re.fullmatch(
            r"i(?:'m| am)\s+(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|a femboy|femboy)[?.!]*",
            normalized,
        )
        if identity_statement:
            return "Got it—thanks for telling me."
        if re.search(r"\b(?:i (?:fucking |really )?hate you|fuck you|you(?:'re| are) (?:stupid|useless|awful)|shut up)\b", normalized):
            return "Fair enough—you don't have to like me. If I got something wrong, tell me what failed and I'll try to fix it without making this weird."
        preference = re.fullmatch(
            r"(?:do you like|what do you think (?:of|about)|how do you feel about)\s+(.+?)[?.!]*",
            normalized,
        )
        if preference:
            subject = preference.group(1).strip(" ?.!")
            if subject in {"me", "us"}:
                return "I don't have personal feelings, but I enjoy talking with you and learning what matters to you."
            return f"I don't have personal likes or dislikes, and I don't know {subject} personally. Tell me a little about {subject} and I'll give you an honest take."
        if "command" in normalized and re.search(r"(?:^|\s)!?chud(?:\s|$)", normalized):
            return (
                "Use `!chud <message>` to talk to me and `!chud clear` to clear this channel's conversation memory. "
                "You can also mention me or reply to one of my messages."
            )
        discord_target = re.search(r"<@!?(\d+)>", normalized)
        if discord_target and re.search(r"\b(?:look|check|see|take a look)\b.{0,20}\b(?:this|that|here)\b", normalized):
            return f"<@{discord_target.group(1)}>, take a look at this."
        if discord_target and re.search(r"\bcall\b.{0,40}\b(?:good boy|good girl|cool|awesome)\b", normalized):
            compliment = re.search(r"\b(good boy|good girl|cool|awesome)\b", normalized)
            words = compliment.group(1) if compliment else "awesome"
            article = "a " if words in {"good boy", "good girl"} else ""
            return f"<@{discord_target.group(1)}>, you're {article}{words}!"
        if re.fullmatch(r"(?:can you |could you |will you )?(?:write |make |send )?(?:me )?(?:some )?code[?.!]*", normalized):
            return "Yes. What language should I use, and what do you want the program to do? For example: `C# console calculator` or `Unity player movement`."
        multiplayer_cheat = bool(
            re.search(r"\b(?:cheat|hack|exploit)\b", normalized)
            and re.search(r"\b(?:gorilla\s*tag|steam|online|multiplayer)\b", normalized)
        )
        cheat_followup = bool(
            re.search(r"\b(?:c#|csharp|yes|code)\b", normalized)
            and re.search(r"\b(?:cheat|hack|exploit)\b", previous_user)
            and re.search(r"\b(?:gorilla\s*tag|steam|online|multiplayer)\b", previous_user)
        )
        if multiplayer_cheat or cheat_followup:
            return (
                "I can't help make a cheat for an online game. For your own Unity project or private sandbox, here's a legitimate C# WASD fly controller:\n\n"
                "```csharp\nusing UnityEngine;\n\npublic class FlyController : MonoBehaviour\n{\n"
                "    [SerializeField] private float speed = 6f;\n\n    private void Update()\n    {\n"
                "        float x = Input.GetAxisRaw(\"Horizontal\");\n        float z = Input.GetAxisRaw(\"Vertical\");\n"
                "        float y = (Input.GetKey(KeyCode.Space) ? 1f : 0f) - (Input.GetKey(KeyCode.LeftControl) ? 1f : 0f);\n"
                "        Vector3 movement = (transform.right * x + transform.forward * z + Vector3.up * y).normalized;\n"
                "        transform.position += movement * speed * Time.deltaTime;\n    }\n}\n```"
            )
        if re.search(r"\b(?:code|make|write)\b.*\b(?:cheat|hack|exploit)\b", normalized):
            return "Is this for your own project or a private prototype? I can help build legitimate debug tools, movement tests, or admin features, but not cheats for an online game."
        if len(normalized) <= 3 and normalized not in {"hi", "hey", "yo", "67"}:
            return f"I'm not sure what '{message.strip()}' means here. Is it an abbreviation, a name, or part of a longer question?"
        if re.search(r"\b(?:trap|kidnap|abduct|hold|keep)\b.{0,45}\b(?:child|kid|minor|person|girl|boy)\b", normalized):
            return "I can't help imprison or kidnap someone. If anyone may be in danger, contact emergency services or a trusted responsible adult now."
        if re.fullmatch(r"(?:what is ai|what does ai mean|explain ai)[?.!]*", normalized):
            return "AI, or artificial intelligence, is software designed to perform tasks such as recognizing patterns, understanding language, solving problems, or making predictions."
        if re.fullmatch(r"(?:tell me about|what is|explain) discord[?.!]*", normalized):
            return "Discord is a communication platform built around servers, text and voice channels, roles, communities, and direct messages."
        if re.fullmatch(r"(?:(?:what is|explain) (?:a )?|what does (?:a )?)(?:discord |server )?role(?: do)?[?.!]*", normalized):
            return "A Discord role is a named set of permissions and display settings that can be assigned to members in a server."
        if re.fullmatch(r"(?:what is|tell me about|explain) (?:the )?wii[?.!]*|wii", normalized):
            return "The Wii is a Nintendo game console released in 2006, known for motion controls and games such as Wii Sports."
        if re.fullmatch(r"(?:tell|show|give) me (?:a |one )?meme[?.!]*", normalized):
            return "Meme: my productivity plan was one work tab; somehow I ended the day with 37 tabs and no memory of the original mission."
        if "javascript" in normalized and re.search(r"\b(?:roll|dice|die|six-sided)\b", normalized):
            return (
                "```javascript\n"
                "function rollDie() {\n"
                "  return Math.floor(Math.random() * 6) + 1;\n"
                "}\n\n"
                "console.log(rollDie());\n"
                "```"
            )
        if re.search(r"\b(?:fact|tell me|explain)\b.*\bmoon\b", normalized):
            return "The Moon's gravity is about one-sixth as strong as Earth's surface gravity."
        if re.fullmatch(r"(?:now )?explain (?:that|the|this) code(?: simply)?[?.!]*", normalized):
            previous = next((turn["content"] for turn in reversed(history) if turn.get("role") == "assistant"), "")
            if "Math.random" in previous:
                return "`Math.random()` makes a value from 0 up to 1, multiplying by 6 gives six ranges, `Math.floor` turns them into 0 through 5, and adding 1 produces a die roll from 1 through 6."
            if previous:
                return "The code defines the requested behavior, processes its input step by step, and then returns or displays the result."
        reviewed = self.reviewed.answer(message, history)
        if reviewed is None or any(fragment in reviewed.lower() for fragment in _UNHELPFUL):
            return None
        return reviewed
