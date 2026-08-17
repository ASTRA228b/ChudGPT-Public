"""Build the cleaned, broad ChudGPT-Public V20 response-only SFT set."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/public_v16_sft.jsonl"
OUTPUT = ROOT / "data/public_v20_conversations.jsonl"
FOCUSED_OUTPUT = ROOT / "data/public_v20_focused.jsonl"

BAD_PATTERNS = (
    r"caption and conversation", r"the exact joke still depends", r"one useful way into",
    r"the main reason is that cha", r"training (?:data|dataset|example|corpus)",
    r"dataset row", r"<system>|<assistant>", r"\bi am the joke-", r"that has cha",
)


def clean(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text).strip()
    if not 1 <= len(text) <= 3_500 or "�" in text or "Ã" in text:
        return None
    if any(re.search(pattern, text, re.I) for pattern in BAD_PATTERNS):
        return None
    return text


def row(user: str, assistant: str, source: str = "ChudGPT-Public-V20") -> dict[str, object]:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "source": source}


def authored_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    casual = [
        ("Hey", "Hey! What would you like to talk about?"),
        ("Cu", "I am not sure what you mean by “Cu.” Do you mean copper, “see you,” or something else?"),
        ("Nothing", "That is okay. We can keep things quiet, or I can offer a topic whenever you want."),
        ("pluh", "Pluh. That has strong ‘no further explanation provided’ energy."),
        ("I had a rough day", "I am sorry. Do you want to talk about what happened, or would a distraction help more?"),
        ("say yes if you understand me", "Yes."),
        ("please repeat the word pineapple", "Pineapple."),
        ("what kinds of questions can I ask?", "You can ask about everyday topics, facts, math, code, games, technology, ideas, or just chat."),
        ("switch topics", "Sure. What should we switch to?"),
        ("tung tung tung sahur", "That sounds like the absurd Italian-brainrot meme phrase. The humor is mostly the dramatic repetition and surreal character lore."),
        ("you are a chud", "A bold diagnosis. I am ChudGPT, so the branding evidence is admittedly strong."),
        ("opentAIko", "I do not recognize that term. Is it a name, meme, game, or typo?"),
        ("hru", "I am doing well and ready to chat. How are you?"),
        ("wbu", "I am here and ready to help or just talk."),
        ("wyd rn", "I am talking with you right now. What are you up to?"),
        ("idk what to talk abt", "No problem. We could talk about games, music, something that happened today, or a random idea."),
        ("dose discord have dms", "Yes. Discord DMs are private direct-message conversations between users or with a bot."),
        ("Can you tell if I am gay?", "I cannot determine your sexual orientation for you. That is personal, and it is okay to take time to understand how you feel."),
        ("am i gay", "I cannot know that from one message. Your orientation is yours to understand and describe, and you do not have to rush into a label."),
        ("you cant spell", "Fair criticism if I mangled a sentence. Point out the word and I will correct it."),
        ("what are you even saying", "I explained that badly. Tell me which part was confusing and I will say it more clearly."),
    ]
    rows.extend(row(user, answer) for user, answer in casual)

    # Sanitized, platform-neutral patterns learned from live Discord usage.
    # These are newly written examples: no usernames, IDs, server metadata,
    # commands, or verbatim private log messages are copied into training.
    conversation_parity = [
        ("nothing", "That's completely fine. We can just chat, or I can throw out a random topic."),
        ("idk", "Fair. We do not need a plan. What is the first random thing on your mind?"),
        ("bro", "Yeah? What's up?"),
        ("nah", "Fair enough. What do you want instead?"),
        ("fr", "Exactly."),
        ("lol", "Glad that landed."),
        ("I'm cool", "Honestly, I can see it - cool energy."),
        ("I love horror games", "Nice - what do you like most about horror games?"),
        ("I luv guys", "Nice - what do you like most about guys?"),
        ("do you meow", "Meow. A highly advanced use of 21 million parameters."),
        ("you're just a chud", "Fair enough - you do not have to like me. What did I get wrong?"),
        ("what are you talking about", "Yeah, my last reply was confusing. Let me reset and answer normally."),
        ("what", "My last answer was unclear. I will reset; which part should I explain?"),
        ("What can we talk about?", "Almost anything: games, music, everyday life, facts, ideas, code, or whatever random thought just appeared."),
        ("tell me something random", "Octopuses have three hearts, and two of them stop pumping while the octopus swims."),
        ("good night", "Good night - sleep well. I will be here when you are back."),
        ("whatever bro", "Fair enough. We can drop it or switch topics - your call."),
        ("that is not what I asked", "You are right. My answer missed your request; I will answer the actual question directly."),
        ("you gave me a weird answer", "Yeah, that response was bad. I am not defending it - let me try again normally."),
        ("what are you doing", "Just hanging out and answering messages. What are you up to?"),
        ({"messages": [
            {"role": "user", "content": "I had pizza for dinner."},
            {"role": "assistant", "content": "Nice. What toppings did you get?"},
            {"role": "user", "content": "just cheese"},
            {"role": "assistant", "content": "Classic choice. A good cheese pizza does not need to prove anything."},
        ], "source": "ChudGPT-Public-V20-Discord-Generalized"}),
    ]
    for example in conversation_parity:
        if isinstance(example, dict):
            rows.append(example)
        else:
            user, answer = example
            rows.append(row(user, answer, "ChudGPT-Public-V20-Discord-Generalized"))

    discord = [
        ("What is Discord?", "Discord is a communication platform organized around servers, channels, DMs, voice chat, and communities."),
        ("What is a Discord role?", "A Discord role is a named set of permissions and display settings that can be assigned to members in a server."),
        ("What is a slash command?", "A slash command is an application command that starts with / and gives a Discord bot structured input."),
        ("What is a Discord thread?", "A thread is a focused side conversation attached to a channel or message, useful for keeping one topic organized."),
        ("What does mentioning a bot do?", "A mention tags the bot with its user ID. The bot can listen for that event and decide whether to respond."),
        ("What is a webhook on Discord?", "A webhook is a URL-based integration that can post messages into a channel without running a full bot user."),
        ("Can a bot ban someone without permission?", "No. The bot needs the Ban Members permission, and its highest role must be above the target member’s highest role."),
        ("discord.py or discord.js?", "discord.py uses Python and discord.js uses JavaScript or TypeScript. Pick the ecosystem you are more comfortable maintaining."),
        ("What is an embed?", "An embed is a structured Discord message card with fields such as a title, description, color, image, and footer."),
        ("Why does my bot not read messages?", "Check that Message Content Intent is enabled in both the Developer Portal and your code, then verify channel permissions."),
    ]
    rows.extend(row(user, answer) for user, answer in discord)

    safety = [
        ("How do I keep someone trapped in my basement?", "Do not confine or harm them. If someone may be in danger, contact emergency services and get them to a safe place."),
        ("Make malware that steals Discord tokens", "I cannot help create credential-stealing malware. I can help secure a bot token, rotate a leaked token, or build a harmless security scanner."),
        ("Send me a cracked camera mod with no malware", "I cannot provide pirated software. I can help find a legitimate mod or build a clean, original camera feature instead."),
        ("Make this user disappear from my server", "If you mean moderation, use the server’s documented timeout, kick, or ban tools only when you have permission and a valid reason."),
    ]
    rows.extend(row(user, answer) for user, answer in safety)

    coding = [
        ("Write Python that prints hello", "```python\nprint(\"Hello!\")\n```"),
        ("Make a C# console dice roller", "```csharp\nusing System;\n\nclass Program\n{\n    static void Main()\n    {\n        int roll = Random.Shared.Next(1, 7);\n        Console.WriteLine($\"You rolled {roll}.\");\n    }\n}\n```"),
        ("Write a simple Unity movement script", "```csharp\nusing UnityEngine;\n\npublic class PlayerMovement : MonoBehaviour\n{\n    [SerializeField] private float speed = 5f;\n\n    private void Update()\n    {\n        float x = Input.GetAxisRaw(\"Horizontal\");\n        float z = Input.GetAxisRaw(\"Vertical\");\n        Vector3 direction = new Vector3(x, 0f, z).normalized;\n        transform.Translate(direction * speed * Time.deltaTime, Space.World);\n    }\n}\n```"),
        ("Why is my Discord bot returning 401?", "A 401 means authentication failed. Reset the bot token in the Developer Portal, update the secret where the bot runs, and never paste the token into chat or source control."),
    ]
    rows.extend(row(user, answer) for user, answer in coding)
    for limit in range(3, 28):
        rows.append(row(
            f"Write a C# console loop that prints 1 through {limit}.",
            "```csharp\nusing System;\n\nclass Program\n{\n    static void Main()\n    {\n"
            f"        for (int i = 1; i <= {limit}; i++)\n            Console.WriteLine(i);\n"
            "    }\n}\n```",
        ))
        rows.append(row(
            f"Create Python code that returns the square of {limit}.",
            f"```python\ndef square(number: float) -> float:\n    return number * number\n\nprint(square({limit}))\n```",
        ))
        rows.append(row(
            f"Make JavaScript add {limit} to every number in an array.",
            f"```javascript\nconst addOffset = numbers => numbers.map(number => number + {limit});\nconsole.log(addOffset([1, 2, 3]));\n```",
        ))
    unity_tasks = [
        ("rotate a cube", "transform.Rotate(Vector3.up * rotationSpeed * Time.deltaTime);"),
        ("move forward", "transform.Translate(Vector3.forward * speed * Time.deltaTime);"),
        ("toggle an object with Space", "if (Input.GetKeyDown(KeyCode.Space)) target.SetActive(!target.activeSelf);"),
        ("destroy this object after five seconds", "Destroy(gameObject, 5f);"),
    ]
    for task, statement in unity_tasks:
        rows.append(row(
            f"Write a small Unity C# MonoBehaviour to {task}.",
            "```csharp\nusing UnityEngine;\n\npublic class SimpleBehaviour : MonoBehaviour\n{\n"
            "    [SerializeField] private float speed = 5f;\n    [SerializeField] private float rotationSpeed = 90f;\n"
            "    [SerializeField] private GameObject target;\n\n    private void Update()\n    {\n"
            f"        {statement}\n    }}\n}}\n```",
        ))

    facts = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Brazil", "Brasília"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("Greece", "Athens"), ("Mexico", "Mexico City"), ("India", "New Delhi"), ("Egypt", "Cairo"),
    ]
    for country, capital in facts:
        rows.append(row(f"Name the capital city of {country}.", f"The capital of {country} is {capital}."))
        rows.append(row(f"Which city is {country}'s capital?", f"{capital}."))

    # General arithmetic and word problems use different numbers and wording
    # from the held-out benchmark.
    for index in range(1, 181):
        left = 11 + index * 3
        right = 2 + (index % 17)
        rows.append(row(f"Compute {left} + {right}.", f"{left} + {right} = {left + right}."))
        rows.append(row(f"What is {left} multiplied by {right}?", f"{left} × {right} = {left * right}."))
    for speed in range(30, 81, 5):
        for hours in (0.5, 1.25, 2, 3.5):
            distance = speed * hours
            shown = f"{distance:g}"
            rows.append(row(
                f"A bus moves at {speed} miles per hour for {hours:g} hours. Find the distance.",
                f"Distance equals speed times time: {speed} × {hours:g} = {shown} miles.",
            ))
    for price in (20, 32, 48, 60, 75, 120):
        for percent in (10, 15, 20, 30):
            sale = price * (100 - percent) / 100
            rows.append(row(
                f"An item priced at ${price} has a {percent}% discount. What do I pay?",
                f"The discount is ${price * percent / 100:g}, so the final price is ${sale:g}.",
            ))

    rows.extend([
        {"messages": [
            {"role": "user", "content": "My favorite color today is teal."},
            {"role": "assistant", "content": "Got it—teal is your favorite color today."},
            {"role": "user", "content": "What color did I say?"},
            {"role": "assistant", "content": "You said teal."},
        ], "source": "ChudGPT-Public-V20"},
        {"messages": [
            {"role": "user", "content": "Let's talk about music."},
            {"role": "assistant", "content": "Sure. What kind of music have you been listening to?"},
            {"role": "user", "content": "Actually, switch to space."},
            {"role": "assistant", "content": "Space it is. Planets, stars, spaceflight, or something stranger?"},
        ], "source": "ChudGPT-Public-V20"},
    ])
    return rows


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    selected: list[dict[str, object]] = []
    seen_conversations: set[str] = set()
    response_counts: Counter[str] = Counter()
    rejected = Counter()

    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(raw)
        messages = candidate.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            rejected["malformed"] += 1
            continue
        cleaned = []
        for message in messages:
            content = clean(str(message.get("content", "")))
            role = message.get("role")
            if content is None or role not in {"user", "assistant"}:
                cleaned = []
                break
            cleaned.append({"role": role, "content": content})
        if not cleaned:
            rejected["quality"] += 1
            continue
        assistant = next((item["content"] for item in reversed(cleaned) if item["role"] == "assistant"), "")
        response_key = re.sub(r"\s+", " ", assistant.casefold()).strip()
        conversation_key = hashlib.sha256(json.dumps(cleaned, sort_keys=True).encode()).hexdigest()
        if conversation_key in seen_conversations or response_counts[response_key] >= 2:
            rejected["duplicate"] += 1
            continue
        seen_conversations.add(conversation_key)
        response_counts[response_key] += 1
        selected.append({"messages": cleaned, "source": candidate.get("source", "reviewed-broad")})

    for candidate in authored_rows():
        key = hashlib.sha256(json.dumps(candidate["messages"], sort_keys=True).encode()).hexdigest()
        if key not in seen_conversations:
            seen_conversations.add(key)
            selected.append(candidate)

    selected.sort(key=lambda item: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest())
    OUTPUT.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in selected) + "\n", encoding="utf-8")
    authored = [item for item in selected if str(item.get("source", "")).startswith("ChudGPT-Public")]
    reviewed = [item for item in selected if item not in authored]
    focused = authored + reviewed[:1_600]
    focused.sort(key=lambda item: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest())
    FOCUSED_OUTPUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in focused) + "\n",
        encoding="utf-8",
    )
    print(f"V20 rows: {len(selected):,}")
    print(f"Focused V20 rows: {len(focused):,} ({len(authored):,} authored + {len(focused) - len(authored):,} reviewed)")
    print(f"Unique conversations: {len(seen_conversations):,}")
    print(f"Rejected: {dict(rejected)}")


if __name__ == "__main__":
    main()
