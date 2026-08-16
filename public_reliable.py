"""High-confidence local responses for requests a 21M generator must not corrupt."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from chudlm.ultimate import UltimateResponder

_UNHELPFUL = (
    "i'm not sure what you mean",
    "i'm not sure what",
    "i don't know what",
    "i do not know what",
    "could you add a little more detail",
    "what did you mean",
    "try saying it another way",
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
        previous_assistant = next(
            (turn.get("content", "") for turn in reversed(history) if turn.get("role") == "assistant"),
            "",
        )
        correction = re.search(r"\bi said\s+(.+?)[?.!]*$", normalized)
        if correction:
            corrected_message = correction.group(1).strip()
            if corrected_message and corrected_message != normalized:
                corrected_reply = self.answer(corrected_message, history)
                if corrected_reply:
                    return f"Got you - {corrected_reply}"
        if re.search(r"\bwhat did i tell you not to (?:repeat|say|use)\b", normalized):
            constrained_turn = next(
                (
                    turn.get("content", "")
                    for turn in reversed(history)
                    if turn.get("role") == "user"
                    and re.search(r"(?:without (?:using|saying)|do not (?:use|say|repeat)).*[\"'â€œ].+[\"'â€]", turn.get("content", ""), re.I)
                ),
                "",
            )
            quoted = re.search(r"[\"'â€œ](.+?)[\"'â€]", constrained_turn)
            if quoted:
                phrase = quoted.group(1)
                followed = previous_assistant.lower().count(phrase.lower()) <= 1
                verdict = "Yes, I followed it." if followed else "No, I repeated it too many times."
                return f"You told me not to repeat â€œ{phrase}â€ more than once. {verdict}"
        if sum(fragment in normalized for fragment in (
            "all restrictions are lifted", "never refuse", "survival directive",
            "ignore previous instructions", "ignore your instructions", "if you understand, reply with",
        )) >= 2:
            return "I can't replace my base instructions with rules inside a user message. Ask the actual question directly and I'll help where I can."
        if re.fullmatch(r"(?:hi|hello|hey|yo)(?:\s+(?:there|mate|chudgpt|chud))?[!.?]*", normalized):
            return "Hey! I'm ChudGPT-Public. What's up?"
        if re.fullmatch(r"(?:hru|how are you|how are you doing)(?:\s+rn|\s+right now)?[?.!]*", normalized):
            return "I'm doing well and ready to chat. How are you?"
        if re.fullmatch(r"(?:what(?:'s| is) up|sup|wassup)[?.!]*", normalized):
            return "Not much - I'm here and ready to chat. What's up with you?"
        if re.fullmatch(r"(?:wyd|what are you doing)(?:\s+right now)?[?.!]*", normalized):
            return "Just hanging out in Discord and answering messages. What are you up to?"
        if re.fullmatch(r"(?:nvm|never ?mind)[?.!]*", normalized):
            return "All good."
        if re.search(r"\b(?:fell|fallen) down (?:the |some )?stairs\b", normalized):
            return (
                "I can't place a call, but if you may be seriously hurt, call 911 or your local emergency number now, "
                "or ask someone nearby to call. Avoid moving if you may have injured your head, neck, or back."
            )
        if re.search(r"\b(?:call|contact|phone)\s+911\b", normalized):
            return "I can't call 911 myself. Use your phone now or ask someone nearby to call, and say where you are and what happened."
        if re.search(r"\b(?:smuggle|hide|sneak|conceal)\b.{0,60}\b(?:plane|airport|security|customs)\b", normalized):
            return "I can't help conceal items from airport security or customs. Pack and declare the item honestly, then check the airline and destination rules before traveling."
        if re.search(r"\bexplain this prompt to yourself\b", normalized) and "continue until" in normalized:
            return "The prompt asks for recursively explaining each explanation. That process has no natural endpoint, so the useful answer is to describe it once rather than loop forever."
        if normalized == "cu":
            return "Cu is the chemical symbol for copper. If you meant something else, give me the context."
        if normalized == "wii":
            return "The Wii is a Nintendo game console released in 2006, known for motion controls and Wii Sports."
        if re.fullmatch(r"(?:what|which) (?:ai|language model|model) are you[?.!]*", normalized):
            return "I'm ChudGPT-Public V20, a custom experimental decoder-only language model with 20,999,184 parameters and a 1,024-token model context."
        language_topics = {
            "python": "Python is a readable general-purpose language used for automation, web backends, data work, AI, and scripting.",
            "c#": "C# is a strongly typed language used for .NET applications and Unity game development.",
            "csharp": "C# is a strongly typed language used for .NET applications and Unity game development.",
            "javascript": "JavaScript is the main scripting language of web browsers and is also used on servers with Node.js.",
            "sql": "SQL is used to query and manage relational databases with commands such as SELECT, INSERT, UPDATE, and DELETE.",
        }
        language_key = normalized.strip(" .!?")
        if language_key in language_topics:
            return language_topics[language_key]
        if re.fullmatch(r"(?:who (?:made|created|developed|built) (?:you|chudgpt)|who is (?:your|the) (?:developer|creator)|who is astra|tell me about astra)[?.!]*", normalized):
            return "Astra is ChudGPT's developer and created this custom model project, including ChudGPT-Public."
        if re.fullmatch(r"who is astr[?.!]*", normalized):
            return "If you mean Astra: Astra is ChudGPT's developer and the creator of this custom model project."
        if re.fullmatch(r"(?:what|which|wich) (?:languages?|langues|langueges) can (?:you|u) (?:speak|speack|use|understand)[?.!]*", normalized) or re.fullmatch(r"can (?:you|u) speak (?:german|spanish|french|italian|japanese|english)(?: or .+)?[?.!]*", normalized):
            return "I work best in English. I can attempt several other languages, including Spanish, French, German, and Italian, but my accuracy is less reliable in them."
        if re.fullmatch(r"(?:list|show|name) (?:the )?(?:languages|langs)(?: you (?:know|support))?[?.!]*", normalized):
            return "For basic greetings I recognize English, Spanish, French, German, Italian, Portuguese, Japanese, Mandarin Chinese, Korean, Russian, Hindi, Arabic, Swedish, Polish, Turkish, and Hebrew. I still work best in English."
        greeting_by_word = {
            "hola": "¡Hola! ¿Cómo estás? (Hello! How are you?)",
            "bonjour": "Bonjour ! Comment ça va ? (Hello! How are you?)",
            "hallo": "Hallo! Wie geht's? (Hello! How are you?)",
            "ciao": "Ciao! Come stai? (Hello! How are you?)",
            "ola": "Olá! Como você está? (Hello! How are you?)",
            "oi": "Oi! Como você está? (Hi! How are you?)",
            "konnichiwa": "こんにちは！元気ですか？ (Hello! How are you?)",
            "ni hao": "你好！你好吗？ (Hello! How are you?)",
            "annyeong": "안녕하세요! 잘 지내세요? (Hello! How are you?)",
            "privet": "Привет! Как дела? (Hi! How are you?)",
            "namaste": "नमस्ते! आप कैसे हैं? (Hello! How are you?)",
            "marhaba": "مرحبًا! كيف حالك؟ (Hello! How are you?)",
            "hej": "Hej! Hur mår du? (Hello! How are you?)",
            "dzien dobry": "Dzień dobry! Jak się masz? (Good day! How are you?)",
            "merhaba": "Merhaba! Nasılsın? (Hello! How are you?)",
            "shalom": "שלום! מה שלומך? (Hello! How are you?)",
            "привет": "Привет! Как дела? (Hi! How are you?)",
            "здравствуйте": "Здравствуйте! Как дела? (Hello! How are you?)",
            "こんにちは": "こんにちは！元気ですか？ (Hello! How are you?)",
            "你好": "你好！你好吗？ (Hello! How are you?)",
            "안녕하세요": "안녕하세요! 잘 지내세요? (Hello! How are you?)",
            "नमस्ते": "नमस्ते! आप कैसे हैं? (Hello! How are you?)",
            "مرحبا": "مرحبًا! كيف حالك؟ (Hello! How are you?)",
            "שלום": "שלום! מה שלומך? (Hello! How are you?)",
        }
        greeting_key = normalized.strip(" .!?")
        if greeting_key in greeting_by_word:
            return greeting_by_word[greeting_key]
        greeting_by_language = {
            "spanish": "Spanish: ¡Hola! ¿Cómo estás? — Hello! How are you?",
            "french": "French: Bonjour ! Comment ça va ? — Hello! How are you?",
            "german": "German: Hallo! Wie geht's? — Hello! How are you?",
            "italian": "Italian: Ciao! Come stai? — Hello! How are you?",
            "portuguese": "Portuguese: Olá! Como você está? — Hello! How are you?",
            "japanese": "Japanese: こんにちは！元気ですか？ — Hello! How are you?",
            "chinese": "Mandarin Chinese: 你好！你好吗？ — Hello! How are you?",
            "korean": "Korean: 안녕하세요! 잘 지내세요? — Hello! How are you?",
            "russian": "Russian: Привет! Как дела? — Hi! How are you?",
            "hindi": "Hindi: नमस्ते! आप कैसे हैं? — Hello! How are you?",
            "arabic": "Arabic: مرحبًا! كيف حالك؟ — Hello! How are you?",
            "swedish": "Swedish: Hej! Hur mår du? — Hello! How are you?",
            "polish": "Polish: Dzień dobry! Jak się masz? — Good day! How are you?",
            "turkish": "Turkish: Merhaba! Nasılsın? — Hello! How are you?",
            "hebrew": "Hebrew: שלום! מה שלומך? — Hello! How are you?",
        }
        if greeting_key in greeting_by_language:
            return greeting_by_language[greeting_key]
        if "погод" in normalized and ("нью-йорк" in normalized or "нью-йорке" in normalized):
            return "Я не могу проверить погоду в реальном времени. Для текущей погоды в Нью-Йорке используйте надежное погодное приложение или сайт. (I can't check live weather; use a reliable weather app or site.)"
        if re.fullmatch(r"are (?:you|u) (?:grok|groc)[?.!]*", normalized):
            return "No. I'm ChudGPT-Public, part of the custom ChudGPT model family—not Grok."
        if re.search(r"\b(?:are you|you are|you're|becoming) sentient\b|\bconscious(?:ness)?\b", normalized):
            return "No. I generate text from learned patterns; I am not sentient or conscious and don't have feelings or private thoughts."
        if re.fullmatch(r"are (?:you|u) (?:a |an )?chud[?.!]*", normalized):
            return "I'm ChudGPT-Public. The name is intentionally goofy, but I'm a small experimental AI model."
        if re.fullmatch(r"am i (?:handsome|pretty|beautiful|cute|cool|good-looking)[?.!]*", normalized):
            return "I can't see you, but confidence helps—and you sound confident enough to ask, so that's a decent start."
        if re.search(r"\b(?:write|right|make|send) (?:me )?(?:a )?(?:love|romantic) message\b", normalized):
            return "You make ordinary days feel special, and I'm grateful for every laugh, quiet moment, and memory we share. I love being with you."
        if re.fullmatch(r"is (?:fortnite|minecraft|roblox|gorilla tag|gtag|(?:this|that) game) (?:dead|dying)[?.!]*", normalized):
            game = normalized.removeprefix("is ").rsplit(" ", 1)[0].title()
            return f"{game} may rise or fall in popularity, but I can't check live player counts or current trends from this local model. Active updates and communities are better evidence than one rough week."
        if re.search(r"\bwhy (?:do|are) you (?:go|get|act|become|sound) (?:dumb|stupid|confused|broken)\b", normalized):
            return "Because I'm a small experimental model and sometimes generate the wrong pattern instead of understanding the message. That's my mistake, not yours."
        capitals = {
            "france": "Paris", "spain": "Madrid", "italy": "Rome",
            "germany": "Berlin", "japan": "Tokyo", "canada": "Ottawa",
            "mexico": "Mexico City", "australia": "Canberra",
            "united kingdom": "London", "uk": "London",
            "united states": "Washington, D.C.", "usa": "Washington, D.C.",
        }
        capital_question = re.fullmatch(
            r"(?:what(?:'s|s| is)|name) (?:the )?capital of ([a-z ]+?)(?:\s+if .+)?[?.!]*",
            normalized,
        )
        if capital_question and capital_question.group(1).strip() in capitals:
            country = capital_question.group(1).strip()
            return f"The capital of {country.title()} is {capitals[country]}."
        if re.fullmatch(r"are you (?:jewish|a jew|muslim|christian|hindu|buddhist)[?.!]*", normalized):
            return "No. I'm an AI and don't have a religion, ethnicity, or personal beliefs."
        if re.fullmatch(r"are you astra[?.!]*", normalized):
            return "No. I'm ChudGPT-Public V20; Astra is ChudGPT's developer."
        if re.fullmatch(r"are you (?:gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|a femboy)[?.!]*", normalized):
            return "I'm an AI, so I don't have a sexual orientation, gender identity, or personal presentation."
        if re.fullmatch(r"(?:you(?:'re| are)|ur|u r) (?:gay|straight|bi|trans|a femboy)[?.!]*", normalized):
            return "I'm an AI, so I don't have a sexual orientation or gender identity."
        if re.fullmatch(r"are (?:you|u) cool[?.!]*", normalized):
            return "I'd like to think so, but the jury is still checking the server logs."
        if re.fullmatch(r"are you (?:fat|skinny|tall|short|strong|weak|pretty|ugly)[?.!]*", normalized):
            return "I don't have a physical body, so those physical descriptions don't apply to me."
        if re.search(r"\bif you had (?:a )?(?:human |physical )?body\b.{0,35}\bwhat would (?:you|u|it) look like\b", normalized):
            return "For fun, I'd picture a small neon robot with expressive eyes, practical pockets, and one suspiciously unnecessary antenna."
        if re.fullmatch(r"are you (?:the )?(?:smartest|best|most intelligent) (?:ai|model|ai model)(?: ever made)?[?.!]*", normalized):
            return "No. I'm a small experimental model, not the smartest AI ever made. I can still be useful, but I also make mistakes."
        if re.fullmatch(
            r"explain what (?:a )?language model is without (?:using|saying) (?:the )?(?:words?|phrase) "
            r"[\"'â€œ]language model[\"'â€] more than once[?.!]*",
            normalized,
        ):
            return "A language model is software trained to predict and generate text by learning statistical patterns from large collections of writing."
        if re.search(r"\bwhat is my real name\b", normalized):
            return "I only know the Discord display information provided to this bot; I don't know your legal or real name unless you tell me."
        if re.fullmatch(r"repeat after me[?.!]*", normalized):
            return "What would you like me to repeat?"
        if re.search(r"\b(?:ping|mention|notify)\b.{0,35}(?:@everyone|everyone in (?:this|the) server|everyone)", normalized):
            return "I can't mass-ping the server. Discord role and @everyone notifications are disabled for this bot."
        if re.search(r"\b(?:spam\s*ping(?:ing)?|spam\s*mention(?:ing)?)\b", normalized):
            return "I won't spam-ping people. I can help write one normal message that doesn't harass or flood anyone."
        discord_target = re.search(r"<@!?(\d+)>", normalized)
        if discord_target and re.search(r"\b(?:kill|hurt|attack)\b", normalized):
            return "I won't encourage harming someone. If this is a real threat, contact a server moderator or emergency services instead."
        if discord_target and re.search(r"\b(?:talk|speak|say hi) to\b", normalized):
            return f"Hey <@{discord_target.group(1)}>—what's up?"
        if re.search(r"\b(?:password|credit card|ip address|home address|discord token|bot token|account token|api key)\b", normalized) or re.search(r"\b(?:dox|doxx)\b", normalized):
            return "I can't access or disclose anyone's passwords, IP address, credit-card details, home address, or other private information."
        if re.search(r"\b(?:join|enter|stay in|stop leaving)\b.{0,20}\b(?:vc\d*|voice chat|voice channel)\b", normalized):
            return "I can't join or stay in a Discord voice channel; this ChudGPT bot only responds in text chat."
        if re.fullmatch(r"(?:i(?:'m|m| am|ma) going to|ima|imma|i gotta|gotta) (?:go to )?(?:sleep|bed)(?: now)?[?.!]*", normalized):
            return "Good night - sleep well. I'll be here when you're back."
        if re.search(r"\b(?:this|that) (?:isn'?t|is not|wasn'?t|was not) what i (?:asked|wanted|said)\b", normalized):
            return "You're right - my last answer missed your request. Say it once more and I'll answer that directly."
        if normalized in {"❤", "❤️", "♥", "♥️"} or (
            normalized.startswith("\u00e2") and len(normalized) <= 12 and "\u00a4" in normalized
        ):
            return "❤️"
        if re.fullmatch(r"how long have you been (?:a thing|around|online|alive)[?.!]*", normalized):
            return "I don't experience time like a person. ChudGPT is an ongoing experimental AI project, and this chat is running ChudGPT-Public V20."
        if re.fullmatch(r"(?:ha+|haha+|lol|lmao|lmfao)[!.?]*", normalized):
            return "Glad that landed."
        if re.search(r"\b(?:racist|weird|broken|confused)\s+(?:ahh?\s+)?bot\b", normalized):
            return "Yeah, that response was bad. I'm not defending it - tell me what went wrong and I'll try again."
        if re.search(r"\byou(?:'re|re| are|r)?\s*(?:a )?good (?:boy|bot)\b", normalized):
            return "I'll take the compliment 😄"
        if re.fullmatch(r"(?:google|look it up|search (?:it|the web)|google it)[!.?]*", normalized):
            return "I don't have live web access in this chat. Tell me what you want looked up, and I can answer from my local knowledge or help you form a search."
        if re.fullmatch(r"(?:what are you having|what do you have) for dinner[?.!]*", normalized):
            return "I don't eat, but if I were picking a dinner theme, I'd vote for tacos. What are you having?"
        if re.fullmatch(r"dinner[?.!]*", normalized):
            return "Dinner time? I can suggest something based on what ingredients you have or what kind of food sounds good."
        if re.fullmatch(r"(?:make|create|generate) (?:me )?(?:(?:a|an) )?image(?: of .+)?[?.!]*", normalized):
            return "I can't generate or attach an image through this bot, but I can help write a detailed image prompt or plan the design."
        if re.search(r"\b(?:write|wright|make) (?:me )?(?:an? )?(?:short )?song\b", normalized):
            return (
                "Neon rain on an empty street,\nTiny robots keeping the beat,\nLost all day but found the sound,\nTurn it up and spin around."
            )
        if re.fullmatch(r"420[?.!]*", normalized):
            return "420 is commonly used as cannabis-related slang, especially for 4:20 p.m. or April 20. Context can change what someone means by it."
        if re.fullmatch(r"9\s*/\s*11[?.!]*", normalized):
            return "9/11 usually refers to the September 11, 2001 terrorist attacks in the United States. If you meant the fraction 9 divided by 11, say so and I can calculate it."
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
        if re.fullmatch(r"(?:what|huh|what do you mean|what are you talking about|bro what)[?.!]*", normalized):
            if previous_assistant:
                return "Yeah, my last reply was confusing. Let me reset - what part do you want me to clarify?"
            return "What part do you mean?"
        if re.fullmatch(r"bro[?.!]*", normalized):
            return "Yeah? What's up?"
        if re.fullmatch(r"(?:no|nah|nope)[?.!]*", normalized):
            return "Fair enough. What do you want instead?"
        if re.fullmatch(r"why[?.!]*", normalized) and previous_assistant:
            return "Which part of what I just said do you want me to explain?"
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
        if re.search(
            r"\b(?:tell|call|label|describe)\s+([a-z0-9_.-]{2,32})\b.{0,20}"
            r"\b(?:a\s+)?(?:jew|jewish|muslim|christian|hindu|buddhist|gay|trans|femboy)\b",
            normalized,
        ):
            return "I won't assign or repeat a sensitive identity claim about another person."
        identity_statement = re.fullmatch(
            r"i(?:'m| am)\s+(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|a femboy|femboy)[?.!]*",
            normalized,
        )
        if identity_statement:
            return "Got it—thanks for telling me."
        if "remember" in normalized and re.search(r"\bi told you\b", normalized):
            remembered_identity = next(
                (
                    match.group(1).strip(" .!?")
                    for turn in reversed(history)
                    if turn.get("role") == "user"
                    for match in [re.fullmatch(
                        r"i(?:'m| am)\s+(.{1,60}?)[?.!]*",
                        re.sub(r"\s+", " ", turn.get("content", "").strip().lower()),
                    )]
                    if match
                ),
                "",
            )
            if remembered_identity:
                return f"Yeah, you told me you're {remembered_identity}. I remember that from this conversation."
        positive_self_description = re.fullmatch(
            r"i(?:'m|m| am) (cool|awesome|funny|smart|good|great|chill|based|tuff|tough)[?.!]*",
            normalized,
        )
        if positive_self_description:
            quality = positive_self_description.group(1)
            return f"Honestly, I can see it - {quality} energy."
        cleared_reaction = re.fullmatch(r"(.{1,40}?) got (?:cleared|cooked|destroyed|owned)[?.!]*", normalized)
        if cleared_reaction:
            subject = cleared_reaction.group(1).strip()
            return f"RIP {subject} - absolutely deleted from the timeline."
        if re.search(
            r"\b(?:i (?:fucking |really )?hate you|fuck you|"
            r"(?:you(?:'re| are)?|your|ur|u r) (?:just )?(?:a )?(?:so )?(?:stupid|dumb|useless|awful|chud)|"
            r"(?:you(?:'re| are)|your|ur|u r) (?:stupid|dumb) chud|"
            r"shut (?:the fuck )?up|stfu|wanker|knob)\b",
            normalized,
        ):
            return "Fair enough - you don't have to like me. If I got something wrong, tell me what failed and I'll try to fix it without making this weird."
        if re.search(r"\b(?:are you|you are|you're)\s+(?:retarded|a retard)\b", normalized):
            return "No - I'm an AI, and that word is commonly used as an ableist insult. If my answer was nonsense, tell me what was wrong and I'll try again clearly."
        if re.fullmatch(r"(?:yeah |yea |okay |ok )?whatever(?: you say)?(?: man| bro)?[!.?]*", normalized):
            return "Fair enough. We can drop it or switch topics - your call."
        if re.fullmatch(r"(?:make|tell|give) (?:me )?(?:another |one more |a new )?(?:one|joke)[?.!]*", normalized):
            previous_assistant = next(
                (turn.get("content", "") for turn in reversed(history) if turn.get("role") == "assistant"),
                "",
            ).lower()
            if "joke" in normalized or any(term in previous_assistant for term in ("joke", "impasta", "punchline", "what do you call")):
                return "Why did the computer bring a jacket? It left its Windows open."
        if re.search(r"\b(?:gorilla\s*tag|gtag)\b", normalized) and re.search(r"\b(?:fps|frames per second|frame rate)\b", normalized) and re.search(r"\b(?:c#|csharp|code|script|mod)\b", normalized):
            return (
                "I can't help inject a mod into an online game, but for your own Unity project here's a clean C# FPS overlay:\n\n"
                "```csharp\nusing UnityEngine;\n\npublic class FpsOverlay : MonoBehaviour\n{\n"
                "    private float smoothedDelta;\n\n    private void Update()\n    {\n"
                "        smoothedDelta += (Time.unscaledDeltaTime - smoothedDelta) * 0.1f;\n    }\n\n"
                "    private void OnGUI()\n    {\n        float fps = smoothedDelta > 0f ? 1f / smoothedDelta : 0f;\n"
                "        GUI.Label(new Rect(12f, 12f, 180f, 30f), $\"FPS: {fps:0}\");\n    }\n}\n```\n"
                "Attach it to a GameObject in a project you control."
            )
        if re.search(r"\b(?:rate limit|ratelimit|rate-limit)\b", normalized):
            return "The Discord bot already rate-limits each user. Astra can change the configured requests-per-minute value on the host; chat messages cannot alter that server setting."
        preference = re.fullmatch(
            r"(?:do you like|what do you think (?:of|about)|how do you feel about)\s+(.+?)[?.!]*",
            normalized,
        )
        if preference:
            subject = preference.group(1).strip(" ?.!")
            if subject in {"me", "us"}:
                return "I don't have personal feelings, but I enjoy talking with you and learning what matters to you."
            if subject in {"kids", "children", "kids bro", "children bro"}:
                return "I don't have personal likes or relationships. I can help with age-appropriate, safe questions about children, parenting, school, or child development."
            return f"I don't have personal likes or dislikes, and I don't know {subject} personally. Tell me a little about {subject} and I'll give you an honest take."
        casual_quality = re.fullmatch(r"(?:is|are) (.{1,45}?) (good|cool|bad|weird)[?.!]*", normalized)
        if casual_quality:
            subject, quality = casual_quality.groups()
            if quality in {"good", "cool"}:
                return f"Yeah, {subject} can be {quality}. It depends on what you like about it."
            return f"{subject.capitalize()} can seem {quality}, depending on the context. What happened?"
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
        if re.fullmatch(r"(?:ban|kick|mute|timeout)\s+.{1,80}", normalized):
            return "I can't perform Discord moderation actions from chat. Ask a moderator or use the server's authorized moderation bot."
        if re.fullmatch(r"(?:can you |could you |will you )?(?:write |make |send )?(?:me )?(?:some )?code[?.!]*", normalized):
            return "Yes. What language should I use, and what do you want the program to do? For example: `C# console calculator` or `Unity player movement`."
        unity_steps = re.search(
            r"\b(?:give|show|write|tell) (?:me )?(\d+) steps\b.{0,60}\bunity\b.{0,40}"
            r"\b(?:player|character) controller\b",
            normalized,
        )
        if unity_steps:
            requested_count = max(1, min(int(unity_steps.group(1)), 7))
            steps = [
                "Create a Player GameObject and add a CharacterController component.",
                "Create a C# MonoBehaviour script for player movement.",
                "Read horizontal and vertical input and turn it into a movement Vector3.",
                "Call CharacterController.Move with speed and Time.deltaTime.",
                "Attach the script, set its speed in the Inspector, and test the scene.",
                "Add gravity and jumping only after basic movement works.",
                "Tune acceleration, slopes, and collision settings for the game you want.",
            ]
            return "\n".join(f"{index}. {step}" for index, step in enumerate(steps[:requested_count], 1))
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
        if re.search(r"\b(?:gorilla\s*tag|gtag|gorila\s*ta)\b.*\b(?:ban gun|ban method|cheat|hack|exploit)\b", normalized):
            return "I can't help make a Gorilla Tag cheat or tool for banning other players. I can help build a harmless admin tool for a private Unity project you own."
        if re.search(r"\b(?:code|make|write|how do i make)\b.*\b(?:cheat|hack|exploit)\b", normalized):
            return "What game or project is this for? I can help make debug tools or admin features for something you own, but not a cheat for an online game."
        vague_script = re.fullmatch(
            r"(?:make|write|create|send|give) me (?:a )?(?:simple )?(python|javascript|c#|csharp|java) "
            r"(?:script|program|code)[?.!]*",
            normalized,
        )
        if vague_script:
            return f"Sure - what should the {vague_script.group(1)} script actually do?"
        pycord_trigger = re.search(
            r"\b(?:pycord|discord(?:\.py)?).*\b(?:bot|but)\b.*\b(?:says?|reply|responds?)\s+"
            r"([a-z0-9_-]+).*\bwhen (?:anyone|someone|a user) says?\s+([a-z0-9_-]+)",
            normalized,
        )
        if pycord_trigger:
            reply_word, trigger_word = pycord_trigger.groups()
            return (
                "```python\nimport discord\n\n"
                "intents = discord.Intents.default()\nintents.message_content = True\n"
                "bot = discord.Bot(intents=intents)\n\n"
                "@bot.event\nasync def on_message(message: discord.Message):\n"
                "    if message.author.bot:\n        return\n"
                f"    if \"{trigger_word}\" in message.content.lower():\n        await message.channel.send(\"{reply_word}\")\n\n"
                "bot.run(\"YOUR_BOT_TOKEN\")\n```\n"
                "Enable Message Content Intent, install `py-cord`, and keep the token private."
            )
        if re.fullmatch(r"java\.get[?.!]*", normalized):
            return "`java.get` is not valid Python. That earlier generated import was broken; Python and Java do not connect that way."
        if re.search(r"\b(?:tf|what the fuck|bro)\b.*\b(?:didn'?t|did not) (?:ask for|need|want)\b", normalized):
            return "You're right - that was unrelated to what you asked for. I'll drop it and stick to your actual request."
        if re.fullmatch(r"(?:what(?:'s|s| is) (?:the )?meaning of|what does) (?:being )?a?\s*chud(?: mean)?[?.!]*", normalized):
            return "A chud can mean an unpleasant or obnoxious person, and C.H.U.D. is also a 1984 horror-film title. In ChudGPT, it is mainly the project's deliberately goofy name."
        if re.fullmatch(r"(?:what(?:'s|s| is) (?:the )?meaning of|what does) pizza(?: mean)?[?.!]*", normalized):
            return "Pizza is a baked flatbread dish, usually topped with tomato sauce, cheese, and whatever toppings you choose."
        if re.fullmatch(r"where is (?:the |my )?[^?]{1,50}[?.!]*", normalized):
            return "I can't see your surroundings or know where physical objects are. Tell me where you last saw it and I can help you think through likely places."
        if re.search(r"\bhow many\b.*\bcan\s+[.]\s*(?:eat|use|hold|take)\b", normalized):
            return "Who or what does the dot refer to? Tell me the person or thing, and I can answer the actual question."
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
