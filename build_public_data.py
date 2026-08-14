"""Build the diverse, project-authored ChudGPT-Public conversation corpus.

The previous 20k-row corpus contained only 4,228 unique conversations and was
dominated by a handful of repeated replies.  This builder creates 30k unique
conversations with balanced response types and validates the result before it
can replace the tracked dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "public_conversations.jsonl"
RAW_OUTPUT = ROOT / "data" / "raw" / "public_corpus.jsonl"
SEED = 228
TARGET_CONVERSATIONS = 21_900

FACTS = [
    ("the sky appears blue", "air molecules scatter shorter blue wavelengths of sunlight more strongly than longer red wavelengths"),
    ("the Moon has phases", "we see different portions of its sunlit half as it orbits Earth"),
    ("Earth has seasons", "Earth's tilted axis changes sunlight angle and day length through the year"),
    ("rain forms", "water vapor cools into droplets that combine and eventually become heavy enough to fall"),
    ("ocean water is salty", "weathering and geological activity carry dissolved minerals into the oceans"),
    ("ice floats", "water expands into a crystal structure that is less dense than liquid water"),
    ("leaves often look green", "chlorophyll absorbs much red and blue light while reflecting more green light"),
    ("metal rusts", "iron reacts with oxygen and water to form iron oxides"),
    ("sound cannot travel through a vacuum", "sound needs matter whose particles can pass along vibrations"),
    ("exercise raises heart rate", "active muscles need blood to deliver more oxygen and remove waste"),
    ("sleep supports learning", "sleep helps the brain consolidate memories and restore attention"),
    ("stars twinkle", "moving layers of Earth's atmosphere bend their light by changing amounts"),
    ("the ocean has tides", "the Moon's gravity, with a smaller contribution from the Sun, pulls on Earth's oceans"),
    ("soap removes grease", "soap molecules attach to both water and oils so water can carry the oils away"),
    ("bread dough rises", "yeast releases carbon dioxide that becomes trapped in the dough"),
    ("a rainbow forms", "water droplets refract, reflect, and separate sunlight into different colors"),
    ("a compass points north", "its magnetized needle aligns with Earth's magnetic field"),
    ("a thermos slows cooling", "insulation reduces heat transfer by conduction, convection, and radiation"),
    ("plants bend toward light", "growth hormones redistribute and cause cells on the darker side to lengthen"),
    ("objects fall near Earth", "Earth's gravity accelerates them toward its center"),
]

KNOWLEDGE = [
    ("capital of France", "Paris"), ("capital of Japan", "Tokyo"),
    ("capital of Canada", "Ottawa"), ("capital of Australia", "Canberra"),
    ("capital of Italy", "Rome"), ("capital of Spain", "Madrid"),
    ("capital of Brazil", "Brasília"), ("capital of Texas", "Austin"),
    ("largest ocean on Earth", "the Pacific Ocean"),
    ("largest planet in the Solar System", "Jupiter"),
    ("planet closest to the Sun", "Mercury"),
    ("natural satellite of Earth", "the Moon"),
    ("author of Romeo and Juliet", "William Shakespeare"),
    ("painter of the Mona Lisa", "Leonardo da Vinci"),
    ("chemical symbol for gold", "Au"), ("chemical symbol for oxygen", "O"),
    ("number of sides on a hexagon", "six"),
    ("number of continents commonly recognized", "seven"),
    ("freezing point of pure water in Celsius at standard pressure", "0 degrees Celsius"),
    ("boiling point of pure water in Celsius at standard pressure", "100 degrees Celsius"),
]

TOPICS = [
    "music", "movies", "games", "books", "space", "animals", "food", "history",
    "art", "sports", "school", "travel", "friendship", "nature", "photography",
    "cars", "robots", "architecture", "gardening", "cooking", "comedy", "weather",
]

NAMES = ["Avery", "Jordan", "Sam", "Maya", "Leo", "Nora", "Kai", "Riley", "Quinn", "Zoe"]
COLORS = ["teal", "purple", "orange", "green", "blue", "red", "silver", "yellow", "indigo", "coral"]
ANIMALS = ["otter", "fox", "crow", "dolphin", "owl", "capybara", "turtle", "cat", "dog", "octopus"]
MEMES = [
    # Representative durable and high-visibility formats from 2016-2026.
    ("Harambe", "a 2016 wave of tributes and absurd jokes after the Cincinnati Zoo gorilla was killed"),
    ("Dat Boi", "the unicycling frog greeted with 'here come dat boi' and 'o snap'"),
    ("Arthur's Fist", "a clenched-fist reaction image showing restrained anger or frustration"),
    ("Evil Kermit", "Kermit facing a hooded double who voices a person's tempting bad idea"),
    ("Damn Daniel", "a repeated video compliment about Daniel's outfit and white Vans"),
    ("Bottle Flip", "the challenge of tossing a partly filled bottle so it lands upright"),
    ("Salt Bae", "the theatrical wrist-bent salt sprinkle used for exaggerated flair"),
    ("Mocking SpongeBob", "alternating-capital text paired with SpongeBob to imitate or mock a statement"),
    ("Blinking White Guy", "Drew Scanlon blinking in disbelief as a reaction to something confusing"),
    ("Roll Safe", "a man tapping his temple to present intentionally flawed clever logic"),
    ("Doge", "a Shiba Inu image paired with intentionally broken phrases such as 'much wow' and 'very skill'"),
    ("This Is Fine", "a cartoon dog calmly sitting in a burning room, used when someone pretends a bad situation is manageable"),
    ("Distracted Boyfriend", "a stock-photo reaction format about abandoning one interest for a tempting new one"),
    ("Surprised Pikachu", "an exaggerated shocked reaction to an outcome that was predictable"),
    ("Drake Hotline Bling", "a two-panel format that rejects one option and approves another"),
    ("Galaxy Brain", "a sequence of increasingly dramatic ideas, often making fun of supposedly smarter choices"),
    ("Woman Yelling at a Cat", "two reaction images combined to represent an emotional argument and a confused response"),
    ("Rickroll", "a bait-and-switch link that unexpectedly leads to Rick Astley's 'Never Gonna Give You Up' video"),
    ("Loss", "a minimalist four-panel pattern referencing a particular Ctrl+Alt+Del comic, often hidden in other images"),
    ("One Does Not Simply", "a Boromir image used to say that a task is much harder than it sounds"),
    ("Change My Mind", "a table-sign format used to present an opinion as a challenge for debate"),
    ("Gru's Plan", "a four-panel presentation in which the speaker notices a flaw in their own plan"),
    ("Moth and Lamp", "a moth obsessively seeking a lamp, used for irresistible attraction"),
    ("Ugandan Knuckles", "a VRChat character and repeated 'do you know da wae' catchphrase that became notorious in 2018"),
    ("Big Chungus", "an exaggerated image of Bugs Bunny used as an absurd character and reaction"),
    ("Stonks", "a deliberately misspelled finance reaction with Meme Man, often mocking questionable money decisions"),
    ("Me and the Boys", "a group of retro Spider-Man villains representing friends preparing for mischief"),
    ("Area 51 Raid", "jokes about a fictional mass raid to see supposed aliens at Area 51"),
    ("OK Boomer", "a dismissive reply to attitudes associated with an older generation"),
    ("Coffin Dance", "pallbearers dancing with a coffin, edited after a mistake to signal an impending comic disaster"),
    ("How It Started vs. How It's Going", "a side-by-side comparison between an origin and a later outcome"),
    ("Always Has Been", "two astronauts revealing that a surprising fact was true all along"),
    ("Among Us Sus", "jokes using 'sus' for suspicious behavior, inspired by the social-deduction game Among Us"),
    ("Bonk, Go to Horny Jail", "a Doge hitting another Doge with a bat as a joking reprimand"),
    ("Bernie Sanders Mittens", "the bundled-up Bernie inauguration photo inserted into many unrelated scenes"),
    ("Trade Offer", "a two-sided proposal format stating what 'I receive' and 'you receive'"),
    ("Amogus", "an intentionally distorted way of saying Among Us, often paired with vaguely crewmate-shaped objects"),
    ("Squid Game Red Light Green Light", "references to the show's giant doll and deadly version of the playground game"),
    ("It's Corn", "an enthusiastic child's description of corn remixed into a viral song"),
    ("Morbin Time", "an ironic catchphrase falsely attributed to the movie Morbius"),
    ("Ohio", "absurd jokes portraying Ohio as a supernatural or impossible place"),
    ("Gigachad", "a highly stylized muscular man used for exaggerated confidence or approval"),
    ("Skibidi Toilet", "an animated web-series and meme about singing heads in toilets fighting camera-headed characters"),
    ("Girl Dinner", "a humorous label for an improvised meal made from assorted snacks or small items"),
    ("My Roman Empire", "a phrase for a subject someone thinks about surprisingly often"),
    ("Grimace Shake", "videos pretending the purple fast-food drink causes bizarre horror-movie consequences"),
    ("Canon Event", "a Spider-Verse-inspired phrase for an important experience supposedly necessary to a person's story"),
    ("Chill Guy", "an original cartoon dog character whose calm pose represents being relaxed despite the situation"),
    ("Just Put My Fries in the Bag Bro", "a dismissive phrase telling someone to stop overexplaining and finish a simple job"),
    ("English or Spanish", "a 2024 video trend built around a freeze challenge after asking which language someone speaks"),
    ("Aura", "slang and jokes about gaining or losing imaginary coolness points"),
    ("Queen Never Cry", "a redraw and reaction format centered on an exaggeratedly stoic baby from a webcomic"),
    ("Italian Brainrot", "surreal AI-generated creatures with pseudo-Italian names and intentionally nonsensical narration"),
    ("Tralalero Tralala", "a shark wearing sneakers associated with the 2025 Italian-brainrot wave"),
    ("Still Water", "a brainrot-style phrase used in exaggerated danger edits about supposedly hazardous motionless water"),
    ("Low Taper Fade", "a phrase and remix format kept alive through repeated jokes about the meme's longevity"),
    ("SDIYBT", "the shortened form of 'sometimes I dream about you, baby,' spread through a robotic SpongeBob clip"),
    ("Great Meme Reset of 2026", "an internet campaign proposing a return from modern brainrot to older 2010s meme styles"),
    ("2026 Is the New 2016", "a nostalgia trend comparing 2026 culture with the music, fashion, filters, and internet mood of 2016"),
    ("Nihilistic Penguin", "a wandering penguin clip used for jokes about leaving the crowd, rebellion, or existential choices"),
    ("POV", "a caption framing the joke from a particular point of view"),
    ("main character energy", "playful slang for behaving with unusual confidence or dramatic importance"),
    ("side quest", "slang for an unexpected optional activity that interrupts the main plan"),
    ("touch grass", "a joking suggestion that someone step away from the internet and reconnect with ordinary life"),
    ("let them cook", "slang for allowing someone to continue an unusual idea because it may turn out well"),
    ("cooked", "slang that can mean exhausted, doomed, defeated, or in serious trouble depending on context"),
    ("no thoughts, head empty", "a playful phrase for a blank, relaxed, or adorably clueless mood"),
    ("67", "a deliberately vague 2025 meme derived from the '6-7' lyric in Skrilla's song 'Doot Doot,' boosted by basketball edits and used mainly for random, confusing humor"),
    ("Trollface", "a rage-comic face associated with trolling, baiting, and smugly causing trouble online"),
    ("Forever Alone", "a rage-comic character used to exaggerate loneliness or social rejection"),
    ("Success Kid", "a toddler making a fist, captioned as a small but satisfying victory"),
    ("Bad Luck Brian", "a school portrait captioned with setups that end in comically terrible luck"),
    ("Overly Attached Girlfriend", "an intense reaction portrait used to parody possessive behavior"),
    ("Grumpy Cat", "a naturally frowning cat used for exaggerated annoyance and dislike"),
    ("Pepe the Frog", "a comic character used across many reaction emotions; its meaning depends heavily on the specific image and community"),
    ("Wojak", "a simple line-drawn character whose many variants represent emotions, personalities, and online archetypes"),
    ("Virgin vs Chad", "a contrast format that exaggerates an insecure or overcomplicated approach against a confident one, often ironically"),
    ("NPC", "slang comparing repetitive or unthinking behavior to a non-player character; it can be joking but also dismissive"),
    ("Press F to Pay Respects", "a game prompt from Call of Duty turned into a concise online expression of sympathy or mock sympathy"),
    ("Leeroy Jenkins", "a World of Warcraft clip about charging in before the group is ready, used for reckless plans"),
    ("The Cake Is a Lie", "a Portal reference used when a promised reward seems fake or manipulative"),
    ("Arrow to the Knee", "a Skyrim guard line repeated as a joke about giving up an adventurous past"),
    ("Do a Barrel Roll", "a Star Fox command that became a gaming catchphrase and playful instruction"),
    ("Keyboard Cat", "a cat appearing to play a keyboard, traditionally used to play someone off after a failure"),
    ("Nyan Cat", "a pixel cat flying through space with a rainbow trail, associated with cheerful early-internet absurdity"),
    ("I Can Has Cheezburger", "a classic LOLcat caption style using intentionally broken grammar"),
    ("Hide the Pain Harold", "a stock-photo smile that looks strained, used for politely hiding discomfort"),
    ("Is This a Pigeon", "an anime screenshot used when someone confidently misidentifies a thing or situation"),
    ("Two Buttons", "a sweating character choosing between two buttons, used for a difficult or hypocritical choice"),
    ("UNO Draw 25", "a choice between doing an unwanted task and drawing 25 cards, used for stubborn refusal"),
    ("They Had Us in the First Half", "a reaction for a statement that begins one way and then unexpectedly reverses"),
    ("Spider-Man Pointing", "matching Spider-Men pointing at one another, used when two similar people or things meet"),
    ("Confused Math Lady", "a reaction surrounded by equations, used for intense confusion rather than an actual calculation"),
    ("Disaster Girl", "a girl smiling near a burning building, used to imply mischievous responsibility for chaos"),
    ("Galaxy Brain", "an escalating brain-image format that ironically ranks increasingly extreme ideas as smarter"),
    ("Nobody Asked", "a blunt reaction saying information was unsolicited; tone can range from teasing to rude"),
    ("Ratio", "a social-media reply predicting or pointing out that a response receives more approval than the original post"),
    ("Based", "slang expressing approval for a confident or unapologetic opinion, often used ironically"),
    ("Mid", "slang dismissing something as mediocre or unimpressive"),
    ("Peak", "strong praise meaning something is at or near the best of its kind, sometimes exaggerated for humor"),
    ("Real", "a brief way to strongly agree or say that a statement feels relatable; it is not automatically a meme"),
    ("Rizz", "slang for charm or skill at attracting someone, often measured or exaggerated as a joke"),
    ("Delulu", "playful slang for being delusional, often about unrealistic hopes or romantic beliefs"),
    ("It's Giving", "a phrase describing the impression, aesthetic, or energy something evokes"),
    ("Bro Is NOT Beating the Allegations", "a joke that someone's new behavior appears to confirm an accusation or running stereotype"),
    ("Average Discord Mod", "a stereotype joke portraying a Discord moderator as overly online, controlling, or socially awkward"),
    ("POV Short-Video Format", "a caption that assigns the viewer a role or situation before a short scene"),
    ("Duet and Stitch Reactions", "short-form-video formats that place a response beside or after another creator's clip"),
]


def conversation(*turns: tuple[str, str], source: str = "chudgpt-public-v3") -> dict[str, object]:
    messages: list[dict[str, str]] = []
    for user, assistant in turns:
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
    return {"messages": messages, "source": source}


def add_unique(records: list[dict[str, object]], seen: set[str], item: dict[str, object]) -> None:
    key = json.dumps(item["messages"], ensure_ascii=False, sort_keys=True)
    if key not in seen:
        seen.add(key)
        records.append(item)


def build(seed: int = SEED, target: int = TARGET_CONVERSATIONS) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    greetings = ["Hi", "Hello", "Hey there", "Good morning", "Good evening", "Hello mate", "Yo", "Hi ChudGPT"]
    greeting_replies = [
        "Hey! What would you like to talk about?", "Hello! What is on your mind?",
        "Hi there. I am ready to chat or help with a question.", "Hey! How is your day going?",
    ]
    capabilities = [
        "I can chat about everyday topics, explain basic facts, solve arithmetic and word problems, write simple code, and follow the current conversation. I am a small experimental model, so important answers should still be checked.",
        "You can ask me questions, talk through an idea, request a short story, try some math, or ask for simple code. I work best when the request is clear.",
        "I am useful for conversation, explanations, basic reasoning, math, and beginner code. I do not have live internet access, and I can make mistakes.",
    ]
    for i in range(1_000):
        greeting = f"{rng.choice(greetings)}{rng.choice(['!', '.', '', '!!'])}"
        add_unique(rows, seen, conversation((greeting, rng.choice(greeting_replies))))
        capability_prompt = rng.choice(["What can you do?", "How can you help me?", "What can we talk about?", "Tell me your abilities."])
        add_unique(rows, seen, conversation((capability_prompt, rng.choice(capabilities))))

    identity_prompts = ["Who are you?", "What are you?", "What is your name?", "Tell me about yourself.", "Which ChudGPT model are you?"]
    identity_answers = [
        "I am ChudGPT Public, a small locally runnable experimental conversational language model.",
        "My name is ChudGPT Public. I generate replies from learned language patterns and the current conversation.",
        "I am the Public ChudGPT model: a 20,999,184-parameter decoder-only transformer with a 1,024-token model context.",
    ]
    for i in range(1_000):
        prompt = f"{rng.choice(identity_prompts)}{rng.choice(['', ' Be concise.', ' Give me the honest version.'])}"
        add_unique(rows, seen, conversation((prompt, rng.choice(identity_answers))))

    # Exact arithmetic and word problems use broad, held-out number ranges.
    arithmetic_forms = ["What is {a} {op} {b}?", "Calculate {a} {op} {b}.", "Solve {a} {op} {b}.", "Give the result of {a} {op} {b}."]
    # Exact helpers guarantee arithmetic at runtime, so thousands of nearly
    # identical equations waste the model's limited learning budget.  Keep a
    # varied foundation without allowing it to dominate ordinary language.
    for i in range(4_000):
        op = rng.choice(["+", "-", "×"])
        a = rng.randint(-9_999, 99_999)
        b = rng.randint(-999, 9_999)
        value = a + b if op == "+" else a - b if op == "-" else a * b
        prompt = rng.choice(arithmetic_forms).format(a=a, op=op, b=b)
        answer = str(value) if i % 5 == 0 else f"{a} {op} {b} = {value}."
        add_unique(rows, seen, conversation((prompt, answer)))
    for i in range(1_200):
        speed = rng.randint(12, 95)
        hours = rng.choice([0.5, 1.5, 2.5, 3, 4.5, 6])
        distance = speed * hours
        person = rng.choice(NAMES)
        prompt = rng.choice([
            f"A train travels {speed} miles per hour for {hours:g} hours. How far does it go?",
            f"{person} drives at {speed} mph for {hours:g} hours. What distance is traveled?",
            f"At a constant {speed} mph, how many miles are covered in {hours:g} hours?",
        ])
        shown = int(distance) if float(distance).is_integer() else distance
        add_unique(rows, seen, conversation((prompt, f"Distance equals speed times time: {speed} × {hours:g} = {shown} miles.")))

    fact_forms = ["Why does {subject}?", "Explain why {subject}.", "How come {subject}?", "In simple terms, why does {subject}?"]
    for i in range(3_500):
        subject, reason = rng.choice(FACTS)
        prompt = rng.choice(fact_forms).format(subject=subject)
        suffix = rng.choice(["", " Keep it brief.", " Use one clear sentence.", " I am curious."])
        answer = rng.choice([
            f"{subject.capitalize()} because {reason}.",
            f"The main reason is that {reason}.",
            f"In short, {reason}.",
        ])
        add_unique(rows, seen, conversation((f"{prompt}{suffix}".strip(), answer)))

    for i in range(2_000):
        subject, answer = rng.choice(KNOWLEDGE)
        prompt = rng.choice([f"What is the {subject}?", f"Tell me the {subject}.", f"Quick question: what is the {subject}?", f"Give a short answer: the {subject}?"])
        add_unique(rows, seen, conversation((prompt, f"The {subject} is {answer}.")))

    commonsense = [
        ("Which is heavier: one kilogram of steel or one kilogram of feathers?", "Neither is heavier. Both have a mass of one kilogram, although the feathers occupy more space."),
        ("If I drop a glass cup on concrete, what is likely to happen?", "It is likely to crack or shatter because glass is brittle and concrete is hard."),
        ("Why should I bring an umbrella when rain is forecast?", "An umbrella can keep you drier if it rains."),
        ("Can I read a closed book without opening it?", "Not normally; you need access to the pages or another copy of the text."),
        ("If all roses are flowers, is every flower a rose?", "No. Roses are one kind of flower, but many flowers are not roses."),
        ("A box is too large for a shelf. Should I force it in?", "No. Measure both first and use a larger shelf or a smaller box to avoid damage."),
        ("If the power is out, will an unplugged lamp turn on?", "No. It needs electrical power unless it has a charged battery or another power source."),
        ("Why does wet pavement look darker?", "Water changes how light reflects and lets more light enter the surface instead of scattering back."),
    ]
    for i in range(1_800):
        prompt, answer = rng.choice(commonsense)
        add_unique(rows, seen, conversation((prompt, answer)))

    # Everyday topic changes and brief replies teach response type and grounding.
    for i in range(3_000):
        topic = rng.choice(TOPICS)
        opener = rng.choice([f"Let's talk about {topic}.", f"I have been thinking about {topic}.", f"What is interesting about {topic}?", f"Switch topics: {topic}."])
        response = rng.choice([
            f"Sure—we can talk about {topic}. What part of it interests you most?",
            f"{topic.capitalize()} gives us plenty to discuss. Do you want ideas, history, opinions, or a relaxed conversation?",
            f"I am with you on the new topic: {topic}. Tell me what brought it to mind.",
        ])
        follow = rng.choice(["The creative side.", "Something surprising.", "Just casually.", "Tell me one thing you know."])
        follow_answer = f"One useful way into {topic} is to pick a specific example and notice what makes it memorable, useful, or enjoyable. We can build from there."
        add_unique(rows, seen, conversation((opener, response), (follow, follow_answer)))

    for i in range(2_000):
        color, animal, name = rng.choice(COLORS), rng.choice(ANIMALS), rng.choice(NAMES)
        kind = i % 3
        if kind == 0:
            turns = ((f"Remember that my favorite color is {color}.", f"Got it—your favorite color is {color} in this conversation."), (rng.choice(["What color did I tell you?", "What is my favorite color?", "Which color was it?"]), f"You told me your favorite color is {color}."))
        elif kind == 1:
            turns = (("Pick an animal.", f"I pick the {animal}."), (rng.choice(["Why?", "What did you pick?", "Which animal did you choose?"]), f"I picked the {animal}."))
        else:
            turns = ((f"My friend {name} likes {animal}s.", f"Got it—{name} likes {animal}s."), ("Who likes them?", f"Your friend {name} likes them."))
        add_unique(rows, seen, conversation(*turns))

    code_templates = [
        ("Python", "add two numbers", "def add(a: float, b: float) -> float:\n    return a + b"),
        ("Python", "test whether a number is even", "def is_even(number: int) -> bool:\n    return number % 2 == 0"),
        ("JavaScript", "reverse a string", "function reverseText(text) {\n  return [...text].reverse().join('');\n}"),
        ("C#", "roll a six-sided die", "using System;\n\nint roll = Random.Shared.Next(1, 7);\nConsole.WriteLine(roll);"),
        ("C#", "clamp a score to zero or higher", "int ClampScore(int score)\n{\n    return Math.Max(0, score);\n}"),
        ("Unity C#", "rotate a GameObject", "using UnityEngine;\n\npublic class Rotator : MonoBehaviour\n{\n    [SerializeField] float speed = 90f;\n    void Update() => transform.Rotate(0f, speed * Time.deltaTime, 0f);\n}"),
    ]
    for language, task, code in code_templates:
        tag = "csharp" if "C#" in language else language.lower()
        add_unique(rows, seen, conversation(("Code me some code.", f"Here is a randomly chosen {language} example that can {task}:\n```{tag}\n{code}\n```")))
    for i in range(2_500):
        language, task, code = rng.choice(code_templates)
        prompt = rng.choice([f"Write {language} code to {task}.", f"Can you make a {language} example that will {task}?", f"Code only: use {language} to {task}."])
        tag = "csharp" if "C#" in language else language.lower()
        answer = f"```{tag}\n{code}\n```"
        add_unique(rows, seen, conversation((prompt, answer)))

    # Recovery, quoted speech, identity comparisons, and concise constraints.
    for i in range(2_500):
        mode = i % 5
        if mode == 0:
            turns = (("I said, 'the project is blue.'", "Understood—you said that the project is blue."), ("What did I say?", "You said that the project is blue."))
        elif mode == 1:
            turns = (("You said the Moon is a planet.", "If I said that, it was incorrect. The Moon is Earth's natural satellite."), ("Who made the mistake?", "I did, according to your correction."))
        elif mode == 2:
            turns = (("Are you better than ChudGPT Pro?", "Public is designed as the stronger general model for short instructions, everyday conversation, and exact helpers. Pro has a longer runtime context, so which performs better can depend on the task and current benchmark."),)
        elif mode == 3:
            turns = (("Answer in exactly one sentence: why do people use maps?", "People use maps to understand locations, routes, distances, and spatial relationships."),)
        else:
            turns = (("That is not what I meant.", "Thanks for correcting me. Tell me the intended topic or result, and I will take another approach."), ("I meant music, not movies.", "Understood—we are talking about music now, not movies."))
        add_unique(rows, seen, conversation(*turns))

    # More numerically varied reasoning expands the corpus without repeating a
    # response template against unrelated language prompts.
    for i in range(7_000):
        divisor = rng.randint(2, 250)
        quotient = rng.randint(-500, 2_000)
        dividend = divisor * quotient
        prompt = rng.choice([
            f"What is {dividend} divided by {divisor}?",
            f"Calculate {dividend} ÷ {divisor}.",
            f"If {dividend} items are split equally into {divisor} groups, how many are in each group?",
        ])
        answer = f"{dividend} ÷ {divisor} = {quotient}."
        add_unique(rows, seen, conversation((prompt, answer)))

    for i in range(4_000):
        percent = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
        base = rng.randint(2, 2_000) * 20
        value = base * percent // 100
        prompt = rng.choice([
            f"What is {percent}% of {base}?",
            f"Find {percent} percent of {base}.",
            f"A {base}-point total is reduced to {percent} percent. What is the new value?",
        ])
        add_unique(rows, seen, conversation((prompt, f"{percent}% of {base} is {value}.")))

    situations = [
        ("I missed the bus", "That is frustrating. Check the next route or a safe alternative, and let anyone waiting for you know about the delay."),
        ("I finished my homework", "Nice—getting it finished can be a relief. Do you want to relax or do something fun now?"),
        ("I am bored", "We could trade a random fact, invent a tiny story, plan a simple activity, or just talk about whatever comes to mind."),
        ("My day was rough", "I am sorry it was rough. You can tell me what happened, or we can keep things light if you need a break."),
        ("I made dinner", "Nice. What did you make, and did it turn out the way you hoped?"),
        ("I cannot decide what game to play", "Pick based on your mood: something familiar for comfort, competitive for energy, or story-driven when you want to settle in."),
        ("I learned something new", "That is a good feeling. What did you learn?"),
        ("I need a quiet evening", "A low-pressure plan could be a shower, comfortable food, gentle music, and one activity that does not demand much attention."),
    ]
    for i in range(3_000):
        prompt, answer = rng.choice(situations)
        context = rng.choice(["today", "after work", "after school", "this afternoon", "tonight", "just now"])
        add_unique(rows, seen, conversation((f"{prompt} {context}.", answer)))

    items = ["notebooks", "apples", "tickets", "paintbrushes", "sandwiches", "game cards", "flower pots", "batteries"]
    for _ in range(1_500):
        item = rng.choice(items)
        count = rng.randint(2, 80)
        price = rng.randint(2, 75)
        total = count * price
        prompt = rng.choice([
            f"Each of {count} {item} costs ${price}. What is the total cost?",
            f"I buy {count} {item} for ${price} each. How much do I pay before tax?",
            f"A shop sells {item} at ${price} each. What do {count} cost?",
        ])
        add_unique(rows, seen, conversation((prompt, f"Multiply quantity by price: {count} × ${price} = ${total}.")))

    meme_prompts = [
        "What does the {name} meme mean?", "Explain {name}.", "Do you know the {name} meme?",
        "Give me the short version of {name}.", "How is {name} usually used online?",
    ]
    for name, meaning in MEMES:
        for prompt_form in meme_prompts:
            for tone in ("", " Keep it brief.", " Explain it casually.", " No giant list."):
                prompt = prompt_form.format(name=name) + tone
                answer = f"{name} is {meaning}. The exact joke still depends on the caption and conversation around it."
                add_unique(rows, seen, conversation((prompt, answer)))
    meme_requests = ["Make a harmless meme caption about homework.", "Give me a meme-style line about a slow computer.", "Make a quick gaming meme.", "Reply with light meme energy: I forgot my password again."]
    meme_replies = [
        "Homework at 3 PM: one page. Homework at 11:59 PM: final boss music begins.",
        "My computer opened one new tab and began negotiating its retirement package.",
        "Me: one peaceful match. The matchmaking system: welcome to the championship finals.",
        "Password memory: no thoughts, head empty. Reset-email button: our greatest hero returns.",
    ]
    for prompt, answer in zip(meme_requests, meme_replies):
        add_unique(rows, seen, conversation((prompt, answer)))

    # Dense, contextual examples teach slang interpretation, negation, and
    # short follow-ups together instead of spending separate rows on filler.
    contextual_meme_dialogues = [
        (("I forgot the assignment and it is due in five minutes. Am I cooked?", "In that context, 'cooked' means you are in serious trouble or badly unprepared. Start with the smallest part you can submit."),
         ("So it is not about food?", "Right. Here it is slang, not literal cooking.")),
        (("My friend recovered from a terrible play and won. That was peak.", "You are praising the comeback as excellent or top-tier. 'Peak' is strong approval here."),
         ("real", "Agreed—the comeback made the whole match.")),
        (("He bragged and immediately fell over. Minus 500 aura.", "That jokes that the embarrassing fall damaged his imaginary coolness or status points."),
         ("Does aura literally use points?", "No. The points are an invented score used for the joke.")),
        (("Bro is NOT beating the allegations after doing that twice.", "That format jokes that repeated behavior is making a running accusation or stereotype look true."),
         ("I do not mean a real crime.", "Understood—you mean the meme format, not a legal allegation.")),
        (("What does Wojak mean?", "Wojak is a simple reaction character with many variants that represent feelings and online archetypes."),
         ("Is every drawing the same joke?", "No. The face, variant, caption, and context determine the emotion or stereotype.")),
        (("Explain Virgin vs Chad without taking it seriously.", "It exaggerates a supposedly awkward approach against a confident one. The contrast is often ironic, so neither side has to be a real judgment."),
         ("Keep it short next time.", "Got it—I will keep the next answer brief.")),
        (("Let's do math.", "Sure. Give me the calculation or word problem."),
         ("No math actually.", "No problem—we can switch topics. What sounds better?")),
        (("Write code for me.", "Tell me the language and what it should do."),
         ("Don't give me code. Just explain the idea.", "Understood. I will explain the approach in plain language without a code block.")),
        (("Tell me the long version.", "I can give the detailed explanation."),
         ("Stop explaining.", "Okay. I will stop there.")),
    ]
    for dialogue in contextual_meme_dialogues:
        add_unique(rows, seen, conversation(*dialogue))

    rng.shuffle(rows)
    if len(rows) < target:
        raise RuntimeError(f"Builder produced only {len(rows):,} unique conversations; need {target:,}")
    return rows[:target]


def validate(records: list[dict[str, object]], target: int) -> dict[str, int]:
    if len(records) != target:
        raise ValueError(f"Expected {target:,} records, got {len(records):,}")
    fingerprints: set[str] = set()
    answers: list[str] = []
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Record {index} has invalid messages")
        roles = [message.get("role") for message in messages]
        if roles[0] != "user" or roles[-1] != "assistant" or any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError(f"Record {index} has broken role order: {roles}")
        if any(not str(message.get("content", "")).strip() for message in messages):
            raise ValueError(f"Record {index} contains empty content")
        fingerprint = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate conversation at record {index}")
        fingerprints.add(fingerprint)
        answers.extend(str(message["content"]) for message in messages if message["role"] == "assistant")
    mojibake = [text for text in answers if re.search(r"(?:Ã.|â€|â€™|ï¿½|�)", text)]
    if mojibake:
        raise ValueError(f"Found {len(mojibake)} mojibake-corrupted assistant messages")
    maximum_frequency = max(Counter(answers).values())
    if maximum_frequency > 150:
        raise ValueError(f"One assistant response appears {maximum_frequency} times; limit is 150")
    return {"records": len(records), "unique": len(fingerprints), "assistant_messages": len(answers), "max_answer_frequency": maximum_frequency}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_CONVERSATIONS)
    args = parser.parse_args()
    records = build(target=args.target)
    metrics = validate(records, args.target)
    serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(serialized, encoding="utf-8")
    RAW_OUTPUT.write_text(serialized, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote balanced Public corpus to {OUTPUT} and {RAW_OUTPUT}")


if __name__ == "__main__":
    main()
