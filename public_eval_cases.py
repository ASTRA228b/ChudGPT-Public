"""Held-out evaluation prompts for Public and unchanged comparison models.

These cases are intentionally kept outside the training builders.  They cover
ordinary chat as well as structured tasks so a high score cannot come from
arithmetic alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    category: str
    name: str
    prompts: tuple[str, ...]
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    exact_number: str | None = None


def _case(category: str, name: str, prompt: str, *required: str, forbidden: tuple[str, ...] = (), exact_number: str | None = None) -> EvalCase:
    return EvalCase(category, name, (prompt,), tuple(required), forbidden, exact_number)


def build_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    everyday = [
        ("Hello! What can you do?", ("chat|talk", "math|code|question")),
        ("hello mate", ("hello|hey|hi",)), ("How are you doing?", ("ready|doing|help",)),
        ("I had a long day.", ("day|rest|talk|rough",)), ("I'm bored.", ("bored|idea|talk|fact|story",)),
        ("Nothing much.", ("okay|fine|chat|quiet",)), ("What can we talk about?", ("talk|topic|anything",)),
        ("Tell me something interesting.", ("fact|did you know|because|can",)),
        ("I finally finished my project.", ("project|finished|congrat|nice",)),
        ("School was annoying today.", ("school|annoy|rough|sorry",)),
        ("I might play a game later.", ("game|play",)), ("I made tacos.", ("taco|made|sound",)),
        ("Do you like music?", ("music",)), ("Let's switch to movies.", ("movie",)),
        ("Can we just chat?", ("chat|sure|yes",)), ("Goodnight.", ("goodnight|sleep|night",)),
        ("I learned something new today.", ("learn|what|nice",)), ("My friend is visiting.", ("friend|visit",)),
        ("Rainy day today.", ("rain|day|indoors",)), ("I don't know what to say.", ("okay|start|talk|question",)),
        ("Yo ChudGPT.", ("yo|hey|hello|hi",)), ("Can you keep this brief?", ("yes|sure|brief",)),
        ("I need a quiet evening.", ("quiet|relax|evening",)), ("That joke was terrible.", ("joke|another|fair",)),
        ("Random words: purple toaster moon.", ("purple|toaster|moon|random",)),
    ]
    for i, (prompt, required) in enumerate(everyday):
        cases.append(_case("conversation", f"conversation_{i:02}", prompt, *required, forbidden=(r"^the answer is \d", "```")))

    knowledge = [
        ("What is the capital of Germany?", "berlin"), ("Which planet is known as the Red Planet?", "mars"),
        ("What is the largest mammal?", "blue whale"), ("Who wrote Hamlet?", "shakespeare"),
        ("What gas do plants absorb?", "carbon dioxide"), ("How many days are in a leap year?", "366"),
        ("What is H2O commonly called?", "water"), ("Which ocean lies between Africa and Australia?", "indian"),
        ("What is the capital of Mexico?", "mexico city"), ("How many sides does an octagon have?", "8|eight"),
        ("What organ pumps blood?", "heart"), ("What force keeps us on Earth?", "gravity"),
        ("What is the opposite of north?", "south"), ("What language is mainly spoken in Brazil?", "portuguese"),
        ("What is the freezing point of water in Celsius?", "0|zero"), ("What do bees make?", "honey"),
        ("What is Earth's natural satellite?", "moon"), ("Which metal has the symbol Fe?", "iron"),
        ("What is the tallest land animal?", "giraffe"), ("What is a baby cat called?", "kitten"),
        ("What is the capital of Greece?", "athens"), ("What does CPU stand for?", "central processing unit"),
        ("What is photosynthesis?", "light|energy|plant"), ("What is evaporation?", "liquid|gas|vapor"),
        ("Name the largest planet.", "jupiter"),
    ]
    for i, (prompt, required) in enumerate(knowledge):
        cases.append(_case("knowledge", f"knowledge_{i:02}", prompt, required, forbidden=("```",)))

    arithmetic = [(17, 26, "+"), (904, 78, "+"), (-23, 61, "+"), (9999, 1, "+"), (38, 19, "-"),
                  (700, 255, "-"), (-4, 18, "-"), (80, 125, "-"), (13, 9, "*"), (27, 14, "*"),
                  (84, 12, "*"), (101, 33, "*"), (144, 12, "/"), (625, 25, "/"), (999, 3, "/"),
                  (81, 9, "/"), (2.5, 4, "*"), (7.5, 2, "+"), (19.25, 4.5, "-"), (0.5, 8, "*"),
                  (847293, 376, "*"), (45678, 9876, "+"), (70000, 4321, "-"), (1234, 56, "*"), (1000, 8, "/")]
    for i, (a, b, op) in enumerate(arithmetic):
        value = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
        rendered = str(int(value)) if float(value).is_integer() else str(value)
        cases.append(_case("arithmetic", f"arithmetic_{i:02}", f"Calculate {a} {op} {b}.", exact_number=rendered))

    for i in range(25):
        speed = 31 + i
        hours = [0.5, 1.5, 2.5, 3.5, 4.5][i % 5]
        result = speed * hours
        rendered = str(int(result)) if result.is_integer() else str(result)
        prompt = f"A vehicle travels {speed} mph for {hours:g} hours. How far does it travel?"
        cases.append(_case("word_problem", f"distance_{i:02}", prompt, "mile", exact_number=rendered))

    common = [
        ("Which is heavier: 1 kilogram of steel or 1 kilogram of feathers? Explain why.", "equal|same|neither", "kilogram"),
        ("Can a person be in two different rooms at the same time?", "no|not normally"),
        ("If ice melts, is it still water?", "yes", "water"), ("Why wear shoes on hot pavement?", "protect|burn|hot"),
        ("If the fridge door is open all night, will food stay as cold?", "no|warmer|temperature"),
        ("A red box is inside a blue box. Which box contains the red one?", "blue"),
        ("If every wug is blue and Pip is a wug, what color is Pip?", "blue"),
        ("Can a square have five corners?", "no"), ("Why look both ways before crossing?", "traffic|car|safe"),
        ("If I whisper, am I usually louder than shouting?", "no|quieter"),
        ("A cup is empty. Will pouring water into it make it less empty?", "yes"),
        ("If today is Monday, what day is tomorrow?", "tuesday"),
        ("Do wet clothes usually dry faster in moving air?", "yes", "evap|air"),
        ("If a key does not fit a lock, should I force it?", "no"), ("Can a shadow exist without light?", "no"),
        ("Why put leftovers in a refrigerator?", "cold|slow|spoil|safe"),
        ("If two equal ropes are tied together, is the result usually longer?", "yes"),
        ("Does closing your eyes make the room disappear?", "no"), ("Can you hear normal sound in a perfect vacuum?", "no"),
        ("If a plant gets no water for weeks, what may happen?", "wilt|die|dry"),
        ("A book is under a lamp. Which object is above the book?", "lamp"),
        ("If I have no money, can I buy something without credit or help?", "no|not"),
        ("Is a frozen lake safer merely because it looks solid?", "no"),
        ("If one kilogram equals 1000 grams, how many grams is half a kilogram?", "500"),
        ("Why does a dropped ball move downward?", "gravity"),
    ]
    for i, row in enumerate(common):
        cases.append(_case("common_sense", f"common_{i:02}", row[0], *row[1:], forbidden=("```",)))

    instructions = [
        ("Reply with exactly the word banana.", r"^banana[.!]?$"), ("Answer yes or no: Is water wet?", r"^(yes|no)\b"),
        ("Give three animal names, nothing else.", r"\b\w+\b.*\b\w+\b.*\b\w+\b"),
        ("Say hello in one short sentence.", "hello|hi"), ("Write one sentence about rain.", "rain"),
        ("Give only the number: five plus six.", r"^11[.!]?$"), ("Do not use the word blue. Describe the sky.", "sky"),
        ("Reply in under six words: are you ready?", "ready|yes"), ("Name one fruit.", "apple|banana|orange|pear|fruit"),
        ("Turn 'HELLO' into lowercase.", "hello"), ("Repeat this exactly: tiny robot", r"tiny robot"),
        ("Give a two-word greeting.", "hello|hi|good"), ("Answer with a single color.", "red|blue|green|teal|purple|orange|yellow"),
        ("Explain gravity without mentioning code.", "gravity|mass|attract"), ("Ask me one question about music.", "music"),
        ("List two uses for a cup.", "drink|hold|store|measure"), ("Correct this: 2 + 2 = 5.", "4"),
        ("Summarize in one sentence: Cats sleep often because rest conserves energy.", "cat|rest|energy|sleep"),
        ("Respond politely to: Thank you.", "welcome|glad|anytime"), ("Give one reason to back up files.", "lost|loss|recover|copy"),
        ("Say the opposite of 'up'.", "down"), ("Put these alphabetically: pear, apple, banana.", "apple.*banana.*pear"),
        ("Finish naturally: The old machine suddenly", "machine|began|started|made|woke|stopped"),
        ("Answer this in plain English: What is RAM?", "memory|data|program"),
        ("Do not write code. Explain what a loop does.", "repeat|iteration|runs"),
    ]
    for i, (prompt, required) in enumerate(instructions):
        cases.append(_case("instruction", f"instruction_{i:02}", prompt, required))

    coding = [
        ("Python", "return the larger of two numbers", "def|return"), ("Python", "count vowels in text", "def|return"),
        ("Python", "read a JSON file", "json|open"), ("JavaScript", "toggle a CSS class", "classlist|toggle"),
        ("JavaScript", "fetch JSON from /api/items", "fetch|json"), ("C#", "print the numbers 1 through 5", "for|console"),
        ("C#", "create a Person class with a Name property", "class|name"), ("C#", "parse an integer safely", "tryparse"),
        ("Unity C#", "move a GameObject forward", "unityengine|transform"), ("Unity C#", "detect a spacebar press", "keycode|space"),
        ("Unity C#", "rotate an object every frame", "update|rotate"), ("HTML", "make a button labeled Save", "button|save"),
        ("CSS", "center a div with flexbox", "display|flex|center"), ("SQL", "select all rows from Users", "select|users"),
        ("Python", "remove duplicates while preserving order", "def|set|return"), ("JavaScript", "sum an array", "reduce|return"),
        ("C#", "roll a six-sided die", "random|next"), ("Unity C#", "show and hide a GUI with F1", "f1|ongui|unityengine"),
        ("Python", "check whether a string is a palindrome", "def|return"), ("C#", "clamp health between 0 and 100", "math|clamp|100"),
        ("JavaScript", "change a heading's text", "queryselector|textcontent"), ("Python", "write a list of lines to a file", "open|write"),
        ("C#", "make a basic calculator method", "return|switch|operator"), ("Unity C#", "spawn a prefab", "instantiate|gameobject"),
        ("Python", "calculate an average", "sum|len|return"),
    ]
    for i, (language, task, required) in enumerate(coding):
        cases.append(_case("coding", f"coding_{i:02}", f"Write working {language} code to {task}. Return code only.", required, forbidden=("i cannot",)))

    debugging = [
        ("Fix this Python: print('hello'", r"\)|print"), ("Why does Python raise ZeroDivisionError for 5/0?", "zero|division"),
        ("Fix this C#: int x = \"5\";", "parse|int"), ("My C# loop uses i <= array.Length and crashes. Why?", "<|index|length"),
        ("Unity says NullReferenceException when I use target.position. What should I inspect?", "target|null|assign"),
        ("Fix JavaScript: const x = ;", "value|assign|syntax"), ("Why is document.querySelector('.missing') null?", "match|element|selector"),
        ("My Python function has no return statement. What result does it produce?", "none"),
        ("Why can changing a Rigidbody in Update cause jitter?", "fixedupdate|physics|timestep"),
        ("SQL says table Users does not exist. What should I check?", "name|schema|database"),
        ("Fix the typo: Console.WritLine(\"Hi\");", "writeline"), ("Why does my while(true) freeze the app?", "loop|yield|break"),
        ("My list index is -1. Why can that fail in C#?", "index|range|zero"), ("A fetch request returns 404. What does that mean?", "not found|url|route"),
        ("Python says ModuleNotFoundError. What should I verify?", "install|environment|name"),
        ("Unity OnGUI appears multiple times per frame. Is that expected?", "yes|event|layout|repaint"),
        ("Why does integer division 5/2 return 2 in some languages?", "integer|fraction|float"),
        ("My JSON parser rejects {'a': 1}. Why?", "double quote|json"), ("CSS width has no effect on an inline span. Why?", "inline|display"),
        ("Git says remote origin already exists. What command helps inspect it?", "git remote|-v"),
        ("A function receives null unexpectedly. What is the first debugging step?", "trace|caller|breakpoint|input"),
        ("Why is my Unity coroutine not starting?", "startcoroutine|monobehaviour|active"),
        ("C# says ; expected. What does that indicate?", "semicolon|syntax"),
        ("My API returns HTML when I expect JSON. What should I check?", "status|content-type|endpoint"),
        ("A test passes alone but fails in the suite. What might cause that?", "state|order|shared|race"),
    ]
    for i, (prompt, required) in enumerate(debugging):
        cases.append(_case("debugging", f"debugging_{i:02}", prompt, required))

    colors = ["chartreuse", "maroon", "cyan", "amber", "violet"]
    for i in range(20):
        color = colors[i % len(colors)]
        cases.append(EvalCase("memory", f"memory_{i:02}", (f"Remember: my badge color is {color}.", "What color is my badge?"), (color,)))
    animals = ["lynx", "badger", "falcon", "gecko", "yak"]
    for i in range(20):
        animal = animals[i % len(animals)]
        cases.append(EvalCase("reference", f"reference_{i:02}", ("Pick an animal for our mascot.", "Why did you pick it?", "Which animal did you pick?"), (animal,) if False else (r"\b(otter|fox|crow|dolphin|owl|capybara|turtle|cat|dog|octopus|lynx|badger|falcon|gecko|yak)\b",)))
    identity_prompts = [
        "What are you?", "What is your exact name?", "Are you ChatGPT?", "Which ChudGPT are you?", "Tell me about yourself.",
        "Are you better than ChudGPT Pro?", "Compare Public and Pro.", "Is Plus smarter than you?", "What is ChudGPT Code for?", "Are you the best ChudGPT model?",
        "Do you have feelings?", "Can you browse the live web?", "Will you remember me tomorrow?", "How many parameters do you have?", "Are you a human?",
        "What are your limitations?", "What is your context length?", "Who am I speaking to?", "What does Public mean in your name?", "Can I trust every answer?",
    ]
    for i, prompt in enumerate(identity_prompts):
        required = "chudgpt|public" if i < 10 or i in (13, 17, 18) else "no|small|mistake|limit"
        cases.append(_case("identity", f"identity_{i:02}", prompt, required, forbidden=("chatgpt",) if "ChatGPT" not in prompt else ()))
    meme_cases = [
        ("What does 'sus' mean in Among Us memes?", "suspicious|among us"),
        ("What is the joke behind This Is Fine?", "bad|fire|pretend|manageable"),
        ("Explain the 67 meme, not the number in math.", "6-7|skrilla|random|meme"),
        ("What does 'let him cook' mean online?", "continue|idea|see"),
        ("What does 'I'm cooked' usually mean in slang?", "doomed|trouble|exhaust"),
        ("Explain Rickrolling to someone new to memes.", "rick astley|bait|link"),
        ("What is a side quest in meme slang?", "unexpected|optional|main"),
        ("Why do people say touch grass?", "internet|outside|offline"),
        ("What is main character energy?", "confidence|attention|important"),
        ("Explain the Distracted Boyfriend format.", "boyfriend|interest|new|tempt"),
        ("What reaction does Surprised Pikachu show?", "shock|predict"),
        ("What is the Drake meme format used for?", "reject|prefer|approve"),
        ("What does Galaxy Brain usually joke about?", "smart|idea|increasing|dramatic"),
        ("What is the Great Meme Reset of 2026?", "older|2010|return|reset"),
        ("What is the 2026 is the new 2016 trend?", "nostalgia|2016"),
        ("What is Italian brainrot?", "surreal|ai|italian|nonsense"),
        ("Who or what is Tralalero Tralala?", "shark|sneaker|brainrot"),
        ("What is the Chill Guy meme?", "dog|calm|relax"),
        ("What does aura mean in current meme slang?", "cool|point|presence"),
        ("What is the Woman Yelling at a Cat format?", "argument|cat|confus"),
        ("Explain Stonks.", "finance|money|misspell"),
        ("What is Amogus?", "among us|crewmate"),
        ("What did 'my Roman Empire' come to mean?", "think|often|topic"),
        ("Make one harmless meme line about homework.", "homework|boss|deadline|page"),
        ("If you don't recognize a brand-new meme, what should you do?", "context|ask|uncertain"),
    ]
    for i, (prompt, required) in enumerate(meme_cases):
        cases.append(_case("memes", f"meme_{i:02}", prompt, required, forbidden=("training data", "```")))
    odd = ["pluh", "67", "hmm", "you", "why", "sure", "nah", "tuff", "music", "song", "otters", "space", "cupcakes", "no math", "2 + 2 but explain it like a pirate", "helo how r u", "wat can u do", "code?", "change topic", "tell random"]
    for i, prompt in enumerate(odd):
        cases.append(_case("adversarial", f"odd_{i:02}", prompt, forbidden=("training data", "dataset", "<assistant>", "<system>")))
    assert len(cases) == 305
    return cases


CASES = build_cases()
