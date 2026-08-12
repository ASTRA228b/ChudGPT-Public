"""Build a large, project-authored conversational dataset for ChudGPT-Public."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "public_conversations.jsonl"
RAW_OUTPUT = ROOT / "data" / "raw" / "public_corpus.jsonl"
SEED = 228
TARGET_CONVERSATIONS = 20_000

FACTS = [
    ("the sky looks blue", "sunlight is scattered by the atmosphere, and shorter blue wavelengths scatter more strongly than red wavelengths"),
    ("plants grow", "they use light, water, and carbon dioxide during photosynthesis to make sugars"),
    ("the Moon has phases", "we see changing portions of its sunlit half as it orbits Earth"),
    ("seasons happen", "Earth's axis is tilted, changing sunlight angle and day length during its orbit"),
    ("rain forms", "water vapor cools and condenses into droplets that become heavy enough to fall"),
    ("oceans are salty", "rivers and geological processes carry dissolved minerals into seawater over long periods"),
    ("sound needs a medium", "sound is a vibration traveling through matter such as air, water, or solids"),
    ("metal can rust", "iron reacts with oxygen and water to form iron oxides"),
    ("exercise raises heart rate", "working muscles need more oxygen and nutrients delivered by blood"),
    ("sleep matters", "sleep supports learning, memory, immune function, and physical recovery"),
    ("Earth is round", "gravity pulls matter toward the center, producing a nearly spherical planet"),
    ("ice floats", "solid water forms an open crystal structure that is less dense than liquid water"),
]

TOPICS = [
    "music", "movies", "games", "books", "space", "animals", "weather", "food",
    "history", "art", "sports", "school", "travel", "friendship", "technology",
    "nature", "photography", "cars", "robots", "architecture", "gardening", "cooking",
]

OPENERS = [
    "Sure", "Absolutely", "Good question", "Let's work through it", "Here is the short version",
    "I can help with that", "That is worth exploring", "A practical answer is",
]

JOKES = [
    "Why did the computer take a nap? It had too many tabs open.",
    "I told my keyboard a joke. It said it needed more space.",
    "Why was the calendar nervous? Its days were numbered.",
    "The robot opened a bakery because it was great at processing dough.",
    "Why did the bicycle stop? It was two-tired.",
]

CODE_EXAMPLES = [
    ("Python", "print a greeting", "name = input('What is your name? ')\nprint(f'Hello, {name}!')"),
    ("Python", "add two numbers", "def add(a: float, b: float) -> float:\n    return a + b"),
    ("C#", "roll a six-sided die", "using System;\n\nclass Program\n{\n    static void Main()\n    {\n        int roll = Random.Shared.Next(1, 7);\n        Console.WriteLine($\"You rolled {roll}\");\n    }\n}"),
    ("JavaScript", "change some page text", "const heading = document.querySelector('h1');\nheading.textContent = 'Hello from JavaScript!';"),
    ("C# for Unity", "move a GameObject", "using UnityEngine;\n\npublic class SimpleMover : MonoBehaviour\n{\n    [SerializeField] private float speed = 5f;\n\n    private void Update()\n    {\n        float x = Input.GetAxisRaw(\"Horizontal\");\n        float z = Input.GetAxisRaw(\"Vertical\");\n        transform.position += new Vector3(x, 0f, z).normalized * speed * Time.deltaTime;\n    }\n}"),
]

INFO_QA = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
    ("What is the capital of Canada?", "Ottawa is the capital of Canada."),
    ("What is the capital of Australia?", "Canberra is the capital of Australia."),
    ("What is the capital of Texas?", "Austin is the capital of the U.S. state of Texas."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the tallest mountain above sea level?", "Mount Everest is the tallest mountain above sea level."),
    ("How many continents are commonly recognized?", "Seven continents are commonly recognized: Africa, Antarctica, Asia, Europe, North America, South America, and Australia."),
    ("What planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
    ("What is the largest planet?", "Jupiter is the largest planet in the Solar System."),
    ("What gas do humans need to breathe?", "Humans need oxygen for cellular respiration, although normal air also contains mostly nitrogen."),
    ("At what temperature does pure water freeze?", "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius or 32 degrees Fahrenheit."),
    ("What is gravity?", "Gravity is the attraction between objects with mass. Near Earth, it pulls objects toward the planet's center."),
    ("What is an atom?", "An atom is a basic unit of ordinary matter made of a nucleus surrounded by electrons."),
    ("What is DNA?", "DNA is a molecule that carries genetic instructions used by living organisms."),
    ("What does CPU mean?", "CPU means central processing unit. It executes instructions and coordinates much of a computer's work."),
    ("What is RAM?", "RAM is short-term working memory that a computer uses for programs and data currently in use."),
    ("What is an operating system?", "An operating system manages hardware and provides common services for applications."),
    ("What is the internet?", "The internet is a global network of interconnected computer networks that communicate using standard protocols."),
    ("What is a web browser?", "A web browser retrieves and displays websites and lets people interact with web content."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is photosynthesis?", "Photosynthesis is the process plants and some other organisms use to turn light energy into chemical energy."),
    ("What is evaporation?", "Evaporation happens when molecules at a liquid's surface gain enough energy to become gas."),
    ("Why should important facts be verified?", "Sources can be outdated, incomplete, or wrong, and a small language model can hallucinate. Important facts should be checked against reliable current sources."),
]


def conversation(user: str, assistant: str, follow_user: str | None = None, follow_assistant: str | None = None) -> dict[str, object]:
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    if follow_user and follow_assistant:
        messages.extend([{"role": "user", "content": follow_user}, {"role": "assistant", "content": follow_assistant}])
    return {"messages": messages, "source": "chudgpt-public-custom"}


def build(seed: int = SEED, target: int = TARGET_CONVERSATIONS) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    identity_prompts = ["What is your name?", "Who are you?", "Tell me about yourself.", "Are you ChudGPT?"]
    identity_answers = [
        "My name is ChudGPT-Public. I am a small experimental decoder-only transformer language model with 20,999,184 trainable parameters and a 1,024-token context window.",
        "I am ChudGPT-Public, a locally runnable conversational language model. I generate text from learned patterns; I am not a person and I do not have feelings or personal experiences.",
        "I am ChudGPT-Public. I can help with basic information, arithmetic, simple code, stories, and conversation, but I am small and can make confident mistakes.",
        "ChudGPT-Public is my name. I have no live internet access unless a separate program explicitly adds it, and I only know the current conversation supplied to me.",
    ]
    for index in range(1_000):
        prompt = identity_prompts[index % len(identity_prompts)]
        records.append(conversation(prompt, identity_answers[index % len(identity_answers)]))
    for index in range(3_000):
        subject, explanation = FACTS[index % len(FACTS)]
        opener = rng.choice(OPENERS)
        records.append(conversation(
            f"Why does {subject}?",
            f"{opener}: {explanation}. That is the main idea, though the full science can include more detail.",
            rng.choice(["Can you say that more simply?", "Why is that useful to know?", "Tell me one more detail."]),
            f"In simple terms, {explanation}. A good way to remember it is to focus on cause and effect rather than memorizing a label.",
        ))
    for index in range(3_000):
        topic = TOPICS[index % len(TOPICS)]
        records.append(conversation(
            rng.choice([f"Let's talk about {topic}.", f"What is interesting about {topic}?", f"I like {topic}."]),
            f"{rng.choice(OPENERS)}. {topic.capitalize()} can be approached through personal taste, history, technique, and the way it connects people. What part of {topic} interests you most?",
            rng.choice(["The creative side.", "Tell me something surprising.", "How do I learn more?"]),
            f"The best next step is to choose one small part of {topic}, explore a real example, and notice what you enjoy or question about it. That gives us something specific to discuss.",
        ))
    for index in range(2_500):
        a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        operation = index % 4
        if operation == 0:
            question, answer = f"What is {a} + {b}?", str(a + b)
        elif operation == 1:
            question, answer = f"What is {a} - {b}?", str(a - b)
        elif operation == 2:
            question, answer = f"What is {a} times {b}?", str(a * b)
        else:
            divisor = b or 1
            question, answer = f"What is {a * divisor} divided by {divisor}?", str(a)
        records.append(conversation(question, f"The answer is {answer}."))
    for index in range(2_000):
        language, task, code = CODE_EXAMPLES[index % len(CODE_EXAMPLES)]
        records.append(conversation(
            f"Write {language} code to {task}.",
            f"Here is a complete small example:\n\n```{language.lower().replace(' for unity', '')}\n{code}\n```\n\nYou can run it as written, then change the names or values to fit your project.",
            "Explain the main idea.",
            "The program keeps the task small: it reads or creates the needed value, performs one clear action, and displays or applies the result. Each named variable makes the behavior easier to modify.",
        ))
    for index in range(1_500):
        character = rng.choice(["robot", "duck", "astronaut", "tiny dragon", "old machine", "lost cat"])
        place = rng.choice(["a moonlit city", "an underwater library", "a quiet forest", "a broken arcade", "a distant station"])
        records.append(conversation(
            f"Write a short story about a {character} in {place}.",
            f"A {character} arrived in {place} carrying one question and no map. It followed a strange light, helped a stranger solve a small problem, and discovered that the answer was not a treasure but a new direction. By morning, the path home was clear, but the {character} chose to stay for one more adventure.",
        ))
    for index in range(1_000):
        records.append(conversation(
            rng.choice(["Tell me a joke.", "Make me laugh.", "Do you know a clean joke?"]),
            JOKES[index % len(JOKES)],
        ))
    for index in range(2_000):
        feeling = rng.choice(["stressed", "bored", "confused", "excited", "tired", "nervous"])
        records.append(conversation(
            f"I feel {feeling} today.",
            f"That sounds {feeling}, and I’m glad you said it. We do not have to solve everything at once. Would you rather talk through what happened, take one small practical step, or just have a lighter conversation for a bit?",
            rng.choice(["Just listen.", "Give me one small step.", "Let's change the topic."]),
            "I’m here with you. We can slow down, keep it simple, and take the conversation in the direction that feels most helpful right now.",
        ))
    for index in range(4_000):
        question, answer = INFO_QA[index % len(INFO_QA)]
        records.append(conversation(question, answer))
    rng.shuffle(records)
    return records[:target]


def main() -> None:
    records = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    OUTPUT.write_text(serialized, encoding="utf-8")
    RAW_OUTPUT.write_text(serialized, encoding="utf-8")
    turns = sum(len(record["messages"]) for record in records)
    print(f"Wrote {len(records):,} conversations and {turns:,} messages.")
    print(f"SFT dataset: {OUTPUT}")
    print(f"Pretraining corpus: {RAW_OUTPUT}")


if __name__ == "__main__":
    main()
