"""Build a deterministic, copyright-safe synthetic SFT corpus for Music V1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUBJECTS = [
    "my WiFi disappearing", "a microwave running for president", "two percent battery",
    "my GPU abandoning me", "dropping one AirPod", "a refrigerator boss battle",
    "a lonely vending machine", "a haunted group chat", "a keyboard demanding vacation",
    "a robot learning to dance", "rain on an empty arcade", "a duck stealing the moon",
    "missing the last bus", "an astronaut homesick for Earth", "a toaster with stage fright",
    "a broken controller", "the final day of summer", "a ghost inside a radio",
    "a very dramatic potato", "late-night city lights", "a friendship surviving distance",
    "an elevator that fears heights", "a cat becoming a DJ", "forgotten save files",
]
STYLES = [
    "dark electronic", "bright synth-pop", "melancholy piano ballad", "chaotic hyperpop",
    "garage rock", "dreamy ambient pop", "playful rap", "cinematic industrial",
    "lo-fi bedroom pop", "dramatic power metal", "minimal acoustic", "glitchy dance",
]
MOODS = ["sad", "funny", "ominous", "hopeful", "absurd", "nostalgic", "angry", "tender"]
NOUNS = ["signal", "static", "midnight", "plastic", "echo", "circuit", "neon", "orbit", "chrome", "rain"]
DETAILS = [
    "under violet streetlights", "during a thunderstorm", "inside an empty arcade",
    "at exactly 3:17 AM", "while the neighbors dance", "with one speaker blown out",
    "beneath a flickering sign", "on the last train home", "with a choir of dial-up modems",
    "as the city loses power", "in a room full of old TVs", "while a cheap fan rattles",
    "with a suspiciously heroic key change", "through a cracked phone speaker",
    "as a tiny robot keeps time", "without using the word love", "with an abrupt quiet ending",
    "with a chorus built for shouting", "like a transmission from nowhere", "with one terrible rhyme",
    "while rain hits the window", "with warm bass and cold vocals", "as the clock skips a minute",
    "with a final chord that refuses to resolve", "while a duck handles percussion",
    "with restrained verses and a huge chorus", "as if the demo file became self-aware",
    "with a hopeful last line", "with a deliberately broken bridge", "as the room slowly wakes up",
]


def record(user: str, assistant: str) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def title(subject: str, style: str, index: int) -> str:
    a = NOUNS[index % len(NOUNS)].title()
    b = subject.split()[(-1 - index) % len(subject.split())].strip(".,").title()
    return f"{a} {b}" if index % 2 else f"{b} After Midnight"


def chorus(subject: str, mood: str, index: int) -> str:
    core = subject.replace("my ", "the ")
    variants = [
        f"I call through the static, but the room stays blue,\n{core.capitalize()}—what am I supposed to do?\nOne blinking light and a heart off track,\nSend me a signal, bring the whole world back.",
        f"Oh, {core}, you glorious mistake,\nRattle every window, make the cheap floor shake.\nWe had no good plan, so we shouted the tune—\nNow the neighbors are dancing with a spoon and the moon.",
        f"Keep the last light on, keep the doorway warm,\nWe can make a little music in the middle of the storm.\nIf {core} fades beyond the view,\nI'll build a beat from the silence and carry it through.",
    ]
    return variants[index % len(variants)]


def full_song(subject: str, style: str, mood: str, index: int) -> str:
    song_title = title(subject, style, index)
    hook = chorus(subject, mood, index)
    return (
        f"[{song_title}]\nStyle: {style}; {mood}, with a clear pulse and a slightly unwise finale\n\n"
        f"[Verse 1]\nThe hallway hums in a tired key,\nA tiny red light keeps judging me.\nI made one promise to make it through,\nThen {subject} split the night in two.\n\n"
        f"[Chorus]\n{hook}\n\n"
        f"[Verse 2]\nThe clock talks big but it moves too slow,\nI pack up the noise with nowhere to go.\nA crooked rhythm crawls out of the wall,\nWe name it courage and dance through it all.\n\n"
        f"[Bridge]\nNo perfect ending, no sensible chord—\nJust one brave beep and a deeply confused reward.\n\n"
        f"[Outro]\nThe signal bends, the screen goes black,\nI laugh once softly and don't look back."
    )


def build(seed: int = 2401) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for index in range(720):
        subject = SUBJECTS[index % len(SUBJECTS)]
        style = STYLES[(index // len(SUBJECTS)) % len(STYLES)]
        mood = MOODS[(index // (len(SUBJECTS) * len(STYLES))) % len(MOODS)]
        detail = DETAILS[(index * 7 + index // len(SUBJECTS)) % len(DETAILS)]
        mode = index % 10
        if mode == 0:
            rows.append(record(f"Write a {mood} {style} song about {subject}, {detail}.", full_song(subject, style, mood, index)))
        elif mode == 1:
            rows.append(record(f"Make only a chorus about {subject}, {detail}.", f"[Chorus]\n{chorus(subject, mood, index)}"))
        elif mode == 2:
            rows.append(record(f"Give me one stupid song title about {subject}, {detail}.", title(subject, style, index)))
        elif mode == 3:
            names = [title(subject, style, index + offset) for offset in range(5)]
            rows.append(record(f"Give me five original song names for {subject}, {detail}.", "\n".join(f"{n + 1}. {name}" for n, name in enumerate(names))))
        elif mode == 4:
            rows.append(record(f"Describe a {style} style for a song about {subject}, {detail}. Do not write lyrics.", f"Style: {style} with {mood} vocals, a restrained opening, textured percussion, and a final section that becomes wider and stranger. Keep the production centered on the feeling of {subject}."))
        elif mode == 5:
            rows.append(record(f"Write a short hook about {subject}, {detail}. Make it catchy.", f"[Hook]\n{chorus(subject, mood, index).splitlines()[0]}\n{chorus(subject, mood, index).splitlines()[1]}"))
        elif mode == 6:
            rows.append(record(f"Turn this original line into a verse: 'The hallway swallowed every sound.' Theme: {subject}, {detail}.", f"[Verse]\nThe hallway swallowed every sound,\nLoose little echoes rolled around.\nI followed the glow where the lost things sleep,\nWhile {subject} kept time underneath."))
        elif mode == 7:
            rows.append(record(f"Improve these original lyrics without making them longer; make them {mood}, {detail}: 'night is bad / i am sad / computer gone / very wrong'", "Night turns cold when the screen goes black,\nI count every light that refuses to come back.\nThe room holds its breath; the old fan drones—\nA digital heart in a house of bones."))
        elif mode == 8:
            rows.append(record(f"Make intentionally horrible lyrics about {subject}, {detail}.", f"[Verse]\nOh {subject}, you happened today,\nThe rhyme department has run away.\nMy beat is a chair and my chorus is bread,\nThis song owes twelve dollars and lives in a shed."))
        else:
            rows.append(record(f"Write an outro for a {mood} song about {subject}, {detail}; no full song.", f"[Outro]\nLet the last note flicker where the streetlights end.\n{subject.capitalize()} won't be the ending—just a strange new bend."))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/music_v1_conversations.jsonl")
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} unique original Music V1 conversations to {output}")


if __name__ == "__main__":
    main()
