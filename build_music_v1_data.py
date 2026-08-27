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
    "starting over after a bad year", "falling for someone at the wrong time",
    "friends growing apart", "driving nowhere after midnight", "wanting to feel alive again",
    "a city that never answers", "outgrowing an old version of yourself",
    "a summer night that ended too soon", "the courage to finally leave",
    "being homesick in your own room", "a victory nobody noticed", "dancing through a power outage",
    "a spaceship held together with tape", "an arcade machine with one last life",
    "a villain who only wanted a day off", "the moon losing its password",
]
STYLES = [
    "dark electronic", "bright synth-pop", "melancholy piano ballad", "chaotic hyperpop",
    "garage rock", "dreamy ambient pop", "playful rap", "cinematic industrial",
    "lo-fi bedroom pop", "dramatic power metal", "minimal acoustic", "glitchy dance",
    "alternative R&B", "pop-punk", "cinematic synthwave", "indie folk",
    "drum and bass", "dark trap", "new-wave rock", "soulful dream pop",
    "orchestral hip-hop", "disco funk", "post-rock", "electro-swing",
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
    openings = [
        ("The hallway hums in a tired key,", "A tiny red light keeps judging me."),
        ("Streetlights blink like they know my name,", "Every green signal turns back into rain."),
        ("I woke to the sound of the ceiling fan,", "Counting every reason I abandoned the plan."),
        ("The dashboard glows on an empty road,", "I carry the quiet like a secret code."),
        ("Morning arrived with its shoes untied,", "Dragging a blue little cloud inside."),
        ("The speakers crackle, the room turns gold,", "We dance like the night will never get old."),
    ]
    second_verses = [
        ("The clock talks big but it moves too slow,", "I pack up the noise with nowhere to go."),
        ("We taped every warning over the door,", "Then turned up the bass and asked it for more."),
        ("A bus rolls past with the windows bright,", "Carrying strangers out of the night."),
        ("I write one truth on a paper cup,", "The world gets strange, but we still show up."),
        ("The old beat stumbles, then finds its feet,", "A beautiful error beneath the street."),
        ("I trade my fear for a second-hand tune,", "And launch every doubt at the innocent moon."),
    ]
    bridges = [
        "No perfect ending, no sensible chord—\nJust one brave beep and a deeply confused reward.",
        "Cut every light; let the low notes breathe.\nSometimes the truth is the sound underneath.",
        "If this is failure, let failure sing.\nGive it a pulse and a cheap gold ring.",
        "Half of me runs; half stays behind.\nThe snare keeps arguing with my mind.",
        "We bend, we break, we miss the cue—\nThen build a louder world from two.",
        "The whole room drops to a heartbeat alone.\nFor four quiet bars, I finally feel home.",
    ]
    first = openings[index % len(openings)]
    second = second_verses[(index * 3 + 1) % len(second_verses)]
    bridge = bridges[(index * 5 + 2) % len(bridges)]
    return (
        f"Title: {song_title}\nStyle: {style}; {mood}, with a clear pulse and a slightly unwise finale\n\n"
        f"[Verse 1]\n{first[0]}\n{first[1]}\nI made one promise to make it through,\nThen {subject} split the night in two.\n\n"
        f"[Chorus]\n{hook}\n\n"
        f"[Verse 2]\n{second[0]}\n{second[1]}\nA crooked rhythm crawls out of the wall,\nWe name it courage and dance through it all.\n\n"
        f"[Bridge]\n{bridge}\n\n"
        f"[Outro]\nThe signal bends, the screen goes black,\nI laugh once softly and don't look back."
    )


def build(seed: int = 2401) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    # Generate extra combinations, then deduplicate by the complete
    # conversation so the final curriculum contains 2,400 real uniques.
    for index in range(2_800):
        subject = SUBJECTS[index % len(SUBJECTS)]
        style = STYLES[(index // len(SUBJECTS)) % len(STYLES)]
        mood = MOODS[(index // (len(SUBJECTS) * len(STYLES))) % len(MOODS)]
        detail = DETAILS[(index * 7 + index // len(SUBJECTS)) % len(DETAILS)]
        mode = index % 12
        if mode == 0:
            rows.append(record(f"Write a {mood} {style} song about {subject}, {detail}.", full_song(subject, style, mood, index)))
        elif mode == 1:
            rows.append(record(f"Make {mood} {style} music about {subject}, {detail}. I want complete original lyrics.", full_song(subject, style, mood, index)))
        elif mode == 2:
            rows.append(record(f"Write a full {mood} {style} song about {subject}, {detail}.", full_song(subject, style, mood, index)))
        elif mode == 3:
            rows.append(record(f"Make only a {mood} {style} chorus about {subject}, {detail}.", f"[Chorus]\n{chorus(subject, mood, index)}"))
        elif mode == 4:
            rows.append(record(f"Give me one stupid {style} song title about {subject}, {detail}.", title(subject, style, index)))
        elif mode == 5:
            names = [title(subject, style, index + offset) for offset in range(5)]
            rows.append(record(f"Give me five original {mood} {style} song names for {subject}, {detail}.", "\n".join(f"{n + 1}. {name}" for n, name in enumerate(names))))
        elif mode == 6:
            rows.append(record(f"Give me a title and production style for a {mood} song about {subject}, {detail}.", f"Title: {title(subject, style, index)}\nStyle: {style}; {mood} vocals, textured percussion, and a chorus that opens up dramatically."))
        elif mode == 7:
            rows.append(record(f"Describe a {style} style for a song about {subject}, {detail}. Do not write lyrics.", f"Style: {style} with {mood} vocals, a restrained opening, textured percussion, and a final section that becomes wider and stranger. Keep the production centered on the feeling of {subject}."))
        elif mode == 8:
            rows.append(record(f"Write a short {mood} {style} hook about {subject}, {detail}. Make it catchy.", f"[Hook]\n{chorus(subject, mood, index).splitlines()[0]}\n{chorus(subject, mood, index).splitlines()[1]}"))
        elif mode == 9:
            rows.append(record(f"Turn this original line into a {mood} {style} verse: 'The hallway swallowed every sound.' Theme: {subject}, {detail}.", f"[Verse]\nThe hallway swallowed every sound,\nLoose little echoes rolled around.\nI followed the glow where the lost things sleep,\nWhile {subject} kept time underneath."))
        elif mode == 10:
            rows.append(record(f"Improve these original lyrics without making them longer; make them {mood}, {detail}: 'night is bad / i am sad / computer gone / very wrong'", "Night turns cold when the screen goes black,\nI count every light that refuses to come back.\nThe room holds its breath; the old fan drones—\nA digital heart in a house of bones."))
        else:
            rows.append({"messages": [
                {"role": "user", "content": f"I want to make a {mood} {style} song about {subject}, {detail}."},
                {"role": "assistant", "content": f"Title: {title(subject, style, index)}\nStyle: {style}; {mood}, with textured percussion and a wide final chorus."},
                {"role": "user", "content": "Write the complete lyrics and keep that title and style."},
                {"role": "assistant", "content": full_song(subject, style, mood, index)},
                {"role": "user", "content": "What style and song name did we choose?"},
                {"role": "assistant", "content": f"Title: {title(subject, style, index)}\nStyle: {style}; {mood}, with textured percussion and a wide final chorus."},
            ]})
    # Vague real-world requests deserve the same full-song behavior as detailed
    # prompts. These are training conversations only; runtime remains pure
    # checkpoint generation with no prompt-to-answer lookup table.
    vague_prompts = (
        "Make music", "Write me a song", "Make me a full song", "Song please",
        "Give me original lyrics", "Create a complete song", "Make something musical",
        "I need a song", "Write the whole song", "Come up with a song for me",
    )
    for offset in range(400):
        index = 2_800 + offset
        subject = SUBJECTS[(offset * 11) % len(SUBJECTS)]
        style = STYLES[(offset * 7) % len(STYLES)]
        mood = MOODS[(offset * 5) % len(MOODS)]
        prompt = vague_prompts[offset % len(vague_prompts)]
        if offset % 3:
            rows.append(record(prompt, full_song(subject, style, mood, index)))
        else:
            chosen_title = title(subject, style, index)
            chosen_style = f"{style}; {mood}, with textured percussion and a wide final chorus"
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_song(subject, style, mood, index)},
                {"role": "user", "content": "What was the title and style?"},
                {"role": "assistant", "content": f"Title: {chosen_title}\nStyle: {chosen_style}."},
                {"role": "user", "content": "Revise the second verse but keep our title and style."},
                {"role": "assistant", "content": f"Title: {chosen_title}\nStyle: {chosen_style}.\n\n[Revised Verse 2]\nThe old beat stumbles, then finds its feet,\nA beautiful error beneath the street.\nI carry {subject} into the hall,\nThen name it courage and dance through it all."},
            ]})
    unique: dict[str, dict[str, object]] = {}
    for item in rows:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        unique.setdefault(key, item)
    rows = list(unique.values())
    if len(rows) < 2_700:
        raise RuntimeError(f"Music corpus produced only {len(rows):,} unique conversations")
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
