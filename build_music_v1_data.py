"""Build a varied, copyright-safe synthetic SFT corpus for Music V1."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUBJECTS = [
    "my WiFi disappearing", "a microwave running for president", "two percent battery",
    "my GPU abandoning me", "dropping one AirPod", "a refrigerator boss battle",
    "a lonely vending machine", "a haunted group chat", "a keyboard demanding vacation",
    "a robot learning to dance", "rain on an empty arcade", "a duck stealing the moon",
    "missing the last bus", "an astronaut homesick for Earth", "a toaster with stage fright",
    "a broken controller", "the final day of summer", "a ghost inside a radio",
    "a dramatic potato", "late-night city lights", "a friendship surviving distance",
    "an elevator afraid of heights", "a cat becoming a DJ", "forgotten save files",
    "starting over after a bad year", "falling for someone at the wrong time",
    "friends growing apart", "driving nowhere after midnight", "wanting to feel alive again",
    "a city that never answers", "outgrowing an old version of yourself",
    "a summer night that ended too soon", "the courage to finally leave",
    "being homesick in your own room", "a victory nobody noticed", "dancing through a power outage",
    "a spaceship held together with tape", "an arcade machine with one last life",
    "a villain who only wanted a day off", "the moon losing its password",
    "an abandoned shopping mall", "a broken robot trying to sing", "coding at three in the morning",
    "running from a storm", "losing an old friend", "first love on the final train",
    "a lighthouse arguing with fog", "an empty carnival after closing", "a tiny dragon with rent due",
    "nothing deciding to become something", "a cassette remembering its owner",
]
GENRES = [
    "dark electronic", "bright synth-pop", "piano ballad", "chaotic hyperpop", "garage rock",
    "dreamy ambient pop", "playful rap", "cinematic industrial", "lo-fi bedroom pop",
    "power metal", "minimal acoustic folk", "glitch dance", "alternative R&B", "pop-punk",
    "cinematic synthwave", "indie folk", "drum and bass", "dark trap", "new-wave rock",
    "dream pop", "orchestral hip-hop", "disco funk", "post-rock", "electro-swing",
    "doom metal", "cloud rap", "space rock", "trip-hop", "jazz-pop", "chiptune punk",
]
MOODS = ["joyful", "heartbroken", "ominous", "hopeful", "absurd", "nostalgic", "furious", "tender", "uneasy", "triumphant"]
DETAILS = [
    "under violet streetlights", "during a thunderstorm", "inside an empty arcade", "at 3:17 AM",
    "while the neighbors dance", "with one speaker blown out", "beneath a flickering sign",
    "on the last train home", "with a choir of dial-up modems", "as the city loses power",
    "in a room full of old televisions", "while a cheap fan rattles", "through a cracked phone speaker",
    "as a tiny robot keeps time", "without using the word love", "with an abrupt quiet ending",
    "with a chorus built for shouting", "like a transmission from nowhere", "with one terrible rhyme",
    "while rain hits the window", "with warm bass and cold vocals", "as the clock skips a minute",
    "with a chord that refuses to resolve", "while a duck handles percussion",
    "with restrained verses and a huge chorus", "as if the demo file became self-aware",
    "with a hopeful last line", "with a deliberately crooked bridge", "as the room slowly wakes up",
    "through fog that looks suspiciously expensive", "while the floor complains in B minor",
]
TEMPOS = ["slow-burning", "restless mid-tempo", "fast and breathless", "half-time", "danceable", "free-time", "marching", "patiently accelerating"]
RHYTHMS = ["clipped drums", "skipping percussion", "heavy toms", "soft breakbeats", "four-on-the-floor drums", "uneven handclaps", "a lurching bass pulse", "barely audible clicks"]
TEXTURES = ["foggy pads", "rusted guitars", "glasslike synths", "warm tape hiss", "detuned piano", "overdriven bass", "hollow choir layers", "arcade-tone arpeggios", "wide strings", "small acoustic details"]
VOCALS = ["close whispered vocals", "a cracked lead vocal", "stacked bright harmonies", "spoken verses", "a shout-along chorus", "low restrained singing", "call-and-response vocals", "a theatrical lead"]
ARCS = ["a chorus that suddenly opens", "a bridge that drops to near silence", "an ending that dissolves", "a final chorus with extra momentum", "a deliberately awkward final hit", "a gradual climb into noisy relief", "a last verse that turns hopeful", "an outro that leaves one instrument alone"]
TITLE_IMAGES = ["Neon", "Static", "Paper", "Borrowed", "Midnight", "Last", "Glass", "Electric", "Crooked", "Quiet", "Velvet", "Unpaid", "Blue", "Afterimage", "Small"]
TITLE_STOPWORDS = {"the", "and", "with", "from", "into", "after", "before", "yourself", "that", "only", "wanted", "become", "becoming"}

VERSE_PAIRS = [
    ("The street keeps blinking like it knows the plot,", "I carry {focus} past the corner shop."),
    ("Morning left a jacket on the kitchen chair,", "The shape of {focus} is still hanging there."),
    ("A tired little motor clears its throat,", "I write {focus} on a paper boat."),
    ("The dashboard paints my hands electric blue,", "Every empty mile keeps pointing back to {focus}."),
    ("Rain taps a password on the window frame,", "I answer with {focus} and forget my name."),
    ("The elevator sighs between the floors,", "It dreams of {focus} behind the closing doors."),
    ("An old cassette rewinds the afternoon,", "It tangles {focus} with a badly tuned moon."),
    ("The parking lot is holding one last star,", "I tell {focus} we have not gone too far."),
    ("A vending machine glows green across the hall,", "It sells {focus} but refuses to make change at all."),
    ("I found a loose beat hiding in my shoe,", "It sounds like {focus} trying something new."),
    ("The ceiling fan conducts an empty room,", "I let {focus} arrive three measures too soon."),
    ("The city folds its maps into a crane,", "Then sends {focus} circling through the rain."),
    ("A tiny warning flashes out of time,", "I turn {focus} into a crooked rhyme."),
    ("The hallway learns to hum beneath its breath,", "It keeps {focus} awake and bores the ghost to death."),
    ("I left the porch light arguing with dawn,", "It swears that {focus} never really left or moved on."),
    ("The speakers cough, recover, then begin,", "A strange version of {focus} comes wandering in."),
    ("My shadow missed the bus by half a beat,", "So {focus} and I walked home on tired feet."),
    ("The clock puts on a suit and calls a vote,", "I nominate {focus}; the toaster clears its throat."),
    ("One cloud gets stuck above the laundromat,", "It looks like {focus} wearing a paper hat."),
    ("The final train pulls daylight from the track,", "I trade it {focus} and it refuses to give it back."),
]
CHORUS_PAIRS = [
    ("Hold on to {focus} while the whole room sways,", "We can lose the map without losing our way."),
    ("Say {focus} loud enough to shake the roof,", "The night may be ridiculous, but here is living proof."),
    ("Run with {focus}; let the red lights stare,", "If the road ends early, we will build another there."),
    ("I keep {focus} in the rhythm of my chest,", "Not perfectly forever—just honestly our best."),
    ("Turn {focus} up until the windows learn,", "Some things disappear and some return."),
    ("Oh, {focus}, magnificent machine,", "You broke the quiet and painted it green."),
    ("Meet me where {focus} crosses midnight rain,", "We will make a chorus from the beautiful remains."),
    ("Let {focus} spin like a coin in the air,", "Call it bad planning; I am still going there."),
    ("Every beat says {focus}, every echo says stay,", "We are not fixed, but we are louder today."),
    ("Carry {focus} through the static and smoke,", "Turn the worst little moment into one decent joke."),
    ("When {focus} falls, let the bass line rise,", "We will tape up the moon and apologize to the sky."),
    ("I asked {focus} for a reason to move,", "It handed me chaos with an excellent groove."),
]
BRIDGES = [
    "Drop every drum; leave the breathing in.\nThe smallest honest sound is where we begin.",
    "Maybe the answer was never the plan.\nMaybe the rhythm just needed a hand.",
    "The lights lean sideways; the floor disagrees.\nWe dance through both arguments with unreasonable ease.",
    "For four quiet bars, let the machinery rest.\nThen wake every wire in the walls of my chest.",
    "I put my doubt in a shopping cart.\nIt stole one wheel and rolled into the dark.",
    "No grand solution, no cinematic cure—\nJust one stubborn note that is strangely sure.",
    "The room goes weightless, the snare disappears.\nA cheap little melody outlives all the fear.",
    "If this is nonsense, give nonsense a key.\nLet it wear a gold cape and harmonize badly with me.",
    "Half of the story was hiding offscreen.\nThe other half arrived in fluorescent green.",
    "We lower the volume and finally hear\nThe ridiculous truth that was already near.",
]
OUTROS = [
    "The last chord limps home, still proud of the trip.\nNight closes the door with a fingertip.",
    "One small note remains when the bright drums end.\nIt does not solve anything; it stays like a friend.",
    "The streetlight clicks off. The melody stays.\nMorning finds both of us slightly amazed.",
    "We miss the last cue and grin at the floor.\nThe song leaves quietly through the side door.",
    "A final low hum settles under the rain.\nNothing is fixed, but we sing it again.",
    "The tape reaches silence, then offers one squeak.\nA flawless conclusion was never our thing.",
    "The moon clocks out; the small speakers glow.\nI wave at the answer and let the beat go.",
    "The lights become dots, the room becomes dawn.\nThe weird little chorus keeps wandering on.",
    "One cymbal complains, then everyone leaves.\nThe quiet wears rhythm like dust on its sleeves.",
    "No fireworks—only the soft motor's tune.\nWe finish too early and blame it on the moon.",
]


def record(user: str, assistant: str) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def focus(subject: str) -> str:
    return re.sub(r"^(?:my|a|an|the)\s+", "", subject, flags=re.I).rstrip(".,")


def generated_title(subject: str, index: int) -> str:
    words = [word.strip(".,").title() for word in focus(subject).split()
             if len(word.strip(".,")) > 2 and word.strip(".,").lower() not in TITLE_STOPWORDS]
    anchor = words[(index * 3) % len(words)] if words else "Signal"
    forms = [f"{TITLE_IMAGES[index % len(TITLE_IMAGES)]} {anchor}", f"{anchor} at Closing Time",
             f"The {anchor} Department", f"{anchor} With the Lights Out",
             f"A Small History of {anchor}", f"Please Hold for {anchor}", f"{anchor}, Probably"]
    return forms[(index // len(TITLE_IMAGES)) % len(forms)]


def generated_style(genre: str, mood: str, index: int) -> str:
    return (f"{genre}; {TEMPOS[index % len(TEMPOS)]}, {RHYTHMS[(index * 3) % len(RHYTHMS)]}, "
            f"{TEXTURES[(index * 5 + 1) % len(TEXTURES)]}, {VOCALS[(index * 7 + 2) % len(VOCALS)]}, "
            f"{mood} mood, and {ARCS[(index * 11 + 3) % len(ARCS)]}")


def contextual_line(line: str, subject: str, salt: int) -> str:
    subject_focus = focus(subject)
    if "{focus}" in line:
        return line.format(focus=subject_focus)
    lowered = line[0].lower() + line[1:]
    frames = [
        f"Near {subject_focus}, {lowered}",
        f"With {subject_focus} in mind, {lowered}",
        f"After {subject_focus}, {lowered}",
        f"As {subject_focus} drifts past, {lowered}",
        f"Thinking of {subject_focus}, {lowered}",
    ]
    return frames[salt % len(frames)]


def contextual_block(text: str, subject: str, salt: int) -> str:
    return "\n".join(contextual_line(line, subject, salt + offset)
                      for offset, line in enumerate(text.splitlines()))


def stanza(subject: str, index: int, offset: int = 0) -> str:
    pair = VERSE_PAIRS[(index + offset * 7) % len(VERSE_PAIRS)]
    return "\n".join(contextual_line(line, subject, index + offset + line_index)
                      for line_index, line in enumerate(pair))


def chorus(subject: str, index: int) -> str:
    pair = CHORUS_PAIRS[(index * 5 + 2) % len(CHORUS_PAIRS)]
    return "\n".join(contextual_line(line, subject, index + line_index)
                      for line_index, line in enumerate(pair))


def song(subject: str, genre: str, mood: str, index: int, *, include_title: bool,
         include_style: bool, full: bool) -> str:
    parts: list[str] = []
    if include_title:
        parts.append(f"Title: {generated_title(subject, index)}")
    if include_style:
        parts.append(f"Style: {generated_style(genre, mood, index)}")
    if parts:
        parts.append("")
    short = [[("Verse", stanza(subject, index)), ("Chorus", chorus(subject, index))],
             [("Intro", stanza(subject, index)), ("Hook", chorus(subject, index))],
             [("Verse 1", stanza(subject, index)), ("Chorus", chorus(subject, index)),
              ("Outro", contextual_block(OUTROS[index % len(OUTROS)], subject, index + 2))]]
    long = [[("Verse 1", stanza(subject, index)), ("Chorus", chorus(subject, index)),
             ("Verse 2", stanza(subject, index, 1)), ("Bridge", contextual_block(BRIDGES[index % len(BRIDGES)], subject, index)),
             ("Final Chorus", chorus(subject, index)), ("Outro", contextual_block(OUTROS[(index * 3) % len(OUTROS)], subject, index + 2))],
            [("Intro", stanza(subject, index)), ("Verse 1", stanza(subject, index, 1)),
             ("Pre-Chorus", stanza(subject, index, 2)), ("Chorus", chorus(subject, index)),
             ("Verse 2", stanza(subject, index, 3)), ("Outro", contextual_block(OUTROS[(index * 7) % len(OUTROS)], subject, index + 2))],
            [("Verse 1", stanza(subject, index)), ("Chorus", chorus(subject, index)),
             ("Instrumental Break", f"{TEXTURES[index % len(TEXTURES)].title()} circle the main motif."),
             ("Bridge", contextual_block(BRIDGES[(index * 3) % len(BRIDGES)], subject, index)), ("Final Chorus", chorus(subject, index)),
             ("Outro", contextual_block(OUTROS[(index * 9) % len(OUTROS)], subject, index + 2))]]
    # Divide by the curriculum cycle so a request mode does not become locked
    # to one structure merely because both cycles share a divisor.
    sections = (long if full else short)[(index // 15 + index) % 3]
    parts.extend(f"[{name}]\n{text}" for name, text in sections)
    return "\n\n".join(parts).strip()


def build(seed: int = 2401) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for index in range(3_600):
        subject = SUBJECTS[(index * 11 + index // 37) % len(SUBJECTS)]
        genre = GENRES[(index * 7 + index // 51) % len(GENRES)]
        mood = MOODS[(index * 13 + index // 67) % len(MOODS)]
        detail = DETAILS[(index * 17 + index // 29) % len(DETAILS)]
        mode = index % 15
        if mode == 0:
            rows.append(record(f"Write a full {mood} {genre} song about {subject}, {detail}, with a title and style.", song(subject, genre, mood, index, include_title=True, include_style=True, full=True)))
        elif mode == 1:
            rows.append(record(f"Write complete lyrics about {subject}, {detail}. Do not add a title or style.", song(subject, genre, mood, index, include_title=False, include_style=False, full=True)))
        elif mode in (2, 3, 4):
            rows.append(record(f"Make a {mood} song about {subject}, {detail}.", song(subject, genre, mood, index, include_title=False, include_style=False, full=False)))
        elif mode == 5:
            rows.append(record(f"Write only a chorus about {subject}, {detail}.", f"[Chorus]\n{chorus(subject, index)}"))
        elif mode == 6:
            rows.append(record(f"Write one verse about {subject}, {detail}.", f"[Verse]\n{stanza(subject, index)}"))
        elif mode == 7:
            rows.append(record(f"Give me a meaningful song title about {subject}.", generated_title(subject, index)))
        elif mode == 8:
            rows.append(record(f"Generate a production style for a {mood} song about {subject}, {detail}. No lyrics.", f"Style: {generated_style(genre, mood, index)}"))
        elif mode == 9:
            rows.append(record(f"Give me a title and style for a song about {subject}, {detail}.", f"Title: {generated_title(subject, index)}\nStyle: {generated_style(genre, mood, index)}"))
        elif mode == 10:
            rows.append(record(f"Write a funny hook about {subject}, {detail}.", f"[Hook]\n{chorus(subject, index)}"))
        elif mode == 11:
            rows.append(record(f"Rewrite this idea as two coherent lyric lines: {subject}, {detail}.", stanza(subject, index)))
        elif mode == 12:
            rows.append({"messages": [
                {"role": "user", "content": f"I want a {mood} {genre} song about {subject}, {detail}. Give it a title and style."},
                {"role": "assistant", "content": f"Title: {generated_title(subject, index)}\nStyle: {generated_style(genre, mood, index)}"},
                {"role": "user", "content": "Now write the full lyrics and keep those choices."},
                {"role": "assistant", "content": song(subject, genre, mood, index, include_title=True, include_style=True, full=True)},
                {"role": "user", "content": "Make the bridge stranger but keep it connected to the song."},
                {"role": "assistant", "content": f"[Revised Bridge]\n{contextual_block(BRIDGES[(index + 4) % len(BRIDGES)], subject, index)}"}]})
        elif mode == 13:
            rows.append({"messages": [
                {"role": "user", "content": f"Write a short song draft about {subject}."},
                {"role": "assistant", "content": song(subject, genre, mood, index, include_title=False, include_style=False, full=False)},
                {"role": "user", "content": "What title and production style would fit it?"},
                {"role": "assistant", "content": f"Title: {generated_title(subject, index)}\nStyle: {generated_style(genre, mood, index)}"}]})
        else:
            rows.append(record(f"Create an instrumental direction for {subject}, {detail}. Do not write lyrics.", f"Style: {generated_style(genre, mood, index)}. Keep the arrangement focused on {focus(subject)}."))
    unique = {json.dumps(item, sort_keys=True, ensure_ascii=False): item for item in rows}
    rows = list(unique.values())
    if len(rows) < 3_200:
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
