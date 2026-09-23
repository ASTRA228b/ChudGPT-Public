"""Broader task bindings and diverse answer forms; no runtime response lookup."""
import json
import random
from pathlib import Path
from collections import Counter
from build_public_games_retrain import rejection_reason

ROOT=Path(__file__).resolve().parent

def build():
    rng=random.Random(49271)
    rows=[json.loads(s) for s in (ROOT/'data/public_assistant_rebalance.jsonl').read_text(encoding='utf-8').splitlines()]
    def add(cat,q,a):
        rows.append({'source':'assistant-transfer-20260923','category':cat,'messages':[{'role':'user','content':q},{'role':'assistant','content':a}]})
    words='apple banana cherry peach lemon melon grape cloud river mountain ocean forest meadow island desert canyon village castle window door bottle basket blanket pillow notebook pencil paper carpet jacket button kettle mirror candle ribbon rocket planet comet asteroid spaceship camera guitar violin piano trumpet bell dolphin whale shark eagle sparrow falcon parrot hamster horse sheep goat cow donkey otter seal badger zebra tiger lion giraffe monkey spider beetle butterfly ant moth orange yellow purple brown silver golden black white red blue indigo pink gray happy sad quiet loud warm cold bright dark gentle brave clever curious sleepy hungry small large round smooth rough robot duck frog bear fox rabbit pirate astronaut wizard knight'.split()
    names='Mira Luca Eva Finn Theo Ella Maya Owen Zoe Max Leo Iris Nora Eli Leah Ada Isaac Oscar Hazel Ruby Miles Ivy Axel Lyra Arlo Mina Kira Hugo Esme Remy'.split()
    cities=['Athens','Dublin','Oslo','Helsinki','Vienna','Prague','Warsaw','Budapest','Brussels','Amsterdam','Cairo','Nairobi','Lagos','Lima','Quito','Bogota','Santiago','Ottawa','Toronto','Montreal','Boston','Seattle','Chicago','Austin','Denver','Miami','Sydney','Perth','Adelaide','Melbourne','Seoul','Beijing','Shanghai','Bangkok','Jakarta','Manila','Hanoi','Mumbai','Delhi','Kathmandu','Stockholm','Copenhagen','Zurich','Bern','Lisbon','Paris','Tokyo','Rome','London','Berlin']
    for word in words:
        for q,a in [(f'Uppercase: {word}',word.upper()),(f'Convert {word} to uppercase.',word.upper()),
                    (f'Lowercase: {word.upper()}',word),(f'Return only this word: {word}',word),
                    (f'Put quotes around {word}.',f'"{word}"')]:add('text_operations',q,a)
    for _ in range(350):
        name=rng.choice(names);city=rng.choice(cities)
        add('copy_context',f'Extract the city from this sentence and return only its name: {name} lives in {city}.',city)
        add('copy_context',f'{name} lives in {city}. Which city is mentioned? Answer with one city.',city)
        word=rng.choice(words);number=rng.randint(10,999)
        add('copy_context',f'The label is {word}-{number}. Repeat the label exactly.',f'{word}-{number}')
    verbs=[('play','played'),('dance','danced'),('jump','jumped'),('smile','smiled'),('look','looked'),('wait','waited'),('call','called'),('cook','cooked'),('talk','talked'),('work','worked'),('clean','cleaned'),('paint','painted'),('visit','visited'),('listen','listened'),('walk','walked'),('learn','learned'),('laugh','laughed'),('watch','watched'),('help','helped'),('start','started')]
    for pronoun in ['I','We','They','You']:
        for verb,past in verbs:
            for ending in ['every day','after lunch','at home']:
                add('rewrite',f'Change this to past tense: {pronoun} {verb} {ending}.',f'{pronoun} {past} {ending}.')
    for name in names:
        for cause,effect in [('missed the train','took a taxi'),('was tired','went to bed early'),('lost a key','called a locksmith'),('felt hungry','made a sandwich'),('finished the work','went for a walk'),('had a flat tire','repaired the bicycle')]:
            add('summary',f'Summarize in one sentence: {name} {cause}. Because of that, {name} {effect}.',f'{name} {effect} after {name} {cause}.')
    nouns='Dolphin Whale Shark Eagle Sparrow Falcon Parrot Hamster Horse Sheep Goat Otter Seal Badger Zebra Tiger Lion Giraffe Spider Beetle Butterfly Ant Moth Rocket Planet Comet Spaceship Camera Guitar Piano Bell Robot Duck Frog Bear Fox Rabbit Pirate Astronaut Wizard Knight'.split()
    moods=[('confused','Lost'),('lonely','Quiet'),('happy','Joyful'),('sleepy','Dreaming'),('curious','Wandering'),('brave','Fearless'),('sad','Rainy'),('excited','Electric')]
    for noun in nouns:
        for mood,adj in moods:
            add('creative_transfer',f'Suggest an original song name about a {mood} {noun.lower()}.',f'The {adj} {noun}')
        for kind,ending in [('game','Quest'),('book','Tale'),('song','Song')]:
            add('creative_transfer',f'Create a two-word title for a {kind} about a {noun.lower()}.',f'{noun} {ending}')
            add('creative_transfer',f'Invent a three-word name for a {kind} featuring a {noun.lower()}.',f'The {noun} {ending}')
    for a in range(1,20):
        b=rng.randrange(20,40);c=rng.randrange(40,80)
        for values in [[c,a,b],[b,c,a]]:
            add('sort',f'Put these numbers in ascending order: {", ".join(map(str,values))}.',f'{a}, {b}, {c}')
    accepted=[];seen=set()
    for row in rows:
        key=json.dumps(row['messages'],sort_keys=True)
        if key not in seen and not rejection_reason(row):seen.add(key);accepted.append(row)
    rng.shuffle(accepted)
    (ROOT/'data/public_assistant_transfer.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    report={'records':len(accepted),'categories':dict(Counter(r.get('category','replay') for r in accepted))}
    (ROOT/'reports/assistant_transfer_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':build()
