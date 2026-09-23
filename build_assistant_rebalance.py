"""Balanced original task examples, reviewed replay, and a small game-retention slice."""
import json
import random
from collections import Counter
from pathlib import Path
from build_public_games_retrain import rejection_reason, GAME_PAIRS

ROOT = Path(__file__).resolve().parent


def build():
    rows = []
    def add(category, question, answer):
        rows.append({'source':'assistant-rebalance-20260923','category':category,
                     'messages':[{'role':'user','content':question},{'role':'assistant','content':answer}]})
    subjects = [('ducks','Duck'),('cats','Cat'),('dogs','Dog'),('robots','Robot'),('pirates','Pirate'),
                ('rabbits','Rabbit'),('frogs','Frog'),('dragons','Dragon'),('astronauts','Astronaut'),
                ('penguins','Penguin'),('bees','Bee'),('dinosaurs','Dinosaur'),('wizards','Wizard'),
                ('foxes','Fox'),('mice','Mouse'),('aliens','Alien'),('snails','Snail'),('bears','Bear'),
                ('ghosts','Ghost'),('knights','Knight')]
    for plural, singular in subjects:
        for kind,ending in [('game','Adventure'),('story','Secret'),('song','Melody'),('book','Journey')]:
            add('creative_title',f'Give a three-word title for a {kind} about {plural}.',f'{singular} Moonlight {ending}')
            add('creative_title',f'Suggest a name for my {kind} about {plural}.',f'{singular} {ending}')
            add('creative_title',f'Invent an original {kind} title about {plural}. Just the title.',f'The {singular} {ending}')
        for mood,title in [('confused','Lost in Thought'),('sleepy','Dreams Before Dawn'),('happy','A Little Sunshine'),('lonely','One Quiet Voice')]:
            add('creative_title',f'Name a song about a {mood} {singular.lower()}.',f'{singular}: {title}')
    for word in ['cloud','tree','pond','river','sun','moon','robot','duck','green','hello','school','book','light','music','garden','friend','pencil','table','orange','star']:
        add('transform',f'Write {word} in uppercase. Only the result.',word.upper())
        add('transform',f'Return {word.upper()} in lowercase.',word)
        add('transform',f'Repeat this word exactly: {word}',word)
        add('transform',f'Put the word {word} in square brackets.',f'[{word}]')
    for name in ['Mara','Alex','Sam','Riley','Noah','Lena','Ari','Jo']:
        for city in ['Lisbon','Paris','Tokyo','Rome','London','Berlin']:
            add('extract',f'Return only the city: {name} lives in {city}.',city)
            add('extract',f'Who lives in {city}? {name} lives in {city}. Answer with the name only.',name)
    for a in range(2,12):
        for b in [2,3,5]:
            add('reasoning',f'I have {a} apples and get {b} more. How many apples do I have?',f'You have {a+b} apples: {a} + {b} = {a+b}.')
            add('reasoning',f'Sort these numbers from smallest to largest: {a+9}, {b}, {a+3}.',', '.join(map(str,sorted([a+9,b,a+3]))))
    pairs = [
        ('Help me with math','Sure. Send me the problem, and I can work through it with you.'),
        ('Can you help me with homework?','Sure. What subject and question are you working on?'),
        ('I need help writing an email','Who is it for, and what do you want to say?'),
        ('Help me debug my code','Paste the code, the error message, and what you expected it to do.'),
        ('What is photosynthesis?', 'Photosynthesis is how plants use sunlight to turn water and carbon dioxide into sugars, releasing oxygen.'),
        ('Why do seasons happen?', "Earth's axis is tilted. As Earth orbits the Sun, the tilt changes the sunlight each hemisphere receives."),
        ('Why does ice float?', 'Ice is less dense than liquid water, so it floats.'),
        ('What is evaporation?', 'Evaporation happens when a liquid changes into a gas at its surface.'),
        ('What is gravity?', 'Gravity is the attraction between objects with mass. It pulls objects toward Earth and keeps planets in orbit.'),
        ('What is a noun?', 'A noun names a person, place, thing, or idea.'),
        ('What is a verb?', 'A verb expresses an action or a state, such as run, think, or be.'),
        ('What is the difference between weather and climate?', 'Weather is the short-term state of the atmosphere. Climate describes patterns over many years.'),
        ('What is the capital of Australia?', 'The capital of Australia is Canberra.'),
        ('Give me three breakfast ideas','1. Oatmeal with fruit\n2. Scrambled eggs on toast\n3. Yogurt with granola'),
        ('List three ways to save water','1. Fix leaks.\n2. Take shorter showers.\n3. Turn off the tap while brushing your teeth.'),
        ('Write a short thank-you note for a gift','Thank you for the thoughtful gift! It was kind of you to think of me.'),
        ('Write a polite reminder about a meeting','Hi, just a reminder about our upcoming meeting. Please let me know if the time no longer works for you.'),
        ('Explain a variable in Python','A variable is a name that refers to a value. For example, score = 10 assigns the value 10 to score.'),
        ('Write a Python function that doubles a number','```python\ndef double(number):\n    return number * 2\n```'),
        ('Write a Python function that adds two numbers','```python\ndef add(a, b):\n    return a + b\n```'),
        ('How do I get the length of a Python list?', 'Use len(items), where items is your list.'),
        ('How should I start learning programming?', 'Pick one language, learn variables and loops, and build small projects. Practice by changing examples and checking what happens.'),
        ('Suggest a simple weekend project','Build a small to-do list: add tasks, mark them complete, and save them to a file.'),
        ('Write one sentence about a rainy day','Rain tapped the window while the street shone under gray clouds.'),
        ('Summarize: Nina missed the bus, so she walked to school.','Nina walked to school after missing the bus.'),
        ('Make this polite: send me the file now','Could you please send me the file when you have a moment?'),
        ('Rewrite in past tense: I walk to school.','I walked to school.'),
        ('Give an example of a metaphor','The classroom was a beehive of activity.'),
    ]
    for q,a in pairs:
        for prefix in ['', 'Please: ', 'Hey, ']: add('assistant',prefix+q,a)
    # Existing reviewed examples retain other everyday skills; game facts appear once.
    rows.extend(json.loads(s) for s in (ROOT/'data/public_games_replay.jsonl').read_text(encoding='utf-8').splitlines())
    for q,a in GAME_PAIRS: add('game_retention',q,a)
    accepted=[]; seen=set(); rejected=Counter()
    for row in rows:
        reason=rejection_reason(row)
        key=json.dumps(row['messages'],sort_keys=True)
        if reason or key in seen:
            rejected[reason or 'duplicate']+=1
            continue
        seen.add(key);accepted.append(row)
    random.Random(20260923).shuffle(accepted)
    (ROOT/'data/public_assistant_rebalance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    report={'records':len(accepted),'categories':dict(Counter(r.get('category','replay') for r in accepted)),'rejected':dict(rejected)}
    (ROOT/'reports/assistant_dataset_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__': build()
