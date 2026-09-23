"""Development comparison: inspect every reply; no headline accuracy claim."""
import argparse
import json
from pathlib import Path
from public_api_server import PublicModelService

CASES = [
    ('regression','Invent a three-word title for a game about ducks.'),
    ('regression','Suggest an original song title about a confused robot.'),
    ('creative','Give a two-word name for a cafe run by owls.'),
    ('creative','Invent a three-word title for a story about turtles.'),
    ('creative','Suggest a song title about a lonely satellite.'),
    ('instructions','Return only the city: Quinn lives in Madrid.'),
    ('instructions','Write violet in uppercase. Only the result.'),
    ('instructions','Sort these numbers from smallest to largest: 8, 3, 6.'),
    ('instructions','Rewrite in past tense: I open the door.'),
    ('instructions','Summarize: Ben forgot his umbrella and got soaked on the way home.'),
    ('assistant','Help me with math'),
    ('assistant','I need help writing a message to my teacher.'),
    ('assistant','Give me three lunch ideas.'),
    ('assistant','Write a short thank-you note to a helpful neighbor.'),
    ('facts','Why does ice float?'),
    ('facts','Explain evaporation in one sentence.'),
    ('facts','What is a verb?'),
    ('facts','What is the difference between weather and climate?'),
    ('code','Give a Python function that doubles a number.'),
    ('code','How do I count the items in a Python list?'),
    ('games','What is Gorilla Tag?'),
    ('games','Describe the Gorilla Tag modding scene.'),
]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    s=PublicModelService(Path(args.checkpoint),'cuda');rows=[]
    for category,q in CASES:
        error=None
        try:
            _,a=s.chat(q,None,max_new_tokens=120,temperature=0.6)
        except RuntimeError as exc:
            a='';error=str(exc)
        r={'category':category,'prompt':q,'reply':a,'assistance':s.last_assistance_reason,
           'error':error,
           'unprompted_gtag':category!='games' and any(t in a.lower() for t in ['another axiom','other gorillas','gorilla tag','multiplayer vr'])}
        rows.append(r);print(json.dumps(r,ensure_ascii=True),flush=True)
    Path(args.output).write_text(json.dumps({'checkpoint':args.checkpoint,'results':rows},indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
