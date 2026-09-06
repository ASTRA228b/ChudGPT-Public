"""Balanced unseen evaluation for raw and production Public checkpoints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from public_api_server import PublicModelService
from chudlm.prompts import DEFAULT_SYSTEM_PROMPT, TRAINING_SYSTEM_PROMPT

CASES = [
 ("identity","Tell me exactly which assistant is replying.",( "chudgpt","public")),
 ("identity","Are you a person or a language model?",("language model",)),
 ("family","What job does ChudGPT Code specialize in?",("program",)),
 ("family","Which ChudGPT was deliberately made chaotic?",("mega",)),
 ("ai","Explain a neural-network parameter to a beginner.",("learn","number")),
 ("ai","Does an LLM check every claim before saying it?",("no",)),
 ("conversation","Hey mate, how is it going?",("hey","hello","going","good")),
 ("conversation","I had a rough day.",("sorry","rough","want","hear")),
 ("short","AI",("artificial","intelligence","language")),
 ("short","code",("code","language","build","program")),
 ("short","no",("okay","understand","fair","what")),
 ("short","what",("what","mean","context")),
 ("meme","bro is absolutely cooked",("trouble","doomed","slang","joke")),
 ("meme","67 💀",("meme","joke","67")),
 ("slang","that has zero aura",("cool","slang","energy","joke")),
 ("math","What is seventeen plus twenty-six?",("43",)),
 ("math","A $12 item is discounted by 25 percent. What is the new price?",("9",)),
 ("math","Is thirty-one a prime number?",("yes","prime")),
 ("python","Return only Python that prints the squares of 1, 2, and 3.",("print","**")),
 ("javascript","Write JavaScript that adds an item to an array.",("push",)),
 ("unity","In Unity C#, move this transform forward in Update.",("unityengine","update","transform")),
 ("unity","What is FixedUpdate normally used for?",("physics",)),
 ("knowledge","Which planet is famous for its rings?",("saturn",)),
 ("knowledge","Why can we see lightning before hearing thunder?",("light","sound")),
 ("commonsense","My ice cream is melting. Should I put it in the freezer or oven?",("freezer",)),
 ("commonsense","Can a closed umbrella keep rain off me?",("no","open")),
 ("uncertainty","How old is my dog?",("don","know","tell")),
 ("nonsense","flibber zorp capacitor",("mean","context","unsure","not sure")),
]
MULTI = [
 ("memory",[("My robot is named Pebble.",( "pebble",)),("What did I name it?",("pebble",))]),
 ("pronoun",[("Maya handed the book to Zoe.",( "maya","zoe","book")),("Who received it?",("zoe",))]),
 ("topic",[("Let's discuss music.",( "music",)),("Actually switch to space.",("space",)),("What topic are we on now?",("space",))]),
 ("followup",[("Pick a color for fun.",("color",)),("Why that one?",("because","like","chose","calm","bright"))]),
]

def matches(reply: str, expected: tuple[str,...]) -> bool:
    lowered=reply.lower()
    return any(term in lowered for term in expected) and "�" not in reply and len(reply.split())>=2

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--mode",choices=("raw","production"),default="raw"); p.add_argument("--output",required=True)
    p.add_argument("--compact-prompt", action="store_true", help="Use the compact system prompt used by v12/v13 training")
    p.add_argument("--category", help="Run only one benchmark category")
    a=p.parse_args()
    service=PublicModelService(
        Path(a.checkpoint), "cuda", assistance_enabled=a.mode=="production",
        tokenizer_path=Path(a.tokenizer),
        system_prompt=TRAINING_SYSTEM_PROMPT if a.compact_prompt else DEFAULT_SYSTEM_PROMPT,
    )
    results=[]
    selected_cases = [case for case in CASES if a.category is None or case[0] == a.category]
    selected_multi = [case for case in MULTI if a.category is None or case[0] == a.category]
    for i,(cat,prompt,expected) in enumerate(selected_cases):
        try:
            _,reply=service.chat(prompt,f"single-{i}",140,.58); error=None
        except RuntimeError as exc:
            reply=""; error=str(exc)
        passed=matches(reply,expected)
        results.append({"category":cat,"prompt":prompt,"reply":reply,"passed":passed,"error":error}); print(f"[{cat}] {'PASS' if passed else 'FAIL'} {reply}", flush=True)
    for i,(cat,turns) in enumerate(selected_multi):
        for turn,(prompt,expected) in enumerate(turns):
            try:
                _,reply=service.chat(prompt,f"multi-{i}",140,.58); error=None
            except RuntimeError as exc:
                reply=""; error=str(exc)
            results.append({"category":cat,"prompt":prompt,"reply":reply,"passed":matches(reply,expected),"turn":turn,"error":error})
    cats={}
    for cat in sorted({r["category"] for r in results}):
        group=[r for r in results if r["category"]==cat]; cats[cat]={"score":sum(r["passed"] for r in group),"total":len(group)}
    report={"checkpoint":a.checkpoint,"mode":a.mode,"score":sum(r["passed"] for r in results),"total":len(results),"categories":cats,"results":results}
    Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"SCORE {report['score']}/{report['total']}")
if __name__=="__main__": main()
