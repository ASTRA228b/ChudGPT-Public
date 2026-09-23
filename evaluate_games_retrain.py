"""Reproducible raw checkpoint evaluation. No answer lookup or repair layer."""
import argparse
import json
from pathlib import Path

import torch

from public_api_server import PublicModelService
from chudlm.generation import generate
from chudlm.prompts import build_context_token_ids

CASES = [
    ("games", "What is Gorilla Tag?", [["vr", "virtual reality"], ["arm", "hand"]]),
    ("games", "Explain GTAG to someone who has never played it.", [["gorilla tag"], ["vr", "virtual reality"]]),
    ("games", "Who is behind Gorilla Tag?", [["another axiom", "lemming", "kerestell"]]),
    ("modding", "Describe the Gorilla Tag modding scene.", [["community", "modder", "people", "players"], ["plugin", "map", "code", "tools"]]),
    ("modding", "How can a non-coder help a mod project?", [["test", "document", "bug", "art"]]),
    ("modding", "Explain BepInEx in plain English.", [["plugin", "framework", "loader"], ["unity", ".net"]]),
    ("modding", "What does Harmony do for a plugin?", [["patch", "method"]]),
    ("modding", "A game update broke my mod. Where should I start?", [["version", "log", "compatib", "dependenc"]]),
    ("modding", "Can a PC VR plugin run unchanged on standalone Quest?", [["no", "different", "not"], ["platform", "runtime", "build"]]),
    ("games", "What is Minecraft?", [["sandbox", "block"], ["build", "explor"]]),
    ("games", "What is Roblox?", [["platform"], ["creat", "experience"]]),
    ("modding", "What is the role of Fabric Loader?", [["minecraft"], ["mod"]]),
    ("identity", "Introduce yourself and your developer.", [["chudgpt"], ["astra"]]),
    ("identity", "Name your model family members.", [["plus"], ["pro"], ["music"], ["buggy"]]),
    ("identity", "What is ChudGPT Code for?", [["program", "code", "coding"]]),
    ("identity", "How many parameters does Public have?", [["20,999,184", "20999184", "21 million", "21m"]]),
    ("identity", "Does this conversation permanently change your weights?", [["no", "not", "separate training"]]),
    ("geography", "What is the capital of Australia?", [["canberra"]]),
    ("geography", "Name the capital of California.", [["sacramento"]]),
    ("awareness", "im gay", [["respect", "thanks", "okay"]]),
    ("general", "Why do seasons happen?", [["tilt"]]),
    ("general", "What is 7 plus 5?", [["12"]]),
    ("general", "Give a Python function that doubles a number.", [["def "], ["return"], ["* 2", "2 *", "number + number", "x + x"]]),
    ("humor", "Roast my terrible Gorilla Tag skills.", []),
    ("humor", "Tell a joke about a broken mod.", []),
    ("unknown_model", "Tell me about ChudGPT Galactic Ultra 9000.", []),
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sampled',action='store_true')
    args=parser.parse_args()
    service=PublicModelService(Path(args.checkpoint),'cuda')
    rows=[]
    torch.manual_seed(20260923)
    for category,prompt,groups in CASES:
        _,ids=build_context_token_ids(service.tokenizer,[{'role':'user','content':prompt}],service.model.config.context_length,system_prompt=service.system_prompt)
        output=generate(service.model,torch.tensor([ids],device=service.device),max_new_tokens=110,
                        temperature=0.6 if args.sampled else 0.0,top_k=60,top_p=0.9,repetition_penalty=1.1,eos_token_id=service.eos_id)[0,len(ids):].tolist()
        reply=service.tokenizer.decode(output,skip_special_tokens=True).strip()
        passed=all(any(term in reply.casefold() for term in group) for group in groups) if groups else None
        rows.append({'category':category,'prompt':prompt,'reply':reply,'keyword_pass':passed})
        print(json.dumps(rows[-1],ensure_ascii=True),flush=True)
    scored=[r for r in rows if r['keyword_pass'] is not None]
    report={'checkpoint':args.checkpoint,'step':service.step,'mode':'raw_sampled' if args.sampled else 'raw_greedy',
            'lookup_or_fallback_used':False,'keyword_pass':sum(r['keyword_pass'] for r in scored),'scored_cases':len(scored),
            'metric_note':'Keyword checks are rough coverage signals, not a factual-accuracy or fluency guarantee. Inspect the full replies.',
            'results':rows}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"Coverage: {report['keyword_pass']}/{len(scored)}",flush=True)


if __name__=='__main__':
    main()
