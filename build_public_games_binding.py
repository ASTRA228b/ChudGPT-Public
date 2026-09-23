"""Add natural phrasings for already-reviewed facts, without reading evaluation answers."""
import json
from pathlib import Path

from build_public_games_retrain import GAME_PAIRS, rejection_reason, row
from project_facts import FAMILY_FACTS, FAMILY_SUMMARY

ROOT=Path(__file__).resolve().parent


def main():
    rows=[json.loads(s) for s in (ROOT/'data/public_games_focus.jsonl').read_text(encoding='utf-8').splitlines()]
    for question, answer in GAME_PAIRS:
        if question.startswith('What is ') and question.endswith('?'):
            subject=question[8:-1]
            for prompt in (f"Explain {subject}.", f"Describe {subject}.", f"Tell me about {subject}.", f"What's {subject}?", f"Can you explain {subject} simply?"):
                rows.append(row('game_phrasing',prompt,answer))
    identity="I'm ChudGPT-Public V20, a small experimental AI language model created by Astra. I can chat about games, modding, code, and general topics, and I can still make mistakes or say something ridiculous."
    for prompt in ("Who are you?", "Tell me about yourself.", "Who are you and who made you?", "What kind of AI are you?", "What is your model name?", "Give me an introduction to ChudGPT-Public.", "Can you introduce yourself?", "Who created ChudGPT-Public?", "What are you called?", "Describe yourself."):
        rows.append(row('identity_phrasing',prompt,identity))
    for prompt in ("What models are in the ChudGPT family?", "Tell me which ChudGPT models exist.", "List your sibling models.", "What are the different ChudGPT versions?", "Which models belong to your family?", "What models does Astra have?", "Describe the whole ChudGPT family.", "Tell me about your model family.", "What are your sibling models called?", "Give me the ChudGPT model lineup."):
        rows.append(row('identity_phrasing',prompt,FAMILY_SUMMARY))
    for prompt in ("How many learned weights are in ChudGPT-Public?", "What is your parameter count?", "Tell me Public's model size.", "How large is ChudGPT-Public?", "How many parameters do you have?"):
        rows.append(row('identity_phrasing',prompt,"ChudGPT-Public has 20,999,184 learned parameters, about 21 million. Its transformer has nine layers and a 1,024-token model context window."))
    for name,answer in FAMILY_FACTS.items():
        for prompt in (f"Explain ChudGPT {name}.",f"Tell me about the {name} model.",f"Describe ChudGPT {name}."):
            rows.append(row('identity_phrasing',prompt,answer))
    lookup=dict(GAME_PAIRS)
    for prompt in ("What kind of game is GTAG?", "Tell me about GTAG in simple words.", "Can you describe GTAG?", "What's GTAG about?", "GTAG means what?", "Explain Gorilla Tag for a beginner."):
        rows.append(row('game_phrasing',prompt,lookup['What is Gorilla Tag?']))
    for prompt in ("What is the community around Gorilla Tag mods like?", "Tell me about the Gorilla Tag mod scene.", "What do people make in the Gorilla Tag modding scene?", "What goes on in Gorilla Tag modding groups?"):
        rows.append(row('game_phrasing',prompt,lookup['What is the Gorilla Tag modding community?']))
    seen=set(); accepted=[]
    for item in rows:
        assert rejection_reason(item) is None
        key=json.dumps(item['messages'],sort_keys=True,ensure_ascii=False)
        if key not in seen:
            seen.add(key);accepted.append(item)
    (ROOT/'data/public_games_binding.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    print('Binding examples:',len(accepted))


if __name__=='__main__':main()
