"""Bounded neural function selection for the public browser simulator."""
import json
import torch
from pydantic import BaseModel, Field

class SimRequest(BaseModel):
    observation: str = Field(min_length=1, max_length=3000)

def select_sim_action(model, tokenizer, device, observation, actor):
    locations = ('Kitchen', 'Bedroom', 'Office', 'Park', 'Bathroom', 'Living room', 'Shop')
    actions = ('eat', 'sleep', 'work', 'relax', 'talk', 'walk', 'wash', 'shop', 'wait',
               'drink', 'cook', 'read', 'exercise', 'garden', 'play_music', 'look', 'listen')
    calls = ([{'name': 'move_to', 'arguments': {'location': p}} for p in locations]
             + [{'name': a, 'arguments': {}} for a in actions]
             + [{'name': 'talk_to', 'arguments': {'person': p}} for p in
                ('Maya', 'Theo', 'ChudGPT-Buggy' if actor == 'public' else 'Public')])
    prefix = tokenizer.encode(observation + '\nfunction_call:').ids[-min(600, model.config.context_length-100):]
    scores = []
    with torch.inference_mode():
        for call in calls:
            suffix = tokenizer.encode(json.dumps(call, separators=(',', ':'))).ids
            logits, _ = model(torch.tensor([prefix + suffix], device=device))
            probs = logits[0, len(prefix)-1:-1].log_softmax(-1)
            targets = torch.tensor(suffix, device=device)
            scores.append(probs.gather(1, targets[:, None]).mean().item())
        distribution = torch.softmax(torch.tensor(scores) / 0.8, dim=0)
        chosen = int(torch.multinomial(distribution, 1).item())
    return {'call': calls[chosen], 'model': actor, 'method': 'constrained_neural_selection',
            'distribution': [{'call': call, 'probability': round(float(p), 5)} for call, p in zip(calls, distribution)]}
