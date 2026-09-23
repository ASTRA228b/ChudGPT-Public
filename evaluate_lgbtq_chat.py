"""Inspect neural responses in both serving contexts; does not judge accuracy."""
import argparse
import json
from pathlib import Path
from public_api_server import PublicModelService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    service = PublicModelService(Path('checkpoints/public_games_v2/step_800.pt'), 'cuda')
    service.lgbtq_service = PublicModelService(Path(args.checkpoint), 'cuda')
    rows = []
    for mode in ['default', 'discord']:
        for question in ["I'm trans", 'im gay', 'Im a femboy', 'are you not gay',
                         'i am bisexual actually', 'I think I might be trans',
                         'btw im nonbinary', 'does being a femboy make me gay?']:
            _, answer = service.chat(question, None, max_new_tokens=90, context_mode=mode)
            row = {'mode':mode, 'prompt':question, 'reply':answer,
                   'profile':service.last_neural_profile, 'assistance':service.last_assistance_reason}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
