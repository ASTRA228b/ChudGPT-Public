from tokenizers import Tokenizer

from chudlm.prompts import build_context_token_ids
from chudlm.sft_data import SupervisedConversationDataset


def test_oversized_request_retains_both_ends_and_roles():
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    prompt, ids = build_context_token_ids(
        tokenizer,
        [{"role": "user", "content": "Explain this code: " + "example " * 2000 + " Return only Java."}],
        128,
        system_prompt="You are ChudGPT-Public.",
    )
    assert len(ids) <= 128
    assert "Explain this code:" in prompt
    assert "Return only Java." in prompt
    assert "<user>:" in prompt
    assert prompt.endswith("<assistant>:")


def test_training_keeps_complete_answer_after_long_history():
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    answer = "The robot is called Cedar."
    dataset = SupervisedConversationDataset(
        [[
            {"role": "user", "content": "Earlier discussion " * 600},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "My robot is called Cedar. What is its name?"},
            {"role": "assistant", "content": answer},
        ]], tokenizer, 96, system_prompt="You are ChudGPT-Public.",
    )
    _, targets = dataset[0]
    learned = targets[targets != -100].tolist()
    assert learned == tokenizer.encode(" " + answer).ids + [tokenizer.token_to_id("<eos>")]
