from __future__ import annotations

import torch

from .model import TransformerLM


@torch.inference_mode()
def generate(
    model: TransformerLM,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative; use zero for greedy decoding")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        if repetition_penalty > 1.0:
            for batch_index in range(token_ids.size(0)):
                used_tokens = torch.unique(token_ids[batch_index])
                scores = next_logits[batch_index, used_tokens]
                next_logits[batch_index, used_tokens] = torch.where(
                    scores < 0, scores * repetition_penalty, scores / repetition_penalty
                )
        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            continue
        next_logits = next_logits / temperature
        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < values[:, [-1]]] = -float("inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = -float("inf")
            next_logits = torch.full_like(next_logits, -float("inf"))
            next_logits.scatter_(1, sorted_indices, sorted_logits)
        next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
        token_ids = torch.cat((token_ids, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return token_ids
