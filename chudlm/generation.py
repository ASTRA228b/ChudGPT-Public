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
    min_new_tokens: int = 0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative; use zero for greedy decoding")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens must be between zero and max_new_tokens")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")
    for generated_count in range(max_new_tokens):
        context = token_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        # Song generation can require a meaningful minimum draft length.
        # Suppressing EOS changes only when sampling stops; it does not insert
        # template text or supply any answer tokens.
        if eos_token_id is not None and generated_count < min_new_tokens:
            next_logits[:, eos_token_id] = -float("inf")
        if no_repeat_ngram_size > 0:
            for batch_index in range(token_ids.size(0)):
                sequence = token_ids[batch_index].tolist()
                prefix_size = no_repeat_ngram_size - 1
                if len(sequence) < prefix_size or prefix_size == 0:
                    continue
                prefix = sequence[-prefix_size:]
                for start in range(len(sequence) - no_repeat_ngram_size + 1):
                    if sequence[start:start + prefix_size] == prefix:
                        next_logits[batch_index, sequence[start + prefix_size]] = -float("inf")
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
