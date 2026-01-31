from math import log
from typing import Callable

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]  # last token logits
        log_probs = logits - mx.logsumexp(
            logits, keepdims=True
        )  # for numerical stability
        if sampler is None:
            y = mx.argmax(log_probs, axis=-1)  # default greedy sampling
        else:
            y = sampler(log_probs)

        return y

    tokens = mx.array(
        tokenizer.encode(prompt, add_special_tokens=False)
    )  # List[int] to mx.array
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while True:
        token = _step(model, tokens)
        tokens = mx.concat([tokens, token])  # Append new token
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
