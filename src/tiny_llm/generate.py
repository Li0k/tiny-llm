from math import log
from typing import Callable

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .kv_cache import TinyKvFullCache
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2


def simple_generate(
    model: Qwen3ModelWeek1,
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
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]  # last token logits
        log_probs = logits - mx.logsumexp(
            logits, keepdims=True
        )  # for numerical stability
        # if sampler is None:
        #     y = mx.argmax(log_probs, axis=-1)  # default greedy sampling
        # else:
        #     y = sampler(log_probs)

        y = mx.argmax(log_probs, axis=-1)

        return y

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    # prefill the cache with the prompt
    token = _step(model, tokens, 0, kv_cache)

    # generate/decode loop
    while True:
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        token = _step(model, token, tokens.size - 1, kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
