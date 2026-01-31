import copy

import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        logprobs = copy.copy(logprobs)  # TODO: do we really need a copy?

        if top_k is not None and top_k > 0:
            # negative logprobs for sorting in order
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[
                :, top_k:
            ]
            logprobs[:, mask_elements] = -mx.inf

        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(
                mx.exp(sorted_logprobs), axis=-1
            )  # exp to get true probs

            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True  # always keep the first token
            logprobs[:, sorted_idx] = mx.where(
                mask_elements, sorted_logprobs, -mx.inf
            )  # indexing to original positions

        logprobs = logprobs / temp  # scale by temperature
        return mx.random.categorical(logprobs, axis=-1)

    return sample
