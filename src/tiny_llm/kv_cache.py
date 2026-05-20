from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx

from .attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, L, D = keys.shape
        assert keys.shape == values.shape
        assert B == self.max_active_requests
        if mask_length is None:
            mask_length = L

        data = []
        for b in range(B):
            kv_cache = self.kv_caches[b]
            if kv_cache is None:
                data.append(None)
                continue
            else:
                key = keys[b : b + 1]
                value = values[b : b + 1]

                new_key, new_value, seq_len, _ = kv_cache.update_and_fetch(key, value)
                data.append((new_key[0], new_value[0], seq_len))

        # Smax
        seq_len = max((item[2] for item in data if item is not None), default=0)

        # output

        batched_keys = mx.zeros((B, H, seq_len, D), dtype=keys.dtype)
        batched_values = mx.zeros((B, H, seq_len, D), dtype=values.dtype)
        batch_mask = mx.full((B, mask_length, seq_len), -mx.inf, dtype=keys.dtype)

        for b, item in enumerate(data):
            if item is None:
                continue

            key, value, request_seq_len = item
            start = seq_len - request_seq_len

            batched_keys[b, :, start:seq_len, :] = key
            batched_values[b, :, start:seq_len, :] = value
            batch_mask[b, :, start:seq_len] = causal_mask(
                mask_length, request_seq_len, dtype=keys.dtype
            )

        return (
            batched_keys,
            batched_values,
            None,
            batch_mask.reshape(B, 1, mask_length, seq_len),
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key_values is None:
            # prefill stage
            assert self.offset == 0
            self.key_values = (key, value)
            B, H, S, D = key.shape
            self.offset = S
            return key, value, key.shape[2], mask
        else:
            # decoding stage
            B, H, S, D = key.shape
            old_key, old_value = self.key_values

            self.key_values = (
                mx.concat([old_key, key], axis=2),
                mx.concat([old_value, value], axis=2),
            )

            self.offset += S
            key, value = self.key_values
            return key, value, self.offset, mask
