from typing import Any

import mlx.core as mx

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        head_dim = hidden_size // num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        # key_value_head_dim = hidden_size // num_kv_heads
        rope = RoPE(head_dim, max_seq_len, theta, traditional=False)

        # self.query_head_dim = query_head_dim
        # self.key_value_head_dim = key_value_head_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = rope
        self.scale = mx.rsqrt(head_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        # assert query.shape == key.shape == value.shape
        projection_q = (
            linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
            # .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
            # .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)
            # .transpose(0, 2, 1, 3)
        )

        projection_q = self.rope(projection_q, offset=slice(0, L))
        projection_k = self.rope(projection_k, offset=slice(0, L))

        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_grouped(
            projection_q.astype(x.dtype),
            projection_k.astype(x.dtype),
            projection_v.astype(x.dtype),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)  # merge heads back

        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
