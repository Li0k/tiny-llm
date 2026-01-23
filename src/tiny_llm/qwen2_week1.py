from typing import Any

import mlx.core as mx
from torch import embedding

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
            # projection_q.astype(x.dtype),
            # projection_k.astype(x.dtype),
            # projection_v.astype(x.dtype),
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # MLP(x) = (SiLU(W_gate(x)) ⊙ W_up(x)) W_down
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


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
        multi_head_attention = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta,
        )
        mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )

        self.multi_head_attention = multi_head_attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_layernormed = self.input_layernorm(x)
        multi_head_attentioned = self.multi_head_attention(input_layernormed, mask)
        # residual connection
        r = x + multi_head_attentioned
        post_attention_layernormed = self.post_attention_layernorm(r)
        mlp_out = self.mlp(post_attention_layernormed)
        out = r + mlp_out  # residual connection
        return out


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args

        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        self.embedding = Embedding(
            args.vocab_size,
            args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
        )

        self.blocks = []
        for i in range(args.num_hidden_layers):
            layer = mlx_model.model.layers[i]
            block = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj).astype(mx.float16),
                wk=dequantize_linear(layer.self_attn.k_proj).astype(mx.float16),
                wv=dequantize_linear(layer.self_attn.v_proj).astype(mx.float16),
                wo=dequantize_linear(layer.self_attn.o_proj).astype(mx.float16),
                bq=layer.self_attn.q_proj.bias.astype(mx.float16),
                bk=layer.self_attn.k_proj.bias.astype(mx.float16),
                bv=layer.self_attn.v_proj.bias.astype(mx.float16),
                w_gate=dequantize_linear(layer.mlp.gate_proj).astype(mx.float16),
                w_up=dequantize_linear(layer.mlp.up_proj).astype(mx.float16),
                w_down=dequantize_linear(layer.mlp.down_proj).astype(mx.float16),
                w_input_layernorm=layer.input_layernorm.weight.astype(mx.float16),
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(
                    mx.float16
                ),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.blocks.append(block)

        self.norm = RMSNorm(
            args.hidden_size,
            mlx_model.model.norm.weight.astype(mx.float16),
            args.rms_norm_eps,
        )

        if mlx_model.args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head).astype(mx.float16)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        output = self.embedding(inputs)
        for transformer_block in self.blocks:
            output = transformer_block(output, mask="causal")
        output = self.norm(output)

        if self.w_lm_head is not None:
            return linear(output, self.w_lm_head)
        else:
            return self.embedding.as_linear(output)
