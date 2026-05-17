from typing import Any

import mlx.core as mx
from numpy import block, ma

from .attention import flash_attention, scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        assert hidden_size % num_heads == 0
        head_dim = hidden_size // num_heads
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.scale = mx.rsqrt(head_dim)

        # dequantize the weights
        # self.wq = mx.dequantize(wq.weight, wq.scales, wq.biases, wq.group_size, wq.bits)
        # self.wk = mx.dequantize(wk.weight, wk.scales, wk.biases, wk.group_size, wk.bits)
        # self.wv = mx.dequantize(wv.weight, wv.scales, wv.biases, wv.group_size, wv.bits)
        # self.wo = mx.dequantize(wo.weight, wo.scales, wo.biases, wo.group_size, wo.bits)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.bq = bq
        self.bk = bk
        self.bv = bv

        # RoPE
        self.rope = RoPE(head_dim, max_seq_len, theta, traditional=False)

        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        H_q, H, D = self.num_heads, self.num_kv_heads, self.head_dim

        # project
        # q = linear(x, self.wq, self.bq).reshape(B, L, H_q, D)
        # k = linear(x, self.wk, self.bk).reshape(B, L, H, D)
        # v = linear(x, self.wv, self.bv).reshape(B, L, H, D)

        q = quantized_linear(x, self.wq, self.bq).reshape(B, L, H_q, D)
        k = quantized_linear(x, self.wk, self.bk).reshape(B, L, H, D)
        v = quantized_linear(x, self.wv, self.bv).reshape(B, L, H, D)

        # apply RoPE
        # offset = offsets[0]
        if isinstance(offsets, int):
            offset_slice = [slice(int(offsets), int(offsets + L))]
        else:
            offset_slice = [slice(int(i), int(i + L)) for i in offsets]
        q = self.rope(q, offset=offset_slice)
        k = self.rope(k, offset=offset_slice)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        k, v, seq_len, _ = cache.update_and_fetch(k, v)

        if self.use_flash_attention:
            x = flash_attention(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=self.scale,
                mask=mask,
            ).astype(x.dtype)
        else:
            x = scaled_dot_product_attention_grouped(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=self.scale,
                mask=mask,
            ).astype(x.dtype)

        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return quantized_linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim

        # dequantize the weights
        # self.w_gate = mx.dequantize(
        #     w_gate.weight, w_gate.scales, w_gate.biases, w_gate.group_size, w_gate.bits
        # )

        # self.w_up = mx.dequantize(
        #     w_up.weight, w_up.scales, w_up.biases, w_up.group_size, w_up.bits
        # )

        # self.w_down = mx.dequantize(
        #     w_down.weight, w_down.scales, w_down.biases, w_down.group_size, w_down.bits
        # )

        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)
        return quantized_linear(
            silu(quantized_linear(x, self.w_gate)) * quantized_linear(x, self.w_up),
            self.w_down,
        )


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.multi_head_attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, weight=w_input_layernorm
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, weight=w_post_attention_layernorm
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_normed = self.input_layernorm(x)
        attention_out = self.multi_head_attention(input_normed, [offset], cache, mask)
        post_attention = x + attention_out  # residual connection after attention
        post_attention_normed = self.post_attention_layernorm(post_attention)
        mlp_out = self.mlp(post_attention_normed)
        return mlp_out + post_attention  # residual connection after MLP


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        # args to initialize the model
        args = mlx_model.args
        self.num_hidden_layers = args.num_hidden_layers
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        # self.max_seq_len = args.max_seq_len
        # self.theta = args.theta
        # self.enable_flash_attn = enable_flash_attn

        self.embeddings = Embedding(
            self.vocab_size,
            self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
        )

        self.blocks = []
        for i in range(self.num_hidden_layers):
            layer = mlx_model.model.layers[i]
            block = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=QuantizedWeights.from_mlx_layer(layer.self_attn.q_proj),
                wk=QuantizedWeights.from_mlx_layer(layer.self_attn.k_proj),
                wv=QuantizedWeights.from_mlx_layer(layer.self_attn.v_proj),
                wo=QuantizedWeights.from_mlx_layer(layer.self_attn.o_proj),
                bq=layer.self_attn.q_proj.bias.astype(mx.float16),
                bk=layer.self_attn.k_proj.bias.astype(mx.float16),
                bv=layer.self_attn.v_proj.bias.astype(mx.float16),
                w_gate=QuantizedWeights.from_mlx_layer(layer.mlp.gate_proj),
                w_up=QuantizedWeights.from_mlx_layer(layer.mlp.up_proj),
                w_down=QuantizedWeights.from_mlx_layer(layer.mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight.astype(mx.float16),
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(
                    mx.float16
                ),
                max_seq_len=args.max_position_embeddings,
                theta=args.rope_theta,
                use_flash_attention=enable_flash_attn,
            )
            self.blocks.append(block)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(mx.float16),
            eps=mlx_model.args.rms_norm_eps,
        )

        if args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head).astype(mx.float16)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        # to embedding
        x = self.embeddings(inputs)

        for i, block in enumerate(self.blocks):
            x = block(x, offset, cache[i], mask="causal")

        x = self.norm(x)
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        else:
            return self.embeddings.as_linear(x)
