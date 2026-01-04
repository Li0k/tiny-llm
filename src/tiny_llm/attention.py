import mlx.core as mx

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # print shape of query, key, value
    # print("Query shape:", query.shape)
    # print("Key shape:", key.shape)
    # print("Value shape:", value.shape)

    # if scale is None:
    #     scale = 1.0 / (query.shape[-1] ** 0.5)  # Default scaling factor
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale

    key_t = key.swapaxes(-1, -2)  # Transpose the last two dimensions
    scores = mx.matmul(query, key_t) * factor
    if mask is not None:
        scores = scores + mask
    attn_weights = softmax(scores, axis=-1)
    output = mx.matmul(attn_weights, value)
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        assert hidden_size % num_heads == 0
        head_dim = hidden_size // num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = mx.rsqrt(head_dim)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert query.shape == key.shape == value.shape
        N, L, E = query.shape

        assert E == self.hidden_size

        # The functionality to project inputs to Q/K/V has been moved to separate
        def project_inputs(
            input: mx.array,
            w: mx.array,
        ) -> mx.array:
            return (
                linear(input, w)
                .reshape(N, L, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

        projection_q = project_inputs(query, self.wq)

        projection_k = project_inputs(key, self.wk)

        projection_v = project_inputs(value, self.wv)

        output = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(N, L, E)

        return linear(output, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # mask = mx.full((L, S), float("-inf"), dtype=dtype)
    mask = mx.tril(mx.ones((L, S), dtype=dtype), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(float("-inf")))

    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,  # mask: N.. x H_q x L x S
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]

    assert H_q % H == 0, "num_heads of query must be multiple of num_heads of key/value"
    n_repeats = H_q // H

    B = query.shape[:-3]
    extended_shape = query.shape

    query = query.reshape(*B, H, n_repeats, L, D)
    key = key.reshape(*B, H, 1, S, D)
    value = value.reshape(*B, H, 1, S, D)
    key_t = key.swapaxes(-1, -2)
    scores = mx.matmul(query, key_t) * factor

    # print("Scores shape:", scores.shape)

    # print("Mask shape:", mask.shape if mask is not None else None)

    if mask is not None:
        # mask = mx.broadcast_to(mask, (*B, H_q, L, S))
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
            # mask = mask.reshape(*B, H, n_repeats, L, S)
            scores = scores + mask
        else:
            mask = mask.reshape(*B, H, n_repeats, L, S)
            scores = scores + mask
    attn_weights = softmax(scores, axis=-1)
    output = mx.matmul(attn_weights, value)
    return output.reshape(extended_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
