import mlx.core as mx

from tiny_llm.basics import linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight  # (vocab_size, embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # print("input shape:", x.shape)
        # print("embedding_dim:", self.embedding_dim)

        # (1, L) = x.shape
        (B, L) = x.shape
        assert x.ndim == 2

        # This can be done with a simple array index lookup operation for Qwen2.
        output = self.weight[x, :]  # (1, L, embedding_dim)
        # print("output shape:", output.shape)
        return output

    def as_linear(self, x: mx.array) -> mx.array:
        # print("Linear embedding input shape:", x.shape)
        return linear(
            x, self.weight
        )  # from (B, L, embedding_dim) to (B, L, vocab_size)
