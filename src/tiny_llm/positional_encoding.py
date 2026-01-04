import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"

        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        # shape: (head_dim // 2,) freqs = base ** (-2 * i / head_dim)
        i = mx.arange(0, dims, step=2, dtype=mx.float32)
        freqs = mx.power(base, -i / dims)
        pos = mx.arange(seq_len)
        theta = mx.outer(pos, freqs)  # shape: (seq_len, head_dim // 2)

        cos_freqs = mx.cos(theta)  # shape: (seq_len, head_dim // 2)
        sin_freqs = mx.sin(theta)  # shape: (seq_len, head_dim // 2)

        self.cos_freqs = cos_freqs
        self.sin_freqs = sin_freqs
        pass

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        offset_array = self.handle_offset(offset, N, S)

        sin_basis = (
            self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset_array, :]
        )

        cos_basis = (
            self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset_array, :]
        )

        cos_basis = cos_basis.reshape(-1, S, 1, D // 2)
        sin_basis = sin_basis.reshape(-1, S, 1, D // 2)

        if self.traditional:
            return self.traditional_rope(x, cos_basis, sin_basis)
        else:
            return self.efficient_rope(x, cos_basis, sin_basis)

    def handle_offset(
        self,
        offset: list[slice] | slice | None,
        N: int,
        S: int,
    ) -> mx.array | None:
        if offset is None:
            return None
        if isinstance(offset, slice):
            assert offset.stop - offset.start == S, f"offset must be of length {S}"
            return mx.array([list(range(offset.start, offset.stop)) for _ in range(N)])
        if isinstance(offset, list):
            assert len(offset) == N, (
                f"offsets must have the same length as batch size {N}"
            )
            for o in offset:
                assert o.stop - o.start == S, f"offset must be of length {S}"
            return mx.array([list(range(i.start, i.stop)) for i in offset])

    def traditional_rope(self, x: mx.array, cos_basis: mx.array, sin_basis: mx.array):
        N, S, H, D = x.shape
        assert D % 2 == 0, "head_dim must be even for traditional RoPE"
        x = x.reshape(N, S, H, D // 2, 2)

        # print("x shape:", x.shape)
        # print("cos_basis shape:", cos_basis.shape)
        # print("sin_basis shape:", sin_basis.shape)

        x1 = x[..., 0]
        x2 = x[..., 1]

        # cos_basis = cos_basis.reshape(-1, S, 1, D // 2)
        # sin_basis = sin_basis.reshape(-1, S, 1, D // 2)

        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)

        # merge back to original shape
        y = mx.stack([real, imag], axis=-1)
        y = y.reshape(N, S, H, D)
        return y

    def efficient_rope(self, x: mx.array, cos_basis: mx.array, sin_basis: mx.array):
        N, S, H, D = x.shape
        assert D % 2 == 0, "head_dim must be even for efficient RoPE"
        # x = x.reshape(N, S, H, D // 2, 2)

        x1 = x[..., 0 : D // 2]
        x2 = x[..., D // 2 : D]

        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)

        y = mx.concat([real, imag], axis=-1)
        y = y.reshape(N, S, H, D)

        return y
