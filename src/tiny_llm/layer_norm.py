import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)

        return (self.weight * x * rms).astype(x.dtype)
