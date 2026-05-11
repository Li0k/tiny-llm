# Week 2 Bench Notes

Command:

```bash
PYTHONPATH=src pdm run python bench.py --solution tiny_llm --loader week2 --model qwen2-0.5b
```

## Before Quantized Kernel Integration

Path:
- `mx.dequantize(...)` once during model init
- regular `linear(...)`
- MLX native float matmul

| Metric | Value |
|---|---:|
| Output throughput | 115.68 tok/s |
| Total throughput | 247.90 tok/s |
| Prefill throughput | 3468.10 tok/s |
| Decode throughput | 119.57 tok/s |

## After Quantized Kernel Integration

Path:
- `quantized_matmul`
- `quantized_linear`
- custom CPU/GPU extension kernels

| Metric | Value |
|---|---:|
| Output throughput | 74.83 tok/s |
| Total throughput | 160.35 tok/s |
| Prefill throughput | 259.92 tok/s |
| Decode throughput | 110.88 tok/s |

## Delta

| Metric | Change |
|---|---:|
| Output throughput | -35.3% |
| Total throughput | -35.3% |
| Prefill throughput | -92.5% |
| Decode throughput | -7.3% |

## Why Slower

- The old path used MLX's optimized float matmul after one-time `mx.dequantize(...)`.
- The new path uses our teaching kernel, which is correctness-first, not fully optimized.
- Prefill is hit hardest because it runs many large linear layers.

## Next Comparison

```bash
PYTHONPATH=src pdm run python bench.py --solution tiny_llm --loader week2 --model qwen2-0.5b --enable-flash-attn
```
