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

## After Flash Attention Integration

Baseline command:

```bash
PYTHONPATH=src pdm run python bench.py --solution tiny_llm --loader week2 --model qwen2-0.5b
```

Flash Attention command:

```bash
PYTHONPATH=src pdm run python bench.py --solution tiny_llm --loader week2 --model qwen2-0.5b --enable-flash-attn
```

| Metric | Without Flash Attention | With Flash Attention | Change |
|---|---:|---:|---:|
| Output throughput | 78.03 tok/s | 61.31 tok/s | -21.4% |
| Total throughput | 167.21 tok/s | 131.38 tok/s | -21.4% |
| Prefill throughput | 268.46 tok/s | 253.80 tok/s | -5.5% |
| Decode throughput | 116.18 tok/s | 84.19 tok/s | -27.5% |

Notes:

- The custom Flash Attention kernel is correctness-first and not optimized.
- Decode is slower because the teaching kernel overhead dominates at small `L=1` decode steps.
- Prefill is closer because larger query blocks better amortize kernel overhead.

## Flash Attention Decode Tile Experiment

Change:

```cpp
const int Br = L == 1 ? 1 : 32;
```

This keeps the prefill path unchanged, but uses one query row per threadgroup during decode.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Output throughput | 61.31 tok/s | 62.79 tok/s | +2.4% |
| Total throughput | 131.38 tok/s | 134.56 tok/s | +2.4% |
| Prefill throughput | 253.80 tok/s | 257.05 tok/s | +1.3% |
| Decode throughput | 84.19 tok/s | 86.60 tok/s | +2.9% |

Notes:

- This reduces wasted decode threads because `L == 1` only has one valid query row.
- The gain is modest because the kernel still scans all K/V tiles serially inside one threadgroup.
- A larger decode improvement would require a separate decode attention kernel, likely with split-S or paged attention.

## Quantized Matmul Int4 Unpack Experiment

Change:

```metal
// Instead of looping over 8 int4 values packed in one uint32,
// manually unroll the 8 unpack + dequantize + multiply-add steps.
```

Microbenchmark on real Qwen2-0.5B weights:

| Layer | M | Ours Before | Ours After | Ref |
|---|---:|---:|---:|---:|
| q_proj | 1 | 0.241 ms | 0.232 ms | 0.331 ms |
| q_proj | 128 | 1.827 ms | 0.887 ms | 0.867 ms |
| q_proj | 256 | 2.178 ms | 1.129 ms | 1.115 ms |
| gate_proj | 1 | 0.179 ms | 0.163 ms | 0.422 ms |
| gate_proj | 128 | 5.666 ms | 2.117 ms | 2.077 ms |
| gate_proj | 256 | 11.002 ms | 3.441 ms | 3.444 ms |
| embed_as_linear | 1 | 0.569 ms | 0.557 ms | 5.466 ms |
| embed_as_linear | 128 | 168.210 ms | 49.785 ms | 49.795 ms |
| embed_as_linear | 256 | 338.802 ms | 102.871 ms | 102.540 ms |

Default benchmark after unrolling, without Flash Attention:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Output throughput | 78.03 tok/s | 99.09 tok/s | +27.0% |
| Total throughput | 167.21 tok/s | 212.35 tok/s | +27.0% |
| Prefill throughput | 268.46 tok/s | 803.97 tok/s | +199.5% |
| Decode throughput | 116.18 tok/s | 114.68 tok/s | -1.3% |

Default benchmark after unrolling, with Flash Attention and `Br = L == 1 ? 1 : 32`:

| Metric | Before quant unroll | After quant unroll | Change |
|---|---:|---:|---:|
| Output throughput | 62.79 tok/s | 77.62 tok/s | +23.6% |
| Total throughput | 134.56 tok/s | 166.33 tok/s | +23.6% |
| Prefill throughput | 257.05 tok/s | 757.98 tok/s | +194.9% |
| Decode throughput | 86.60 tok/s | 87.39 tok/s | +0.9% |

Notes:

- The main previous prefill bottleneck was quantized matmul int4 unpack, not Flash Attention.
- Manual unrolling makes large-M prefill kernels roughly match the reference implementation.
- M=1 decode was already relatively strong, so decode throughput barely changes.
- Flash Attention still underperforms regular attention in the default decode-heavy benchmark.

## Quantized Matmul Dispatch Experiment

Temporary change:

```cpp
// Original ours:
grid_dims = MTL::Size(M, K, 1);
compute_encoder.dispatch_threads(grid_dims, group_dims);

// Temporary ref-style dispatch:
grid_dims = MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1);
compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
```

Simple example for decode:

```text
decode has one input token
=> quantized matmul input A is [M, N]
=> M = 1
```

Assume:

```text
M = 1
K = 64
x_size = 32
y_size = 8
```

Original ours with `dispatch_threads`:

```text
logical grid = M x K = 1 x 64
valid output cells = 64
```

Ref-style `dispatch_threadgroups`:

```text
threadgroups = ceil(1 / 32) x ceil(64 / 8) = 1 x 8
threads per group = 32 x 8
physical threads = 1 * 8 * 32 * 8 = 2048
valid output cells = 1 * 64 = 64
```

So for each K block, only row `i=0` is useful:

```text
i = 0       valid
i = 1..31   invalid because M = 1
```

This means the ref-style dispatch wastes most of the M dimension work during decode.

Microbenchmark after temporarily switching ours to ref-style dispatch:

| Layer | M | Ours Original | Ours Ref-Style Dispatch | Ref |
|---|---:|---:|---:|---:|
| q_proj | 1 | 0.232 ms | 0.427 ms | 0.381 ms |
| q_proj | 128 | 0.887 ms | 0.803 ms | 0.674 ms |
| gate_proj | 1 | 0.163 ms | 0.377 ms | 0.338 ms |
| gate_proj | 128 | 2.117 ms | 1.950 ms | 2.004 ms |
| embed_as_linear | 1 | 0.557 ms | 5.886 ms | 5.852 ms |
| embed_as_linear | 128 | 49.785 ms | 52.842 ms | 52.798 ms |

Notes:

- The temporary ref-style dispatch made our `M=1` decode-shaped kernels fall back near ref performance.
- The strongest evidence is `embed_as_linear M=1`: `0.557 ms -> 5.886 ms`, almost exactly matching ref `5.852 ms`.
- This confirms our faster decode mainly comes from `dispatch_threads(M, K)`, not from Flash Attention.
- The temporary dispatch change was reverted after this experiment.
