#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64(
    const device T* scales [[buffer(0)]],
    const device T* biases [[buffer(1)]],
    const device T* a [[buffer(2)]],
    const device uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int group_size = 64;
    const int bits = 4;
    const int groups_per_row = N / group_size;
    const int packs_per_u32 = 32 / bits;      // 8
    const int u32_per_group = group_size / packs_per_u32; // 8
    const uint32_t pack_mask = (1u << bits) - 1u;

    const int i = gid.x;
    const int j = gid.y;

    if (i >= M || j >= K) {
        return;
    }

    float sum = 0.0f;

    for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
        const int scale_idx = j * groups_per_row + group_idx;
        const int bias_idx = j * groups_per_row + group_idx;

        const T scale = scales[scale_idx];
        const T bias = biases[bias_idx];

        int a_idx = i * N + group_idx * group_size;
        int b_idx = (j * N + group_idx * group_size) / packs_per_u32;


        for (int pack_idx = 0; pack_idx < u32_per_group; pack_idx++) {
            const uint32_t packed = b[b_idx];

            for (int lane = 0; lane < packs_per_u32; lane++) {
                const uint32_t q = (packed >> (lane * bits)) & pack_mask;

                const float a_val = (float)a[a_idx];
                const float b_val = (float)q * (float)scale + (float)bias;

                sum += a_val * b_val;
                a_idx += 1;
            }

            b_idx += 1;
        }
    }

    out[i * K + j] = (T)sum;
}

instantiate_kernel("quantized_matmul_w4a16_g64_f16", quantized_matmul_w4a16_g64, float16_t);
instantiate_kernel("quantized_matmul_w4a16_g64_bf16", quantized_matmul_w4a16_g64, bfloat16_t);
