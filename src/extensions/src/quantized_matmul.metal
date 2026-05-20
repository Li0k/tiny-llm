#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, int group_size>
[[kernel]] void quantized_matmul_w4a16(
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
            sum += ((float)((packed >> 0) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx];
            sum += ((float)((packed >> 4) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 1];
            sum += ((float)((packed >> 8) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 2];
            sum += ((float)((packed >> 12) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 3];
            sum += ((float)((packed >> 16) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 4];
            sum += ((float)((packed >> 20) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 5];
            sum += ((float)((packed >> 24) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 6];
            sum += ((float)((packed >> 28) & pack_mask) * (float)scale + (float)bias) * (float)a[a_idx + 7];

            a_idx += packs_per_u32;
            b_idx += 1;
        }
    }

    out[i * K + j] = (T)sum;
}

instantiate_kernel("quantized_matmul_w4a16_g64_f16", quantized_matmul_w4a16, float16_t, 64);
instantiate_kernel("quantized_matmul_w4a16_g64_bf16", quantized_matmul_w4a16, bfloat16_t, 64);
instantiate_kernel("quantized_matmul_w4a16_g128_f16", quantized_matmul_w4a16, float16_t, 128);
instantiate_kernel("quantized_matmul_w4a16_g128_bf16", quantized_matmul_w4a16, bfloat16_t, 128);
