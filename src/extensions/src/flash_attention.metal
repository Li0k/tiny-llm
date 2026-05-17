#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int* mask_shape [[buffer(5)]],
    constant const int64_t* mask_strides [[buffer(6)]],
    device const int& is_causal [[buffer(7)]],
    device const int& N [[buffer(8)]],
    device const int& L [[buffer(9)]],
    device const int& S [[buffer(10)]],
    device const int& E [[buffer(11)]],
    device const int& num_kv_heads [[buffer(12)]],
    device const int& num_heads [[buffer(13)]],
    device const float& scale [[buffer(14)]],
    device const int& Br [[buffer(15)]],
    device const int& Bc [[buffer(16)]],
    [[maybe_unused]] device const int& Tr [[buffer(17)]],
    device const int& Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // One threadgroup computes one CPU-equivalent unit:
    //   CPU: for query_head_idx, for query_tile_idx -> compute one output tile.
    //   GPU: group_id.x = query_head_idx, group_id.y = query_tile_idx.
    const int query_head_idx = group_id.x;
    const int query_tile_idx = group_id.y;

    // Within the threadgroup:
    //   simd_gid chooses the row inside Q_tile.
    //   simd_lid chooses the column inside the current K/V tile.
    // Together, one lane computes one score_tile[row, col].
    const int query_tile_row = simd_gid;
    const int kv_tile_col = simd_lid;

    const int query_row = query_tile_idx * Br + query_tile_row;

    // Last query/KV tiles can be smaller than Br/Bc. Invalid lanes must not read out of bounds
    // and should contribute 0 to softmax/output.
    const bool query_in_range = query_head_idx < N && query_tile_row < Br && query_row < L;

    // GQA mapping: several query heads share one K/V head.
    const int q_kv_ratio = num_heads / num_kv_heads;
    device const float* q_head = q + query_head_idx * L * E;
    device const float* k_head = k + (query_head_idx / q_kv_ratio) * S * E;
    device const float* v_head = v + (query_head_idx / q_kv_ratio) * S * E;

    // q_local is Q_tile [Br, E], reused while this threadgroup scans all K/V tiles.
    // o_i is the running output numerator, same as CPU o_i [Br, E].
    threadgroup float q_local[32][128];
    threadgroup float o_i[32 * 128];

    // One lane per row loads Q_tile[row, :] into threadgroup memory and initializes o_i.
    // simd_lid == 0 means lane 0 in each SIMD group; there is one such lane for every query row.
    if (simd_lid == 0 && query_tile_row < Br) {
        for (int dim = 0; dim < E; dim++) {
            q_local[query_tile_row][dim] = query_in_range ? q_head[query_row * E + dim] : 0.0f;
            o_i[query_tile_row * E + dim] = 0.0f;
        }
    }

    // Synchronize the whole threadgroup so all lanes see initialized q_local/o_i before use.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-query-row online softmax state. These correspond to CPU m_i[row] and l_i[row].
    float m_i = -INFINITY;
    float l_i = 0.0f;
    const int causal_offset = S - L;

    // CPU equivalent: for kv_tile_idx in num_kv_tiles.
    // This loop is serial inside one threadgroup because all K/V tiles update the same m_i/l_i/o_i.
    for (int kv_tile_idx = 0; kv_tile_idx < Tc; kv_tile_idx++) {
        const int kv_start = kv_tile_idx * Bc;
        const int key_col = kv_start + kv_tile_col;

        // Causal fast path: if this entire K/V tile is future tokens, skip it.
        const int query_tile_end = min((query_tile_idx + 1) * Br, L) - 1;
        if (is_causal && kv_start > query_tile_end + causal_offset) {
            continue;
        }

        // Last K/V tile can have fewer than Bc columns.
        const bool kv_in_range = kv_tile_col < Bc && key_col < S;
        const int kv_tile_end = min((kv_tile_idx + 1) * Bc, S) - 1;

        // If the whole tile is valid under causal masking, mask is all zeros and can be skipped.
        const bool block_all_valid = is_causal && kv_tile_end <= query_tile_idx * Br + causal_offset;

        // Formula: score = dot(Q_tile[row, :], K_tile[col, :]) * scale + mask.
        // CPU equivalent: scores[query_tile_row, kv_tile_col].
        float score = 0.0f;
        if (query_in_range && kv_in_range) {
            for (int dim = 0; dim < E; dim++) {
                score += q_local[query_tile_row][dim] * k_head[key_col * E + dim];
            }
            score *= scale;

            if (!block_all_valid) {
                const int64_t mask_idx = elem_to_loc(
                    query_head_idx * L * S + query_row * S + key_col,
                    mask_shape,
                    mask_strides,
                    3
                );
                score += mask[mask_idx];
            }
        } else {
            score = -INFINITY;
        }

        // Formula: m_new = max(m_old, max(score_tile_row)).
        // CPU loop over kv_tile_col becomes simd_max across lanes in one SIMD group.
        const float rowmax = simd_max(score);
        const float m_new = max(m_i, rowmax);

        // Formula term: exp(m_old - m_new), used to rescale old l_i/o_i.
        const float prev_scale = exp(m_i - m_new);
        m_i = m_new;

        // Formula term: p = exp(score_tile - m_new).
        const float p = (query_in_range && kv_in_range) ? exp(score - m_i) : 0.0f;

        // Formula term for l_i: sum(exp(score_tile_row - m_new)).
        // CPU rowsum loop over kv_tile_col becomes simd_sum across lanes.
        const float rowsum = simd_sum(p);
        l_i = prev_scale * l_i + rowsum;

        // o_i lives in threadgroup memory and is updated every K/V tile. Ensure previous
        // writes are visible before reading/updating it for this tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int dim = 0; dim < E; dim++) {
            // Formula term for o_i: sum(exp(score_tile - m_new) * value_tile).
            // Each lane computes one p[col] * V[col, dim], then simd_sum reduces over columns.
            const float value_part = (query_in_range && kv_in_range) ? p * v_head[key_col * E + dim] : 0.0f;
            const float weighted_value_sum = simd_sum(value_part);

            // Only one lane writes the reduced value for this row/dim.
            if (simd_lid == 0 && query_in_range) {
                o_i[query_tile_row * E + dim] = prev_scale * o_i[query_tile_row * E + dim] + weighted_value_sum;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final formula after all K/V tiles: output = o / l.
    if (simd_lid == 0 && query_in_range) {
        for (int dim = 0; dim < E; dim++) {
            out[query_head_idx * L * E + query_row * E + dim] = o_i[query_tile_row * E + dim] / l_i;
        }
    }
}
