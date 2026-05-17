#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {
mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const bool is_causal, const int num_kv_heads, const int num_heads,
                          mx::StreamOrDevice s) {
    // The Python wrapper flattens inputs to q=[N,L,E], k/v=[N_kv,S,E], mask=[N,L,S].
    // The CPU/GPU kernels below assume float32 and simple 3D indexing.
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32 ||
        mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be float32");
    }
    if (q.shape().size() != 3 || k.shape().size() != 3 || v.shape().size() != 3 || mask.shape().size() != 3) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be 3D");
    }

    // GQA maps several query heads to one key/value head, so the ratio must be integral.
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }

    // q.shape[0] is batch * query_heads; k/v.shape[0] is batch * kv_heads.
    if (q.shape()[0] % num_heads != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (k.shape()[0] % num_kv_heads != 0 || v.shape()[0] % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: k/v shape[0] must be divisible by num_kv_heads");
    }
    if (q.shape()[0] / num_heads != k.shape()[0] / num_kv_heads) {
        throw std::runtime_error("flash_attention: q and k batch dimensions do not match");
    }
    if (k.shape()[0] != v.shape()[0]) {
        throw std::runtime_error("flash_attention: k and v batch-head dimensions must match");
    }

    // q=[N,L,E], k/v=[N_kv,S,E], so E must match and k/v must share S.
    if (q.shape()[2] != k.shape()[2] || q.shape()[2] != v.shape()[2]) {
        throw std::runtime_error("flash_attention: q, k, and v head dimensions must match");
    }
    if (k.shape()[1] != v.shape()[1]) {
        throw std::runtime_error("flash_attention: k and v sequence lengths must match");
    }

    // Each flattened query head has one LxS mask slice.
    if (mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] || mask.shape()[2] != k.shape()[1]) {
        throw std::runtime_error("flash_attention: mask shape must be [N, L, S]");
    }

    // MLX is lazy: this creates an output node. eval_cpu/eval_gpu does the real computation later.
    return mx::array(q.shape(), mx::float32,
                     std::make_shared<FlashAttention>(to_stream(s), scale, is_causal, num_kv_heads, num_heads),
                     std::vector<mx::array>{q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }
    if (!mask.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: mask must be contiguous");
    }

    encoder.dispatch([out_ptr = out.data<float>(), q = mx::array::unsafe_weak_copy(q),
                      k = mx::array::unsafe_weak_copy(k), v = mx::array::unsafe_weak_copy(v),
                      mask = mx::array::unsafe_weak_copy(mask), scale = scale_, is_causal = is_causal_,
                      num_heads = num_heads_, num_kv_heads = num_kv_heads_]() {
        // Global flattened shapes:
        // q=[N, L, E], k/v=[N_kv, S, E], mask=[N, L, S].
        const int64_t num_query_heads_flat = q.shape()[0];
        const int64_t query_len = q.shape()[1];
        const int64_t kv_len = k.shape()[1];
        const int64_t head_dim = q.shape()[2];

        // Tile sizes. Br splits query rows; Bc splits key/value columns.
        const int64_t Br = 32;
        const int64_t Bc = 32;

        const int64_t num_query_tiles = (query_len + Br - 1) / Br;
        const int64_t num_kv_tiles = (kv_len + Bc - 1) / Bc;

        // GQA: this many query heads share one key/value head.
        const int64_t q_kv_heads_ratio = num_heads / num_kv_heads;

        // Contiguous flattened storage size for one head.
        const int64_t query_head_size = query_len * head_dim;
        const int64_t kv_head_size = kv_len * head_dim;

        // In decode, query_len can be smaller than kv_len, so causal columns are shifted by kv_len - query_len.
        const int64_t causal_offset = kv_len - query_len;

        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *mask_ptr = mask.data<float>();

        for (int64_t query_head_idx = 0; query_head_idx < num_query_heads_flat; ++query_head_idx) {
            // q is flattened as [batch * query_heads, L, E].
            const float *q_head = q_ptr + query_head_idx * query_head_size;

            // GQA: several query heads share one KV head.
            const int64_t kv_head_idx = query_head_idx / q_kv_heads_ratio;
            const float *k_head = k_ptr + kv_head_idx * kv_head_size;
            const float *v_head = v_ptr + kv_head_idx * kv_head_size;

            for (int64_t query_tile_idx = 0; query_tile_idx < num_query_tiles; ++query_tile_idx) {
                // Current query tile covers rows [q_start, q_start + q_count).
                const int64_t query_start = query_tile_idx * Br;
                const int64_t query_count = std::min(Br, query_len - query_start);

                // Online softmax state for this query tile.
                // m_i: row-wise running max score.
                // l_i: row-wise running softmax denominator.
                // o_i: row-wise running output numerator, shape [query_count, head_dim].
                std::vector<float> m_i(query_count, -std::numeric_limits<float>::infinity());
                std::vector<float> l_i(query_count, 0.0f);
                std::vector<float> o_i(query_count * head_dim, 0.0f);

                for (int64_t kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
                    // Current key/value tile covers columns [kv_start, kv_start + kv_count).
                    const int64_t kv_start = kv_tile_idx * Bc;
                    const int64_t kv_count = std::min(Bc, kv_len - kv_start);

                    // Last query row in this query tile.
                    // This is the row that can see the furthest to the right under causal masking.
                    const int64_t query_max = query_start + query_count - 1;

                    // If even the latest query row cannot see the first key in this KV tile,
                    // then the entire score tile is masked to -inf and contributes nothing.
                    if (is_causal && kv_start > query_max + causal_offset) {
                        continue;
                    }

                    // Last key column in this KV tile.
                    const int64_t kv_max = kv_start + kv_count - 1;

                    // If even the earliest query row can see the last key in this KV tile,
                    // then every entry in this score tile is causally valid.
                    // In that case the causal mask is all zeros and we can skip reading mask.
                    const bool block_all_valid = is_causal && kv_max <= query_start + causal_offset;

                    // scores is the current small score matrix:
                    //   Q_tile [query_count, head_dim] @ K_tile.T [head_dim, kv_count]
                    // = scores [query_count, kv_count]
                    std::vector<float> scores(query_count * kv_count, 0.0f);

                    for (int64_t query_tile_row = 0; query_tile_row < query_count; ++query_tile_row) {
                        // Convert tile-local row index to global query position.
                        const int64_t query_row = query_start + query_tile_row;

                        for (int64_t kv_tile_col = 0; kv_tile_col < kv_count; ++kv_tile_col) {
                            // Convert tile-local column index to global key/value position.
                            const int64_t key_col = kv_start + kv_tile_col;

                            // Compute one attention score:
                            // score = dot(q[query_row, :], k[key_col, :])
                            float score = 0.0f;
                            for (int64_t dim = 0; dim < head_dim; ++dim) {
                                score += q_head[query_row * head_dim + dim] * k_head[key_col * head_dim + dim];
                            }

                            // Apply scaled dot-product attention factor.
                            score *= scale;

                            // If the block is not known to be fully valid, add the precomputed mask.
                            // Valid positions add 0; invalid causal positions add -inf.
                            if (!block_all_valid) {
                                score += mask_ptr[query_head_idx * query_len * kv_len + query_row * kv_len + key_col];
                            }

                            // Store into scores[query_tile_row, kv_tile_col].
                            scores[query_tile_row * kv_count + kv_tile_col] = score;
                        }
                    }

                    // Convert the current score tile into unnormalized softmax weights.
                    // p has the same logical shape as scores: [query_count, kv_count].
                    std::vector<float> p(scores.size(), 0.0f);

                    for (int64_t query_tile_row = 0; query_tile_row < query_count; ++query_tile_row) {
                        // Formula: m_new = max(m_old, max(score_tile_row)).
                        float rowmax = -std::numeric_limits<float>::infinity();
                        for (int64_t kv_tile_col = 0; kv_tile_col < kv_count; ++kv_tile_col) {
                            rowmax = std::max(rowmax, scores[query_tile_row * kv_count + kv_tile_col]);
                        }

                        const float m_new = std::max(m_i[query_tile_row], rowmax);

                        // Formula term: exp(m_old - m_new), used to rescale old l_i/o_i.
                        const float prev_scale = std::exp(m_i[query_tile_row] - m_new);
                        m_i[query_tile_row] = m_new;

                        // Formula term for l_i: sum(exp(score_tile_row - m_new)).
                        float rowsum = 0.0f;
                        for (int64_t kv_tile_col = 0; kv_tile_col < kv_count; ++kv_tile_col) {
                            const float prob = std::exp(scores[query_tile_row * kv_count + kv_tile_col] - m_new);
                            p[query_tile_row * kv_count + kv_tile_col] = prob;
                            rowsum += prob;
                        }

                        // Formula: l_new = exp(m_old - m_new) * l_old + sum(exp(score_tile - m_new)).
                        l_i[query_tile_row] = prev_scale * l_i[query_tile_row] + rowsum;

                        for (int64_t dim = 0; dim < head_dim; ++dim) {
                            // Formula term for o_i: sum(exp(score_tile - m_new) * value_tile).
                            float weighted_value_sum = 0.0f;

                            for (int64_t kv_tile_col = 0; kv_tile_col < kv_count; ++kv_tile_col) {
                                const int64_t value_row = kv_start + kv_tile_col;
                                weighted_value_sum +=
                                    p[query_tile_row * kv_count + kv_tile_col] * v_head[value_row * head_dim + dim];
                            }

                            // Formula: o_new = exp(m_old - m_new) * o_old + weighted_value_sum.
                            o_i[query_tile_row * head_dim + dim] =
                                prev_scale * o_i[query_tile_row * head_dim + dim] + weighted_value_sum;
                        }
                    }
                }

                // Final formula after all KV tiles: output = o / l.
                for (int64_t query_tile_row = 0; query_tile_row < query_count; ++query_tile_row) {
                    const int64_t query_row = query_start + query_tile_row;

                    for (int64_t dim = 0; dim < head_dim; ++dim) {
                        out_ptr[query_head_idx * query_head_size + query_row * head_dim + dim] =
                            o_i[query_tile_row * head_dim + dim] / l_i[query_tile_row];
                    }
                }
            }
        }
    });
}

#ifdef _METAL_
void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("flash_attention_f32_e128", library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_vector_bytes(mask.shape(), 5);
    compute_encoder.set_vector_bytes(mask.strides(), 6);

    if (!q.flags().row_contiguous || !k.flags().row_contiguous || !v.flags().row_contiguous ||
        !mask.flags().row_contiguous || !out.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: inputs and output must be row contiguous");
    }

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int S = k.shape()[1];
    const int E = q.shape()[2];
    const int Br = L == 1 ? 1 : 32;
    const int Bc = 32;
    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;
    const int is_causal = static_cast<int>(is_causal_);

    if (E > 128) {
        throw std::runtime_error("flash_attention: GPU implementation supports E <= 128");
    }

    compute_encoder.set_bytes(is_causal, 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(L, 9);
    compute_encoder.set_bytes(S, 10);
    compute_encoder.set_bytes(E, 11);
    compute_encoder.set_bytes(num_kv_heads_, 12);
    compute_encoder.set_bytes(num_heads_, 13);
    compute_encoder.set_bytes(scale_, 14);
    compute_encoder.set_bytes(Br, 15);
    compute_encoder.set_bytes(Bc, 16);
    compute_encoder.set_bytes(Tr, 17);
    compute_encoder.set_bytes(Tc, 18);

    size_t simd_width = kernel->threadExecutionWidth();
    MTL::Size num_threadgroups = MTL::Size(N, Tr, 1);
    MTL::Size num_threads_per_group = MTL::Size(Br, simd_width, 1);

    compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);
}
#else
void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("FlashAttention has no GPU implementation.");
}
#endif

}  // namespace tiny_llm_ext
