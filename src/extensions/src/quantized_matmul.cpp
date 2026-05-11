#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {
mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                           const mx::array &biases,         // Input array biases
                           const int group_size,            // Group size
                           const int bits,                  // Number of bits
                           const mx::array &a,              // Input array a (not quantized)
                           const mx::array &b,              // Input array b (quantized)
                           const bool transpose_b,          // Whether to transpose b
                           mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
) {
    if (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16 && scales.dtype() != mx::float32) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16 or float32");
    }
    if (scales.dtype() != biases.dtype()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same dtype");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must have the same dtype as scales");
    }
    if (a.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 64) {
        throw std::runtime_error("quantized_matmul: group_size must be 64");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: transpose_b must be true");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b must have the same number of rows as scales");
    }
    if (b.shape()[1] != scales.shape()[1] * group_size / 8) {
        throw std::runtime_error("quantized_matmul: packed b shape does not match scales/group_size");
    }
    if (a.shape()[1] != b.shape()[1] * 8) {
        throw std::runtime_error("quantized_matmul: a shape does not match packed b");
    }

    auto out_shape = a.shape();
    out_shape[1] = b.shape()[0];

    return mx::array(out_shape, a.dtype(), std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
                     std::vector<mx::array>{scales, biases, a, b});
}

template <typename T>
void quantized_matmul_impl_typed(const mx::array &scales, const mx::array &biases, const mx::array &a,
                                 const mx::array &b, mx::array &out, int group_size, int bits, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be row contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be row contiguous");
    }

    encoder.dispatch([out_ptr = out.data<T>(), out_shape = out.shape(), out_strides = out.strides(),
                      group_size = group_size, bits = bits, a = mx::array::unsafe_weak_copy(a),
                      b = mx::array::unsafe_weak_copy(b), scales = mx::array::unsafe_weak_copy(scales),
                      biases = mx::array::unsafe_weak_copy(biases)]() {
        const int M = a.shape()[0];
        const int N = a.shape()[1];
        const int K = b.shape()[0];

        const int groups_per_row = N / group_size;
        const int packs_per_u32 = 32 / bits;
        const int u32_per_group = group_size / packs_per_u32;
        const uint32_t pack_mask = (1u << bits) - 1u;

        const T *a_ptr = a.data<T>();
        const uint32_t *b_ptr = b.data<uint32_t>();
        const T *scales_ptr = scales.data<T>();
        const T *biases_ptr = biases.data<T>();

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                float sum = 0.0f;  // accumulate in higher precision to reduce quantization error

                for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
                    int64_t scale_idx =
                        mx::elem_to_loc(j * groups_per_row + group_idx, scales.shape(), scales.strides());
                    int64_t bias_idx =
                        mx::elem_to_loc(j * groups_per_row + group_idx, biases.shape(), biases.strides());

                    const T scale = scales_ptr[scale_idx];
                    const T bias = biases_ptr[bias_idx];

                    int64_t a_idx = mx::elem_to_loc(i * N + group_idx * group_size, a.shape(), a.strides());
                    int64_t b_idx =
                        mx::elem_to_loc((j * N + group_idx * group_size) / packs_per_u32, b.shape(), b.strides());

                    for (int pack_idx = 0; pack_idx < u32_per_group; ++pack_idx) {
                        const uint32_t packed = b_ptr[b_idx];

                        for (int lane = 0; lane < packs_per_u32; ++lane) {
                            const uint32_t q = (packed >> (lane * bits)) & pack_mask;

                            const float a_val = static_cast<float>(a_ptr[a_idx]);
                            const float b_val =
                                static_cast<float>(q) * static_cast<float>(scale) + static_cast<float>(bias);
                            sum += a_val * b_val;

                            a_idx++;
                        }

                        b_idx++;
                    }
                }

                int64_t out_idx = mx::elem_to_loc(i * K + j, out_shape, out_strides);
                out_ptr[out_idx] = static_cast<T>(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    switch (a.dtype()) {
        case mx::float16:
            quantized_matmul_impl_typed<mx::float16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        case mx::float32:
            quantized_matmul_impl_typed<float>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        case mx::bfloat16:
            quantized_matmul_impl_typed<mx::bfloat16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        default:
            throw std::runtime_error("quantized_matmul: unsupported dtype");
    }
}

#ifdef _METAL_
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    const char *kernel_name;
    if (a.dtype() == mx::float16) {
        kernel_name = "quantized_matmul_w4a16_g64_f16";
    } else if (a.dtype() == mx::bfloat16) {
        kernel_name = "quantized_matmul_w4a16_g64_bf16";
    } else {
        throw std::runtime_error("quantized_matmul: a must be float16 or bfloat16 on GPU");
    }

    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel(kernel_name, library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = 32;
    const int y_size = static_cast<int>(tgp_size / x_size);

    MTL::Size group_dims = MTL::Size(x_size, y_size, 1);
    MTL::Size grid_dims = MTL::Size(M, K, 1);

    compute_encoder.dispatch_threads(grid_dims, group_dims);
}
#else
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("QuantizedMatmul has no GPU implementation.");
}
#endif

}  // namespace tiny_llm_ext
