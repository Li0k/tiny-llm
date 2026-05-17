// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "axpby.h"
#include "tiny_llm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a, "a"_a,
          "b"_a, "transpose_b"_a = false, nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Quantized matrix multiplication.

            Computes ``a @ b.T`` when ``transpose_b=True`` using packed quantized
            weights with per-group scale and bias.

            Args:
                scales (array): Per-group scales.
                biases (array): Per-group biases.
                group_size (int): Number of values per quantization group.
                bits (int): Quantization bit width.
                a (array): Activation matrix.
                b (array): Packed quantized weight matrix.
                transpose_b (bool): Whether to treat ``b`` as transposed.

            Returns:
                array: Quantized matmul result.
        )");

    m.def("flash_attention", &tiny_llm_ext::flash_attention, "query"_a, "key"_a, "value"_a, "mask"_a, "scale"_a = 1.0,
          "is_causal"_a = false, nb::kw_only(), "num_kv_heads"_a, "num_heads"_a, "stream"_a = nb::none(),
          R"(
        Flash attention layer.

        Computes ``softmax(query @ key.T * scale + mask) @ value``.
    )");
}
