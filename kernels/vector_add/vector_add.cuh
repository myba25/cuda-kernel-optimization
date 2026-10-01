// Stage 0 warm-up: vector add, c[i] = a[i] + b[i] for i in [0, n).
//
// One thread per element. The grid is rounded up to whole blocks, so the last block has threads
// past the end of the array; the `i < n` check keeps them from reading or writing out of bounds.
//
// Check: build/bin/test_vector_add
#pragma once

#include "common.cuh"

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, size_t n) {
    // Widen to size_t before multiplying: blockIdx.x * blockDim.x is 32-bit and overflows at 2^32.
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

inline void launch_vector_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
    constexpr unsigned kBlockSize = 256;  // a multiple of the 32-thread warp
    if (n == 0) return;                   // a grid of 0 blocks is an invalid launch
    const size_t grid = ceil_div(n, kBlockSize);
    REQUIRE(grid <= 0x7FFFFFFFu, "n is too large for a 1D grid");
    vector_add_kernel<<<static_cast<unsigned>(grid), kBlockSize, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_LAST();
}
