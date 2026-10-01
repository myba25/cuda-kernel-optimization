// bench_copy: the practical DRAM bandwidth peak.
//
// Copies N floats for N = 2^20 ... 2^28 and reports GB/s = 2 * N * 4 bytes / time
// (every element is read once and written once). Variants:
//   cudaMemcpy  device-to-device copy done by the driver (baseline)
//   copy        launch_copy() from kernels/copy/copy.cuh
//
// Output: results/<gpu>/bench_copy.csv
// Plot:   python scripts/plot.py results/<gpu>/bench_copy.csv
#include "common.cuh"
#include "copy/copy.cuh"

#include <cstring>

namespace {

using CopyFn = void (*)(const float* in, float* out, size_t n, cudaStream_t stream);

void memcpy_d2d(const float* in, float* out, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

struct Variant {
    const char* name;
    CopyFn fn;
};

// Add new copy versions at the end: plot colors follow this order.
const Variant kVariants[] = {
    {"cudaMemcpy", memcpy_d2d},
    {"copy", launch_copy},
};

// Runs the variant once and compares the output with the input bit for bit.
bool copies_correctly(const Variant& v, const DeviceBuffer<float>& in, DeviceBuffer<float>& out,
                      const std::vector<float>& host_in, size_t n) {
    CUDA_CHECK(cudaMemset(out.data(), 0xFF, n * sizeof(float)));  // NaN pattern: unwritten elements fail
    v.fn(in.data(), out.data(), n, nullptr);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = out.download(n);
    return std::memcmp(got.data(), host_in.data(), n * sizeof(float)) == 0;
}

}  // namespace

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    // Largest size: 2^28 floats (1 GiB per buffer), smaller if the GPU does not have the memory.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    int max_log2 = 28;
    while (max_log2 > 20 && 2 * (size_t(1) << max_log2) * sizeof(float) > free_bytes / 10 * 8) --max_log2;
    const size_t max_n = size_t(1) << max_log2;

    const std::vector<float> host_in = random_vector(max_n);
    DeviceBuffer<float> in(max_n), out(max_n);
    in.upload(host_in);

    CsvWriter csv(results_path(dev, "bench_copy.csv"),
                  {"gpu", "kernel", "variant", "n", "bytes", "median_ms", "min_ms", "gbps", "peak_gbps",
                   "pct_of_peak"});

    std::printf("\n%-12s %12s %10s %10s %9s %7s\n", "variant", "n", "MiB moved", "median ms", "GB/s", "% peak");
    for (int lg = 20; lg <= max_log2; ++lg) {
        const size_t n = size_t(1) << lg;
        const double bytes = 2.0 * n * sizeof(float);  // read once + write once
        for (const Variant& v : kVariants) {
            if (!copies_correctly(v, in, out, host_in, n)) {
                std::printf("%-12s %12zu  wrong result (not implemented yet?), skipped\n", v.name, n);
                continue;
            }
            const BenchStats s = benchmark([&] { v.fn(in.data(), out.data(), n, nullptr); });
            const double bw = gbps(bytes, s.median_ms);
            const double pct = 100.0 * bw / dev.peak_bw_gbps;
            std::printf("%-12s %12zu %10.0f %10.4f %9.1f %6.1f%%\n", v.name, n, bytes / (1 << 20), s.median_ms,
                        bw, pct);
            csv.row(dev.name, "copy", v.name, n, static_cast<size_t>(bytes), s.median_ms, s.min_ms, bw,
                    dev.peak_bw_gbps, pct);
        }
    }
    std::printf("\nSaved %s\n", csv.path().c_str());
    return EXIT_SUCCESS;
}
