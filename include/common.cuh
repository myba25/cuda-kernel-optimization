// common.cuh - shared helpers for benchmarks and tests (host side only, no kernels here).
//
//   CUDA_CHECK, CUDA_CHECK_LAST, REQUIRE  error handling
//   DeviceBuffer<T>                        RAII device memory with upload/download
//   random_vector()                        deterministic test data, identical on every compiler
//   compare(), report()                    check a result against a reference
//   GpuTimer, benchmark()                  cudaEvent timing: warm-up, many runs, median
//   gbps(), gflops()                       throughput from bytes/flops and milliseconds
//   DeviceInfo                             GPU name, SM count, theoretical DRAM bandwidth
//   CsvWriter, results_path()              CSV output to results/<gpu>/
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#ifndef RESULTS_DIR
#define RESULTS_DIR "results"
#endif

// ---------------------------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------------------------

// Wrap every CUDA API call.
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n  call: %s\n  %s\n",            \
                         cudaGetErrorName(err_), __FILE__, __LINE__, #call,               \
                         cudaGetErrorString(err_));                                       \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Call right after every kernel launch: catches invalid launch configurations.
// Errors that happen while the kernel runs show up at the next synchronizing call.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// For launchers that only support some shapes: reject loudly instead of computing garbage.
#define REQUIRE(cond, msg)                                                                \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "Requirement failed at %s:%d: %s\n  (%s)\n", __FILE__,   \
                         __LINE__, msg, #cond);                                           \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T))); }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return n_; }
    size_t bytes() const { return n_ * sizeof(T); }

    void upload(const T* host, size_t count) {
        REQUIRE(count <= n_, "upload larger than the buffer");
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

    std::vector<T> download(size_t count) const {
        REQUIRE(count <= n_, "download larger than the buffer");
        std::vector<T> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    std::vector<T> download() const { return download(n_); }

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------------------------

// splitmix64: tiny and fast. Unlike std::uniform_real_distribution (implementation-defined),
// it produces the same numbers with MSVC, GCC and Clang, so a seed means the same data everywhere.
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform floats in [lo, hi) from 24 random bits.
inline void fill_uniform(float* data, size_t n, uint64_t seed, float lo, float hi) {
    uint64_t state = seed;
    const float scale = (hi - lo) / 16777216.0f;  // 2^24
    for (size_t i = 0; i < n; ++i) data[i] = lo + static_cast<float>(splitmix64(state) >> 40) * scale;
}

// Note for reductions: values in [-1, 1) sum to roughly zero, which makes relative error
// meaningless. Use [0, 1) there.
inline std::vector<float> random_vector(size_t n, uint64_t seed = 42, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    fill_uniform(v.data(), n, seed, lo, hi);
    return v;
}

// ---------------------------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------------------------

struct CompareResult {
    bool ok = true;
    size_t n = 0;
    size_t n_bad = 0;         // elements outside tolerance; NaN / Inf always count as bad
    size_t first_bad = 0;     // index of the first bad element (handy for tail / off-by-one bugs)
    double got_first_bad = 0.0;
    double ref_first_bad = 0.0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

// An element passes if |got - ref| <= atol + rtol * |ref| (the numpy.allclose rule).
// Do not expect bit-identical results: a different summation order changes the rounding.
template <typename TGot, typename TRef>
CompareResult compare(const TGot* got, const TRef* ref, size_t n, double rtol, double atol) {
    CompareResult r;
    r.n = n;
    for (size_t i = 0; i < n; ++i) {
        const double g = static_cast<double>(got[i]);
        const double e = static_cast<double>(ref[i]);
        const double abs_err = std::fabs(g - e);
        bool bad = !std::isfinite(g);
        if (!bad) {
            r.max_abs_err = std::max(r.max_abs_err, abs_err);
            r.max_rel_err = std::max(r.max_rel_err, abs_err / std::max(std::fabs(e), 1e-30));
            bad = abs_err > atol + rtol * std::fabs(e);
        }
        if (bad) {
            if (r.n_bad == 0) {
                r.first_bad = i;
                r.got_first_bad = g;
                r.ref_first_bad = e;
            }
            ++r.n_bad;
        }
    }
    r.ok = (r.n_bad == 0);
    return r;
}

// Prints one PASS/FAIL line and returns r.ok.
inline bool report(const std::string& name, const CompareResult& r) {
    if (r.ok) {
        std::printf("PASS  %-32s max abs err %.3e, max rel err %.3e\n", name.c_str(), r.max_abs_err,
                    r.max_rel_err);
    } else {
        std::printf("FAIL  %-32s %zu of %zu wrong, first at [%zu]: got %.9g, expected %.9g\n",
                    name.c_str(), r.n_bad, r.n, r.first_bad, r.got_first_bad, r.ref_first_bad);
    }
    return r.ok;
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    void stop(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }

    // Waits for the stop event, then returns the GPU time between the two events.
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

struct BenchConfig {
    int warmup = 10;                // untimed runs first: clocks ramp up, caches and TLBs warm up
    int iters = 100;                // timed runs; the median is reported
    cudaStream_t stream = nullptr;  // the stream the launch function uses
    std::function<void()> reset;    // optional, runs before every launch and is NOT timed
                                    // (e.g. zero an atomicAdd accumulator)
};

struct BenchStats {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    int iters = 0;
};

// Times `launch()` with cudaEvents around each run: kernel time only, no host<->device copies.
// `launch` must enqueue its work on cfg.stream.
template <typename Launch>
BenchStats benchmark(Launch&& launch, const BenchConfig& cfg = BenchConfig{}) {
    for (int i = 0; i < cfg.warmup; ++i) {
        if (cfg.reset) cfg.reset();
        launch();
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(cfg.stream));

    GpuTimer timer;
    std::vector<double> ms;
    ms.reserve(cfg.iters);
    for (int i = 0; i < cfg.iters; ++i) {
        if (cfg.reset) cfg.reset();
        timer.start(cfg.stream);
        launch();
        timer.stop(cfg.stream);
        CUDA_CHECK_LAST();
        ms.push_back(timer.elapsed_ms());
    }

    std::sort(ms.begin(), ms.end());
    BenchStats s;
    s.iters = cfg.iters;
    s.min_ms = ms.front();
    s.max_ms = ms.back();
    const size_t mid = ms.size() / 2;
    s.median_ms = (ms.size() % 2) ? ms[mid] : 0.5 * (ms[mid - 1] + ms[mid]);
    return s;
}

inline double gbps(double bytes, double ms) { return bytes / (ms * 1e6); }    // GB/s,   1 GB = 1e9 B
inline double gflops(double flops, double ms) { return flops / (ms * 1e6); }  // GFLOPS

// ---------------------------------------------------------------------------------------------
// Device description
// ---------------------------------------------------------------------------------------------

struct DeviceInfo {
    int id = 0;
    std::string name;           // "NVIDIA GeForce RTX 4050 Laptop GPU"
    std::string slug;           // "rtx_4050_laptop_gpu", folder name under results/
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    size_t global_mem_bytes = 0;
    int mem_clock_khz = 0;
    int mem_bus_width_bits = 0;
    double peak_bw_gbps = 0.0;  // theoretical: 2 (double data rate) * memory clock * bus width
    int runtime_version = 0;    // 13040 -> 13.4
    int driver_api_version = 0; // highest CUDA version the installed driver supports
};

inline std::string make_slug(const std::string& name) {
    std::string s;
    for (char c : name) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u)) {
            s += static_cast<char>(std::tolower(u));
        } else if (!s.empty() && s.back() != '_') {
            s += '_';
        }
    }
    while (!s.empty() && s.back() == '_') s.pop_back();
    for (const std::string prefix : {"nvidia_", "geforce_"}) {
        if (s.rfind(prefix, 0) == 0) s.erase(0, prefix.size());
    }
    return s;
}

inline DeviceInfo get_device_info(int device = 0) {
    DeviceInfo d;
    d.id = device;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    d.name = p.name;
    d.slug = make_slug(d.name);
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.sm_count = p.multiProcessorCount;
    d.global_mem_bytes = p.totalGlobalMem;
    // Attributes instead of cudaDeviceProp fields: memoryClockRate was removed from the struct in CUDA 13.
    CUDA_CHECK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&d.mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));
    d.peak_bw_gbps = 2.0 * d.mem_clock_khz * 1e3 * (d.mem_bus_width_bits / 8.0) / 1e9;
    CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&d.driver_api_version));
    return d;
}

inline void print_device_info(const DeviceInfo& d) {
    std::printf("GPU %d: %s, sm_%d%d, %d SMs, %.1f GiB\n", d.id, d.name.c_str(), d.cc_major, d.cc_minor,
                d.sm_count, d.global_mem_bytes / double(1ull << 30));
    std::printf("Theoretical DRAM bandwidth: %.1f GB/s (%d-bit bus, %d MHz)\n", d.peak_bw_gbps,
                d.mem_bus_width_bits, d.mem_clock_khz / 1000);
    std::printf("CUDA runtime %d.%d, driver supports up to CUDA %d.%d\n", d.runtime_version / 1000,
                (d.runtime_version % 1000) / 10, d.driver_api_version / 1000,
                (d.driver_api_version % 1000) / 10);
}

// ---------------------------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------------------------

// results/<gpu>/<filename>
inline std::string results_path(const DeviceInfo& d, const std::string& filename) {
    return std::string(RESULTS_DIR) + "/" + d.slug + "/" + filename;
}

// Overwrites the file on every run, so a CSV always holds one complete benchmark run.
class CsvWriter {
public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header) : path_(path) {
        const std::filesystem::path dir = std::filesystem::path(path).parent_path();
        if (!dir.empty()) std::filesystem::create_directories(dir);
        out_.open(path);
        if (!out_) {
            std::fprintf(stderr, "Cannot open %s for writing\n", path.c_str());
            std::exit(EXIT_FAILURE);
        }
        out_.precision(8);
        for (size_t i = 0; i < header.size(); ++i) out_ << (i ? "," : "") << header[i];
        out_ << '\n';
    }

    // Values must not contain commas.
    template <typename... Ts>
    void row(const Ts&... values) {
        const char* sep = "";
        ((out_ << sep << values, sep = ","), ...);
        out_ << '\n';
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::ofstream out_;
};
