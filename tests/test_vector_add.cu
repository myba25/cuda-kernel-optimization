// Checks launch_vector_add() against a CPU reference in double on sizes that are not multiples
// of any block size, and catches writes past the end of the output.
// Stricter check for out-of-bounds reads and writes:
//   compute-sanitizer --tool memcheck build/bin/test_vector_add
#include "common.cuh"
#include "vector_add/vector_add.cuh"

#include <cstring>

namespace {

constexpr size_t kGuard = 256;  // extra output elements after n that must stay untouched

bool tail_untouched(const std::vector<float>& out, size_t n) {
    const std::vector<unsigned char> pattern(kGuard * sizeof(float), 0xFF);
    return std::memcmp(out.data() + n, pattern.data(), pattern.size()) == 0;
}

bool run(size_t n) {
    const std::vector<float> a = random_vector(n, /*seed=*/1);
    const std::vector<float> b = random_vector(n, /*seed=*/2);
    std::vector<double> ref(n);
    for (size_t i = 0; i < n; ++i) ref[i] = double(a[i]) + double(b[i]);

    DeviceBuffer<float> d_a(n), d_b(n), d_c(n + kGuard);
    d_a.upload(a);
    d_b.upload(b);
    CUDA_CHECK(cudaMemset(d_c.data(), 0xFF, d_c.bytes()));  // NaN pattern: unwritten elements fail

    launch_vector_add(d_a.data(), d_b.data(), d_c.data(), n, nullptr);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> c = d_c.download();

    const std::string name = "vector_add n=" + std::to_string(n);
    bool ok = report(name, compare(c.data(), ref.data(), n, /*rtol=*/1e-6, /*atol=*/1e-7));
    if (!tail_untouched(c, n)) {
        std::printf("FAIL  %-32s wrote past the end of the output\n", name.c_str());
        ok = false;
    }
    return ok;
}

}  // namespace

int main() {
    const size_t sizes[] = {1, 1000, 1023, size_t(1) << 20, (size_t(1) << 20) + 3};
    int failed = 0;
    for (size_t n : sizes) failed += run(n) ? 0 : 1;
    if (failed) {
        std::printf("\n%d size(s) failed\n", failed);
        return EXIT_FAILURE;
    }
    std::printf("\nAll sizes passed\n");
    return EXIT_SUCCESS;
}
