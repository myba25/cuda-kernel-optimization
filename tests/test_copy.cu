// Checks launch_copy() bit for bit on sizes that are not multiples of any block size,
// and catches writes past the end of the output.
// Stricter check for out-of-bounds reads and writes:
//   compute-sanitizer --tool memcheck build/bin/test_copy
#include "common.cuh"
#include "copy/copy.cuh"

#include <cstring>

namespace {

constexpr size_t kGuard = 256;  // extra output elements after n that must stay untouched

bool tail_untouched(const std::vector<float>& out, size_t n) {
    const std::vector<unsigned char> pattern(kGuard * sizeof(float), 0xFF);
    return std::memcmp(out.data() + n, pattern.data(), pattern.size()) == 0;
}

bool run(size_t n) {
    const std::vector<float> in = random_vector(n, /*seed=*/3);

    DeviceBuffer<float> d_in(n), d_out(n + kGuard);
    d_in.upload(in);
    CUDA_CHECK(cudaMemset(d_out.data(), 0xFF, d_out.bytes()));  // NaN pattern: unwritten elements fail

    launch_copy(d_in.data(), d_out.data(), n, nullptr);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> out = d_out.download();

    const std::string name = "copy n=" + std::to_string(n);
    bool ok = report(name, compare(out.data(), in.data(), n, /*rtol=*/0.0, /*atol=*/0.0));  // must be exact
    if (!tail_untouched(out, n)) {
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
