// Device-resident NTT benchmark hook — times kernel only, no PCIe transfer. Not compiled by
// build.rs or called by benches/ntt_bench.rs yet; cuda/ntt_bench.cu is the standalone equivalent.
#include <ff/baby_bear.hpp>
#include <ntt/ntt.cuh>
#include <chrono>

extern "C" int cuda_ntt_bench(uint32_t lg_domain_size, uint32_t warmup,
                               uint32_t samples, uint64_t* best_ns)
{
    try {
        auto& gpu = select_gpu(0);
        size_t domain_size = (size_t)1 << lg_domain_size;

        // Allocate device memory
        dev_ptr_t<fr_t> d_data{domain_size, gpu};

        // Initialize with zeros via cudaMemset
        cudaMemset(&d_data[0], 0, domain_size * sizeof(fr_t));
        gpu.sync();

        // Warm up
        for (uint32_t w = 0; w < warmup; w++) {
            NTT::Base_dev_ptr(gpu[0], &d_data[0], lg_domain_size,
                              NTT::InputOutputOrder::RN,
                              NTT::Direction::forward,
                              NTT::Type::standard);
        }
        gpu.sync();

        // Timed samples — take minimum
        uint64_t min_ns = UINT64_MAX;
        for (uint32_t s = 0; s < samples; s++) {
            auto start = std::chrono::high_resolution_clock::now();
            NTT::Base_dev_ptr(gpu[0], &d_data[0], lg_domain_size,
                              NTT::InputOutputOrder::RN,
                              NTT::Direction::forward,
                              NTT::Type::standard);
            gpu.sync();
            auto end = std::chrono::high_resolution_clock::now();
            uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            if (ns < min_ns) min_ns = ns;
        }

        *best_ns = min_ns;
        return 0;
    } catch (...) {
        return -1;
    }
}
