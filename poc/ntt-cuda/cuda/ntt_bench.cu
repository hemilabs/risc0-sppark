#include <cstdio>
#include <cstdlib>

#define FEATURE_BABY_BEAR
#include <ff/baby_bear.hpp>

#define protected public
#include <ntt/ntt.cuh>
#undef protected

int main()
{
    auto& gpu = select_gpu(0);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Device: %s, SMs: %d\n", props.name, props.multiProcessorCount);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =====================================================
    // Part 1: Kernel-only timing (data already on device)
    // =====================================================
    printf("\n=== Kernel-only timing (data on device) ===\n");
    printf("%-6s  %10s  %10s  %10s  %10s\n",
           "Size", "NR(GS)", "NN(BR+CT)", "BitRev", "EffBW(NR)");

    for (int lg = 18; lg <= 27; lg++) {
        size_t domain_size = (size_t)1 << lg;
        size_t bytes = domain_size * sizeof(fr_t);

        fr_t* h = (fr_t*)malloc(bytes);
        for (size_t i = 0; i < domain_size; i++)
            ((uint32_t*)h)[i] = (uint32_t)(i * 2654435761u) % 0x78000001;

        dev_ptr_t<fr_t> d{domain_size, gpu};
        gpu.HtoD(&d[0], h, domain_size);
        gpu.sync();

        int warmup = 5, iters = lg >= 25 ? 10 : 20;

        for (int i = 0; i < warmup; i++) {
            NTT::NTT_internal(&d[0], lg, NTT::InputOutputOrder::NR,
                NTT::Direction::forward, NTT::Type::standard, gpu);
            gpu.sync();
        }

        float best_nr = 1e9, best_nn = 1e9, best_br = 1e9;

        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            NTT::NTT_internal(&d[0], lg, NTT::InputOutputOrder::NR,
                NTT::Direction::forward, NTT::Type::standard, gpu);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_nr) best_nr = ms;
        }
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            NTT::NTT_internal(&d[0], lg, NTT::InputOutputOrder::NN,
                NTT::Direction::forward, NTT::Type::standard, gpu);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_nn) best_nn = ms;
        }
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            NTT::bit_rev(&d[0], &d[0], lg, gpu);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_br) best_br = ms;
        }

        int ntt_passes = (lg <= 10) ? 1 : (lg <= 18) ? 2 : (lg <= 30) ? 3 : 4;
        double nr_bw = (double)bytes * 2.0 * ntt_passes / (best_nr * 1e-3) / 1e9;

        printf("2^%-4d  %8.3fms  %8.3fms  %8.3fms  %6.0f GB/s\n",
               lg, best_nr, best_nn, best_br, nr_bw);
        free(h);
    }

    // =====================================================
    // Part 2: Transfer timing comparison (pageable vs pinned)
    // =====================================================
    printf("\n=== PCIe Transfer: pageable vs pinned memory ===\n");
    printf("%-6s  %12s  %12s  %12s  %12s\n",
           "Size", "Page_HtoD", "Page_DtoH", "Pin_HtoD", "Pin_DtoH");

    for (int lg = 22; lg <= 27; lg++) {
        size_t domain_size = (size_t)1 << lg;
        size_t bytes = domain_size * sizeof(fr_t);

        // Pageable memory
        fr_t* h_page = (fr_t*)malloc(bytes);
        memset(h_page, 0x42, bytes);

        // Pinned memory
        fr_t* h_pin;
        cudaMallocHost(&h_pin, bytes);
        memset(h_pin, 0x42, bytes);

        dev_ptr_t<fr_t> d{domain_size, gpu};

        int iters = 5;

        // Pageable HtoD
        float best_ph = 1e9;
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            gpu.HtoD(&d[0], h_page, domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_ph) best_ph = ms;
        }

        // Pageable DtoH
        float best_pd = 1e9;
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            gpu.DtoH(h_page, &d[0], domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_pd) best_pd = ms;
        }

        // Pinned HtoD
        float best_ih = 1e9;
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            gpu.HtoD(&d[0], h_pin, domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_ih) best_ih = ms;
        }

        // Pinned DtoH
        float best_id = 1e9;
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start, gpu);
            gpu.DtoH(h_pin, &d[0], domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_id) best_id = ms;
        }

        printf("2^%-4d  %8.3fms    %8.3fms    %8.3fms    %8.3fms\n",
               lg, best_ph, best_pd, best_ih, best_id);

        free(h_page);
        cudaFreeHost(h_pin);
    }

    // =====================================================
    // Part 3: Full end-to-end NTT (host API) with pinned mem
    // =====================================================
    printf("\n=== Full NTT (host API) for 2^27 ===\n");
    {
        int lg = 27;
        size_t domain_size = (size_t)1 << lg;
        size_t bytes = domain_size * sizeof(fr_t);

        fr_t* h_page = (fr_t*)malloc(bytes);
        fr_t* h_pin;
        cudaMallocHost(&h_pin, bytes);

        for (size_t i = 0; i < domain_size; i++) {
            uint32_t v = (uint32_t)(i * 2654435761u) % 0x78000001;
            ((uint32_t*)h_page)[i] = v;
            ((uint32_t*)h_pin)[i] = v;
        }

        // Pageable: full NTT via public API
        // warmup
        for (int i = 0; i < 2; i++)
            NTT::Base(gpu, h_page, lg, NTT::InputOutputOrder::NN,
                      NTT::Direction::forward, NTT::Type::standard);

        float best_page = 1e9;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(start, gpu);
            NTT::Base(gpu, h_page, lg, NTT::InputOutputOrder::NN,
                      NTT::Direction::forward, NTT::Type::standard);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_page) best_page = ms;
        }

        // Pinned: manual HtoD + kernel + DtoH
        dev_ptr_t<fr_t> d{domain_size, gpu};
        float best_pinned = 1e9;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(start, gpu);
            gpu.HtoD(&d[0], h_pin, domain_size);
            NTT::NTT_internal(&d[0], lg, NTT::InputOutputOrder::NN,
                NTT::Direction::forward, NTT::Type::standard, gpu);
            gpu.DtoH(h_pin, &d[0], domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_pinned) best_pinned = ms;
        }

        // Pinned NR: avoid bit reversal
        float best_pinned_nr = 1e9;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(start, gpu);
            gpu.HtoD(&d[0], h_pin, domain_size);
            NTT::NTT_internal(&d[0], lg, NTT::InputOutputOrder::NR,
                NTT::Direction::forward, NTT::Type::standard, gpu);
            gpu.DtoH(h_pin, &d[0], domain_size);
            cudaEventRecord(stop, gpu); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            if (ms < best_pinned_nr) best_pinned_nr = ms;
        }

        printf("  Pageable NTT_NN:     %8.3f ms\n", best_page);
        printf("  Pinned NTT_NN:       %8.3f ms\n", best_pinned);
        printf("  Pinned NTT_NR:       %8.3f ms\n", best_pinned_nr);
        printf("  Speedup pinned/page: %.1fx\n", best_page / best_pinned);

        free(h_page);
        cudaFreeHost(h_pin);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
