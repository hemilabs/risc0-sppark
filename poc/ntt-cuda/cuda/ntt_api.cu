// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
# include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
# include <ff/pasta.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#elif defined(FEATURE_GOLDILOCKS)
# include <ff/goldilocks.hpp>
#elif defined(FEATURE_BABY_BEAR)
# include <ff/baby_bear.hpp>
#else
# error "no FEATURE"
#endif

#include <ntt/ntt.cuh>

SPPARK_FFI
RustError::by_value compute_ntt(size_t device_id,
                                fr_t* inout, uint32_t lg_domain_size,
                                NTT::InputOutputOrder ntt_order,
                                NTT::Direction ntt_direction,
                                NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}

// Device-pointer API: operates on data already on the GPU.
// Caller is responsible for device memory allocation and transfers.
SPPARK_FFI
RustError::by_value compute_ntt_on_device(size_t device_id,
                                          fr_t* d_inout,
                                          uint32_t lg_domain_size,
                                          NTT::InputOutputOrder ntt_order,
                                          NTT::Direction ntt_direction,
                                          NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    if (lg_domain_size == 0)
        return RustError{cudaSuccess};

    try {
        gpu.select();
        NTT::Base_dev_ptr(gpu, d_inout, lg_domain_size,
                          ntt_order, ntt_direction, ntt_type);
        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Allocate device memory and copy host data to it.
// Returns a device pointer via *d_out.
SPPARK_FFI
RustError::by_value ntt_alloc_and_upload(size_t device_id,
                                         fr_t** d_out,
                                         const fr_t* h_in,
                                         size_t nelems)
{
    try {
        auto& gpu = select_gpu(device_id);
        gpu.select();
        CUDA_OK(cudaMallocAsync(d_out, nelems * sizeof(fr_t), gpu));
        CUDA_OK(cudaMemcpyAsync(*d_out, h_in, nelems * sizeof(fr_t),
                                cudaMemcpyHostToDevice, gpu));
        gpu.sync();
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

// Download device data to host and free device memory.
SPPARK_FFI
RustError::by_value ntt_download_and_free(size_t device_id,
                                          fr_t* h_out,
                                          fr_t* d_in,
                                          size_t nelems)
{
    try {
        auto& gpu = select_gpu(device_id);
        gpu.select();
        CUDA_OK(cudaMemcpyAsync(h_out, d_in, nelems * sizeof(fr_t),
                                cudaMemcpyDeviceToHost, gpu));
        gpu.sync();
        CUDA_OK(cudaFreeAsync(d_in, gpu));
        gpu.sync();
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

// Batch API: allocate device memory for ncols columns and upload all at once.
// h_in points to ncols*nelems contiguous elements.
// Uses pinned transfers via gpu stream for maximum PCIe throughput.
SPPARK_FFI
RustError::by_value ntt_batch_alloc_and_upload(size_t device_id,
                                               fr_t** d_out,
                                               const fr_t* h_in,
                                               size_t nelems,
                                               size_t ncols)
{
    try {
        auto& gpu = select_gpu(device_id);
        gpu.select();
        size_t total = nelems * ncols;
        size_t bytes = total * sizeof(fr_t);
        bool pinned = (cudaHostRegister((void*)h_in, bytes,
                                        cudaHostRegisterReadOnly) == cudaSuccess);
        CUDA_OK(cudaMalloc(d_out, bytes));
        gpu.HtoD(*d_out, h_in, total);
        gpu.sync();
        if (pinned)
            cudaHostUnregister((void*)h_in);
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

// Batch NTT: apply NTT to each of ncols columns sequentially.
// d_inout points to ncols contiguous columns of 2^lg_domain_size elements each.
SPPARK_FFI
RustError::by_value ntt_batch_compute(size_t device_id,
                                      fr_t* d_inout,
                                      uint32_t lg_domain_size,
                                      size_t ncols,
                                      NTT::InputOutputOrder ntt_order,
                                      NTT::Direction ntt_direction,
                                      NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    if (lg_domain_size == 0)
        return RustError{cudaSuccess};

    try {
        gpu.select();
        size_t nelems = (size_t)1 << lg_domain_size;
        for (size_t i = 0; i < ncols; i++) {
            NTT::Base_dev_ptr(gpu, d_inout + i * nelems, lg_domain_size,
                              ntt_order, ntt_direction, ntt_type);
        }
        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Batch download and free.
// NOTE: Host buffer must have pre-faulted pages (touch before calling)
// to avoid page fault overhead during DtoH transfer.
SPPARK_FFI
RustError::by_value ntt_batch_download_and_free(size_t device_id,
                                                fr_t* h_out,
                                                fr_t* d_in,
                                                size_t nelems,
                                                size_t ncols)
{
    try {
        auto& gpu = select_gpu(device_id);
        gpu.select();
        size_t total = nelems * ncols;
        size_t bytes = total * sizeof(fr_t);
        gpu.sync();
        CUDA_OK(cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(d_in));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

// ===================================================================
// RISC Zero batch API: operates on device-resident data using a
// caller-provided CUstream. No PCIe transfers, no implicit sync.
// ===================================================================

// Forward NTT, RN ordering (bit-reversed in -> natural out, CT algorithm)
SPPARK_FFI
RustError::by_value sppark_batch_NTT(cudaStream_t cuda_stream, fr_t* d_inout,
                                     uint32_t lg_domain_size,
                                     uint32_t poly_count)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return RustError{cudaSuccess};

    try {
        (void)cudaGetLastError();  // clear any stale errors
        auto& gpu = select_gpu(-1);
        stream_t stream(cuda_stream, gpu.id());
        size_t domain_size = (size_t)1 << lg_domain_size;
        for (uint32_t i = 0; i < poly_count; i++) {
            NTT::Base_dev_ptr(stream, d_inout + i * domain_size,
                              lg_domain_size, NTT::InputOutputOrder::RN,
                              NTT::Direction::forward, NTT::Type::standard);
        }
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Inverse NTT, NR ordering (natural in -> bit-reversed out, GS algorithm)
SPPARK_FFI
RustError::by_value sppark_batch_iNTT(cudaStream_t cuda_stream, fr_t* d_inout,
                                      uint32_t lg_domain_size,
                                      uint32_t poly_count)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return RustError{cudaSuccess};

    try {
        (void)cudaGetLastError();  // clear any stale errors
        auto& gpu = select_gpu(-1);
        stream_t stream(cuda_stream, gpu.id());
        size_t domain_size = (size_t)1 << lg_domain_size;
        for (uint32_t i = 0; i < poly_count; i++) {
            NTT::Base_dev_ptr(stream, d_inout + i * domain_size,
                              lg_domain_size, NTT::InputOutputOrder::NR,
                              NTT::Direction::inverse, NTT::Type::standard);
        }
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Batched LDE expand kernel: zero-pad input into expanded output.
// All poly_count columns are processed in a single launch via blockIdx.y.
// Uses vectorized stores for coalesced writes (16-byte uint4 for 4x blowup).
__launch_bounds__(256) __global__
void batch_expand_kernel(fr_t* __restrict__ d_out,
                         const fr_t* __restrict__ d_in,
                         uint32_t lg_domain_size, uint32_t lg_blowup)
{
    const size_t domain_size = (size_t)1 << lg_domain_size;
    const size_t ext_size = domain_size << lg_blowup;
    const uint32_t col = blockIdx.y;

    // Per-column base pointers
    const uint32_t* in  = (const uint32_t*)(d_in  + col * domain_size);
    uint32_t* out       = (uint32_t*)(d_out + col * ext_size);

    if (lg_blowup == 2) {
        // 4x blowup: write {val, 0, 0, 0} as a single uint4 (16-byte coalesced store)
        uint4* out4 = (uint4*)out;
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < domain_size;
             idx += gridDim.x * (size_t)blockDim.x)
        {
            out4[idx] = make_uint4(in[idx], 0, 0, 0);
        }
    } else if (lg_blowup == 1) {
        // 2x blowup: write {val, 0} as uint2 (8-byte coalesced store)
        uint2* out2 = (uint2*)out;
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < domain_size;
             idx += gridDim.x * (size_t)blockDim.x)
        {
            out2[idx] = make_uint2(in[idx], 0);
        }
    } else if (lg_blowup == 3) {
        // 8x blowup: write {val, 0,0,0, 0,0,0,0} as 2x uint4 stores
        uint4* out4 = (uint4*)out;
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < domain_size;
             idx += gridDim.x * (size_t)blockDim.x)
        {
            out4[idx * 2]     = make_uint4(in[idx], 0, 0, 0);
            out4[idx * 2 + 1] = make_uint4(0, 0, 0, 0);
        }
    } else {
        // General case: scalar stores
        const uint32_t blowup = 1u << lg_blowup;
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < domain_size;
             idx += gridDim.x * (size_t)blockDim.x)
        {
            size_t out_base = idx << lg_blowup;
            out[out_base] = in[idx];
            for (uint32_t j = 1; j < blowup; j++)
                out[out_base + j] = 0;
        }
    }
}

// LDE expansion: bit-reversed input -> zero-padded expanded output
SPPARK_FFI
RustError::by_value sppark_batch_expand(cudaStream_t cuda_stream,
                                        fr_t* d_out, const fr_t* d_in,
                                        uint32_t lg_domain_size,
                                        uint32_t lg_blowup,
                                        uint32_t poly_count)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return RustError{cudaSuccess};

    try {
        (void)cudaGetLastError();  // clear any stale errors
        size_t domain_size = (size_t)1 << lg_domain_size;

        // Use batched kernel: one launch for all columns via blockIdx.y
        uint32_t block_size = 256;
        uint32_t num_blocks = (domain_size + block_size - 1) / block_size;
        // Cap grid.x to avoid excessive blocks
        if (num_blocks > 1024)
            num_blocks = 1024;
        dim3 grid(num_blocks, poly_count);
        batch_expand_kernel<<<grid, block_size, 0, cuda_stream>>>
            (d_out, d_in, lg_domain_size, lg_blowup);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Batched coset shift kernel: multiply each element by the appropriate
// generator power. Uses blockIdx.y for columns — single launch for all polys.
__launch_bounds__(1024) __global__
void batch_shift_kernel(fr_t* __restrict__ d_inout,
                        uint32_t lg_domain_size,
                        const fr_t (*gen_powers)[WINDOW_SIZE])
{
    const size_t domain_size = (size_t)1 << lg_domain_size;
    fr_t* data = d_inout + blockIdx.y * domain_size;

    for (index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
         idx < domain_size;
         idx += blockDim.x * gridDim.x)
    {
        fr_t r = data[idx];
        index_t pow = bit_rev(idx, lg_domain_size);
        r *= get_intermediate_root(pow, gen_powers);
        data[idx] = r;
    }
}

// Coset shift: multiply each element by the appropriate generator power
SPPARK_FFI
RustError::by_value sppark_batch_zk_shift(cudaStream_t cuda_stream,
                                          fr_t* d_inout,
                                          uint32_t lg_domain_size,
                                          uint32_t poly_count)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return RustError{cudaSuccess};

    try {
        (void)cudaGetLastError();  // clear any stale errors
        auto& gpu = select_gpu(-1);
        const auto gen_powers =
            NTTParameters::all(false)[gpu.id()].partial_group_gen_powers;

        uint32_t num_blocks_x = gpu.sm_count();
        dim3 grid(num_blocks_x, poly_count);
        batch_shift_kernel<<<grid, 1024, 0, cuda_stream>>>
            (d_inout, lg_domain_size, gen_powers);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// In-place bit-reversal permutation (per-column loop using tiled kernel)
SPPARK_FFI
RustError::by_value sppark_batch_bit_reverse(cudaStream_t cuda_stream,
                                             fr_t* d_inout,
                                             uint32_t lg_domain_size,
                                             uint32_t poly_count)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return RustError{cudaSuccess};

    try {
        (void)cudaGetLastError();  // clear any stale errors
        auto& gpu = select_gpu(-1);
        stream_t stream(cuda_stream, gpu.id());
        size_t domain_size = (size_t)1 << lg_domain_size;
        for (uint32_t i = 0; i < poly_count; i++) {
            NTT::bit_rev_dev_ptr(stream, d_inout + i * domain_size,
                                 lg_domain_size);
        }
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// ===================================================================
// Stream and device memory helpers for Rust benchmarks
// ===================================================================

SPPARK_FFI
RustError::by_value sppark_create_stream(cudaStream_t* out)
{
    try {
        CUDA_OK(cudaStreamCreateWithFlags(out, cudaStreamNonBlocking));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

SPPARK_FFI
RustError::by_value sppark_sync_stream(cudaStream_t stream)
{
    try {
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

SPPARK_FFI void sppark_destroy_stream(cudaStream_t stream)
{
    (void)cudaStreamDestroy(stream);
}

SPPARK_FFI
RustError::by_value sppark_alloc_gpu(void** d_out, size_t bytes)
{
    try {
        CUDA_OK(cudaMalloc(d_out, bytes));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

SPPARK_FFI void sppark_free_gpu(void* d_ptr)
{
    (void)cudaFree(d_ptr);
}

SPPARK_FFI
RustError::by_value sppark_htod_on_stream(void* d_dst, const void* h_src,
                                          size_t bytes,
                                          cudaStream_t stream)
{
    try {
        CUDA_OK(cudaMemcpyAsync(d_dst, h_src, bytes,
                                cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}

SPPARK_FFI
RustError::by_value sppark_dtoh_on_stream(void* h_dst, const void* d_src,
                                          size_t bytes,
                                          cudaStream_t stream)
{
    try {
        CUDA_OK(cudaMemcpyAsync(h_dst, d_src, bytes,
                                cudaMemcpyDeviceToHost, stream));
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};
}
