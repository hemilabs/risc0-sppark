use core::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

#[allow(unused_imports)]
use ntt_cuda::{
    sppark_alloc_gpu, sppark_batch_NTT, sppark_batch_bit_reverse, sppark_batch_expand,
    sppark_batch_iNTT, sppark_batch_zk_shift, sppark_create_stream, sppark_destroy_stream,
    sppark_dtoh_on_stream, sppark_free_gpu, sppark_htod_on_stream, sppark_sync_stream,
    DeviceBatchNTTData,
};
use sppark::NTTInputOutputOrder;

const ELEM_SZ: usize = 4; // baby_bear element size in bytes

fn check(err: sppark::Error) {
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

fn random_bb31_vec(n: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((state >> 33) as u32) % 0x78000001);
    }
    v
}

fn min_of(times: &[f64]) -> f64 {
    times.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn median_of(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

/// Test iNTT(NR) -> NTT(RN) round-trip via batch API
fn test_round_trip(stream: *mut c_void) {
    let lg: u32 = 16;
    let n = 1usize << lg;
    let ncols: u32 = 4;
    let total = n * ncols as usize;

    let original = random_bb31_vec(total);

    let dev = DeviceBatchNTTData::upload(0, &original, n);
    let d_ptr = dev.as_mut_ptr();

    check(unsafe { sppark_batch_iNTT(stream, d_ptr, lg, ncols) });
    check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) });
    check(unsafe { sppark_sync_stream(stream) });

    let mut result = vec![0u32; total];
    dev.download(&mut result);
    assert_eq!(result, original, "iNTT+NTT round-trip failed");
    print!("  round-trip OK");
}

/// Test bit_reverse involution (apply twice = identity)
fn test_bit_reverse_involution(stream: *mut c_void) {
    let lg: u32 = 16;
    let n = 1usize << lg;
    let ncols: u32 = 4;
    let total = n * ncols as usize;

    let original = random_bb31_vec(total);
    let dev = DeviceBatchNTTData::upload(0, &original, n);
    let d_ptr = dev.as_mut_ptr();

    check(unsafe { sppark_batch_bit_reverse(stream, d_ptr, lg, ncols) });
    check(unsafe { sppark_batch_bit_reverse(stream, d_ptr, lg, ncols) });
    check(unsafe { sppark_sync_stream(stream) });

    let mut result = vec![0u32; total];
    dev.download(&mut result);
    assert_eq!(result, original, "bit_reverse involution failed");
    print!(", bit_reverse OK");
}

/// Measure per-column cost at fixed size, varying ncols
fn profile_scaling(stream: *mut c_void, lg: u32) {
    let n = 1usize << lg;
    let max_ncols: u32 = 200;
    let total = n * max_ncols as usize;
    let data = random_bb31_vec(total);

    let dev = DeviceBatchNTTData::upload(0, &data, n);
    let d_ptr = dev.as_mut_ptr();

    // Warmup
    for _ in 0..5 {
        check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, max_ncols) });
        check(unsafe { sppark_sync_stream(stream) });
    }

    let iters = 10;

    println!("  2^{} per-column scaling (NTT RN):", lg);
    println!("  {:>6}  {:>10}  {:>10}  {:>10}", "Cols", "Total(ms)", "Per-col(us)", "Marginal(us)");

    let mut prev_time = 0.0;
    for &ncols in &[1u32, 2, 4, 8, 16, 32, 50, 100, 150, 200] {
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) });
            check(unsafe { sppark_sync_stream(stream) });
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let best = min_of(&times);
        let per_col = best * 1000.0 / ncols as f64;
        let marginal = if ncols == 1 {
            per_col
        } else {
            let prev_ncols = match ncols {
                2 => 1, 4 => 2, 8 => 4, 16 => 8, 32 => 16,
                50 => 32, 100 => 50, 150 => 100, 200 => 150,
                _ => 1
            };
            (best - prev_time) * 1000.0 / (ncols - prev_ncols) as f64
        };
        println!("  {:>6}  {:>10.3}  {:>10.1}  {:>10.1}", ncols, best, per_col, marginal);
        prev_time = best;
    }

    std::mem::forget(dev);
    unsafe { sppark_free_gpu(d_ptr) };
}

/// Measure per-operation cost for each batch function
fn profile_operations(stream: *mut c_void, lg: u32, ncols: u32) {
    let n = 1usize << lg;
    let total = n * ncols as usize;
    let data = random_bb31_vec(total);

    let dev = DeviceBatchNTTData::upload(0, &data, n);
    let d_ptr = dev.as_mut_ptr();

    let warmup = 5;
    let iters = 15;

    // Warmup all operations
    for _ in 0..warmup {
        check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_batch_iNTT(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_batch_bit_reverse(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_batch_zk_shift(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
    }

    let ops: &[(&str, Box<dyn Fn()>)] = &[
        ("NTT(RN)", Box::new(|| check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) }))),
        ("iNTT(NR)", Box::new(|| check(unsafe { sppark_batch_iNTT(stream, d_ptr, lg, ncols) }))),
        ("BitRev", Box::new(|| check(unsafe { sppark_batch_bit_reverse(stream, d_ptr, lg, ncols) }))),
        ("Shift", Box::new(|| check(unsafe { sppark_batch_zk_shift(stream, d_ptr, lg, ncols) }))),
    ];

    let bytes_per_col = n * ELEM_SZ;
    let ntt_passes = if lg <= 10 { 1 } else if lg <= 18 { 2 } else if lg <= 30 { 3 } else { 4 };

    println!("  2^{} x {} cols:", lg, ncols);
    println!("  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Op", "Total(ms)", "Per-col(us)", "Bytes(MB)", "EffBW(GB/s)");

    for (name, op) in ops {
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            op();
            check(unsafe { sppark_sync_stream(stream) });
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let best = min_of(&times);
        let per_col = best * 1000.0 / ncols as f64;
        let total_bytes = bytes_per_col as f64 * ncols as f64;
        // NTT/iNTT: 2 * passes * total_bytes (read+write per pass)
        // BitRev/Shift: 2 * total_bytes (single read+write)
        let rw_bytes = if name.starts_with("NTT") || name.starts_with("iNTT") {
            2.0 * ntt_passes as f64 * total_bytes
        } else {
            2.0 * total_bytes
        };
        let bw = rw_bytes / (best * 1e-3) / 1e9;
        println!("  {:>10}  {:>10.3}  {:>10.1}  {:>10.1}  {:>10.0}",
                 name, best, per_col, total_bytes / 1e6, bw);
    }

    std::mem::forget(dev);
    unsafe { sppark_free_gpu(d_ptr) };
}

/// Profile expand kernel: varies blowup factor and measures bandwidth
fn profile_expand(stream: *mut c_void, lg: u32, ncols: u32) {
    let dom_size = 1usize << lg;
    let total_in = dom_size * ncols as usize;

    let data = random_bb31_vec(total_in);

    let mut d_in: *mut c_void = null_mut();
    check(unsafe { sppark_alloc_gpu(&mut d_in, total_in * ELEM_SZ) });
    check(unsafe {
        sppark_htod_on_stream(d_in, data.as_ptr() as *const _, total_in * ELEM_SZ, stream)
    });

    // First do iNTT to get bit-reversed coefficients (expand expects bit-reversed input)
    check(unsafe { sppark_batch_iNTT(stream, d_in, lg, ncols as u32) });
    check(unsafe { sppark_sync_stream(stream) });

    let iters = 10;

    println!("  2^{} x {} cols expand profiling:", lg, ncols);
    println!("  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Blowup", "Total(ms)", "Per-col(us)", "Write(MB)", "WriteBW(GB/s)");

    for &lg_blowup in &[1u32, 2, 3] {
        let ext_size = 1usize << (lg + lg_blowup);
        let total_out = ext_size * ncols as usize;

        let mut d_out: *mut c_void = null_mut();
        check(unsafe { sppark_alloc_gpu(&mut d_out, total_out * ELEM_SZ) });

        // Warmup
        for _ in 0..3 {
            check(unsafe {
                sppark_batch_expand(stream, d_out, d_in as *const _, lg, lg_blowup, ncols as u32)
            });
            check(unsafe { sppark_sync_stream(stream) });
        }

        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            check(unsafe {
                sppark_batch_expand(stream, d_out, d_in as *const _, lg, lg_blowup, ncols as u32)
            });
            check(unsafe { sppark_sync_stream(stream) });
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let best = min_of(&times);
        let per_col = best * 1000.0 / ncols as f64;
        let write_bytes = total_out as f64 * ELEM_SZ as f64;
        let write_bw = write_bytes / (best * 1e-3) / 1e9;

        println!("  {:>5}x  {:>10.3}  {:>10.1}  {:>10.1}  {:>10.0}",
                 1u32 << lg_blowup, best, per_col, write_bytes / 1e6, write_bw);

        unsafe { sppark_free_gpu(d_out) };
    }

    unsafe { sppark_free_gpu(d_in) };
}

/// Measure kernel launch overhead: sync after every column vs all columns then sync
fn profile_launch_overhead(stream: *mut c_void, lg: u32) {
    let n = 1usize << lg;
    let ncols = 100u32;
    let total = n * ncols as usize;
    let data = random_bb31_vec(total);

    let dev = DeviceBatchNTTData::upload(0, &data, n);
    let d_ptr = dev.as_mut_ptr();

    let warmup = 3;
    let iters = 10;

    // Warmup
    for _ in 0..warmup {
        check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
    }

    // Method 1: All columns in one batch call (launches 3*ncols kernels, sync once)
    let mut times_batch = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        check(unsafe { sppark_batch_NTT(stream, d_ptr, lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
        times_batch.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // Method 2: Sync after EACH column (advancing pointer correctly)
    let mut times_sync_each = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for i in 0..ncols {
            let col_ptr = unsafe { (d_ptr as *mut u8).add(i as usize * n * ELEM_SZ) as *mut c_void };
            check(unsafe { sppark_batch_NTT(stream, col_ptr, lg, 1) });
            check(unsafe { sppark_sync_stream(stream) });
        }
        times_sync_each.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // Method 3: Launch 1 col at a time but sync only at end (advancing pointer correctly)
    let mut times_loop_nosync = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for i in 0..ncols {
            let col_ptr = unsafe { (d_ptr as *mut u8).add(i as usize * n * ELEM_SZ) as *mut c_void };
            check(unsafe { sppark_batch_NTT(stream, col_ptr, lg, 1) });
        }
        check(unsafe { sppark_sync_stream(stream) });
        times_loop_nosync.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let batch = min_of(&times_batch);
    let sync_each = min_of(&times_sync_each);
    let loop_nosync = min_of(&times_loop_nosync);
    let overhead_per_sync = (sync_each - batch) / ncols as f64 * 1000.0;
    let overhead_per_call = (loop_nosync - batch) / ncols as f64 * 1000.0;

    println!("  2^{} x {} cols:", lg, ncols);
    println!("    Batch (C++ loop, 1 sync):    {:>8.3} ms", batch);
    println!("    Rust loop (sync each col):   {:>8.3} ms", sync_each);
    println!("    Rust loop (sync at end):     {:>8.3} ms", loop_nosync);
    println!("    Per-sync overhead:           {:>8.1} us", overhead_per_sync);
    println!("    Per-call overhead (Rust→C):  {:>8.1} us", overhead_per_call);

    std::mem::forget(dev);
    unsafe { sppark_free_gpu(d_ptr) };
}

/// Full LDE pipeline timing
fn time_lde_pipeline(stream: *mut c_void, lg: u32, ncols: u32, lg_blowup: u32) {
    let dom_size = 1usize << lg;
    let ext_size = 1usize << (lg + lg_blowup);
    let total_in = dom_size * ncols as usize;
    let total_out = ext_size * ncols as usize;
    let data = random_bb31_vec(total_in);

    let mut d_in: *mut c_void = null_mut();
    let mut d_out: *mut c_void = null_mut();
    check(unsafe { sppark_alloc_gpu(&mut d_in, total_in * ELEM_SZ) });
    check(unsafe { sppark_alloc_gpu(&mut d_out, total_out * ELEM_SZ) });

    check(unsafe {
        sppark_htod_on_stream(d_in, data.as_ptr() as *const _, total_in * ELEM_SZ, stream)
    });
    check(unsafe { sppark_sync_stream(stream) });

    let warmup = 3;
    let iters = if lg >= 24 { 3 } else { 5 };

    let ext_lg = lg + lg_blowup;

    for _ in 0..warmup {
        check(unsafe {
            sppark_htod_on_stream(d_in, data.as_ptr() as *const _, total_in * ELEM_SZ, stream)
        });
        check(unsafe { sppark_batch_iNTT(stream, d_in, lg, ncols) });
        check(unsafe {
            sppark_batch_expand(stream, d_out, d_in as *const _, lg, lg_blowup, ncols)
        });
        check(unsafe { sppark_batch_NTT(stream, d_out, ext_lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
    }

    let mut times_intt = Vec::new();
    let mut times_expand = Vec::new();
    let mut times_ntt = Vec::new();
    let mut times_total = Vec::new();

    for _ in 0..iters {
        check(unsafe {
            sppark_htod_on_stream(d_in, data.as_ptr() as *const _, total_in * ELEM_SZ, stream)
        });
        check(unsafe { sppark_sync_stream(stream) });

        let t_total = Instant::now();

        let t = Instant::now();
        check(unsafe { sppark_batch_iNTT(stream, d_in, lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
        times_intt.push(t.elapsed().as_secs_f64() * 1000.0);

        let t = Instant::now();
        check(unsafe {
            sppark_batch_expand(stream, d_out, d_in as *const _, lg, lg_blowup, ncols)
        });
        check(unsafe { sppark_sync_stream(stream) });
        times_expand.push(t.elapsed().as_secs_f64() * 1000.0);

        let t = Instant::now();
        check(unsafe { sppark_batch_NTT(stream, d_out, ext_lg, ncols) });
        check(unsafe { sppark_sync_stream(stream) });
        times_ntt.push(t.elapsed().as_secs_f64() * 1000.0);

        times_total.push(t_total.elapsed().as_secs_f64() * 1000.0);
    }

    unsafe {
        sppark_free_gpu(d_in);
        sppark_free_gpu(d_out);
    }

    println!(
        "  2^{:>2} x {:>3}: iNTT {:>7.2}ms  expand {:>7.2}ms  NTT(2^{}) {:>7.2}ms  total {:>7.2}ms",
        lg, ncols,
        min_of(&times_intt),
        min_of(&times_expand),
        ext_lg,
        min_of(&times_ntt),
        min_of(&times_total),
    );
}

fn main() {
    println!("RISC Zero Batch NTT Profiling (Baby Bear, RTX 5090)");
    println!("====================================================\n");

    // Initialize GPU
    let mut warmup = [1u32, 2];
    ntt_cuda::NTT(0, &mut warmup, NTTInputOutputOrder::NN);

    let mut stream: *mut c_void = null_mut();
    check(unsafe { sppark_create_stream(&mut stream) });

    // === Correctness ===
    print!("Correctness:");
    test_round_trip(stream);
    test_bit_reverse_involution(stream);
    println!(" -- all passed\n");

    // === 1. Per-column scaling ===
    println!("=== 1. Per-column scaling (how does cost scale with ncols?) ===");
    profile_scaling(stream, 20);
    println!();
    profile_scaling(stream, 22);
    println!();

    // === 2. Per-operation cost breakdown ===
    println!("=== 2. Per-operation cost breakdown ===");
    profile_operations(stream, 20, 200);
    println!();
    profile_operations(stream, 22, 200);
    println!();
    profile_operations(stream, 24, 50);
    println!();

    // === 3. Kernel launch overhead ===
    println!("=== 3. Kernel launch overhead measurement ===");
    profile_launch_overhead(stream, 20);
    println!();
    profile_launch_overhead(stream, 22);
    println!();

    // === 4. Expand kernel profiling ===
    println!("=== 4. Expand kernel profiling ===");
    profile_expand(stream, 20, 200);
    println!();
    profile_expand(stream, 22, 200);
    println!();

    // === 5. Full LDE pipeline ===
    println!("=== 5. Full LDE pipeline: iNTT -> expand(4x) -> NTT ===");
    for &(lg, ncols) in &[(20u32, 50u32), (20, 200), (22, 50), (22, 200), (24, 50)] {
        time_lde_pipeline(stream, lg, ncols, 2);
    }

    unsafe { sppark_destroy_stream(stream) };

    println!("\nDone.");
}
