use ntt_cuda::{DeviceNTTData, DeviceBatchNTTData};
use sppark::NTTInputOutputOrder;
use std::time::Instant;

const DEFAULT_GPU: usize = 0;

fn random_bb31_vec(n: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((state >> 33) as u32) % 0x78000001);
    }
    v
}

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn min(times: &[f64]) -> f64 {
    times.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn main() {
    println!("Baby Bear NTT Benchmark — All Optimizations (RTX 5090)");
    println!("=======================================================\n");

    // =========================================================
    // Part 1: Host API comparison (NN vs NR order, with pinned mem)
    // =========================================================
    println!("--- Part 1: Host API (NTT + iNTT round-trip, includes HtoD/DtoH) ---");
    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Size", "NN(ms)", "NR(ms)", "NR_save", "NR/NN");

    for lg in [20, 22, 24, 25, 26, 27] {
        let n = 1usize << lg;
        let base = random_bb31_vec(n);

        let warmup = 3;
        let iters = if lg >= 26 { 3 } else { 5 };

        // NTT_NN via host API
        for _ in 0..warmup {
            let mut v = base.clone();
            ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN);
        }
        let mut times_nn = Vec::new();
        for _ in 0..iters {
            let mut v = base.clone();
            let t = Instant::now();
            ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN);
            times_nn.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        // NTT_NR via host API
        for _ in 0..warmup {
            let mut v = base.clone();
            ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NR);
        }
        let mut times_nr = Vec::new();
        for _ in 0..iters {
            let mut v = base.clone();
            let t = Instant::now();
            ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NR);
            times_nr.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let nn = min(&times_nn);
        let nr = min(&times_nr);
        println!("2^{:>2}   {:>9.3}  {:>9.3}  {:>8.3}ms  {:>8.1}%",
                 lg, nn, nr, nn - nr, (1.0 - nr/nn) * 100.0);
    }

    // =========================================================
    // Part 2: Device API — kernel-only timing (no transfers)
    // =========================================================
    println!("\n--- Part 2: Device API (kernel only, no HtoD/DtoH) ---");
    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Size", "NR(ms)", "NN(ms)", "NR_BW", "Speedup");

    for lg in [20, 22, 24, 25, 26, 27] {
        let n = 1usize << lg;
        let base = random_bb31_vec(n);
        let bytes = n * 4;

        let warmup = 5;
        let iters = if lg >= 26 { 10 } else { 20 };

        // Upload once
        let mut dev = DeviceNTTData::upload(DEFAULT_GPU, &base);

        // Warmup
        for _ in 0..warmup {
            dev.NTT(NTTInputOutputOrder::NR);
        }

        // NR timing
        let mut times_nr = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            dev.NTT(NTTInputOutputOrder::NR);
            times_nr.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        // NN timing
        let mut times_nn = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            dev.NTT(NTTInputOutputOrder::NN);
            times_nn.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let nr = min(&times_nr);
        let nn = min(&times_nn);
        let ntt_passes = if lg <= 10 { 1 } else if lg <= 18 { 2 } else { 3 };
        let nr_bw = (bytes as f64) * 2.0 * (ntt_passes as f64) / (nr * 1e-3) / 1e9;

        // Download to verify (also frees device memory)
        let mut result = vec![0u32; n];
        dev.download(&mut result);

        println!("2^{:>2}   {:>9.3}  {:>9.3}  {:>6.0} GB/s  {:>7.1}x",
                 lg, nr, nn, nr_bw, nn / nr);
    }

    // =========================================================
    // Part 3: Device API — repeated NTT+iNTT (simulating prover)
    // =========================================================
    println!("\n--- Part 3: Repeated NTT+iNTT (simulating prover pipeline) ---");
    println!("{:>6}  {:>12}  {:>12}  {:>12}",
             "Size", "Host(ms/op)", "Device(ms/op)", "Speedup");

    for lg in [22, 24, 26, 27] {
        let n = 1usize << lg;
        let base = random_bb31_vec(n);
        let num_ops = if lg >= 26 { 5 } else { 10 };

        // Host API: each call does HtoD + kernel + DtoH
        let mut v = base.clone();
        // warmup
        ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NR);
        ntt_cuda::iNTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::RN);

        let t = Instant::now();
        for _ in 0..num_ops {
            ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NR);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::RN);
        }
        let host_total = t.elapsed().as_secs_f64() * 1000.0;
        let host_per_op = host_total / (num_ops as f64 * 2.0);

        // Device API: upload once, run all on device, download once
        let mut dev = DeviceNTTData::upload(DEFAULT_GPU, &base);
        // warmup
        dev.NTT(NTTInputOutputOrder::NR);
        dev.iNTT(NTTInputOutputOrder::RN);

        let t = Instant::now();
        for _ in 0..num_ops {
            dev.NTT(NTTInputOutputOrder::NR);
            dev.iNTT(NTTInputOutputOrder::RN);
        }
        let dev_total = t.elapsed().as_secs_f64() * 1000.0;
        let dev_per_op = dev_total / (num_ops as f64 * 2.0);

        let mut result = vec![0u32; n];
        dev.download(&mut result);

        // Verify correctness (NTT+iNTT should return original)
        assert_eq!(result, base, "NTT+iNTT round-trip failed for 2^{}", lg);

        println!("2^{:>2}   {:>10.3}  {:>10.3}    {:>10.1}x",
                 lg, host_per_op, dev_per_op, host_per_op / dev_per_op);
    }

    // =========================================================
    // Part 4: Prover simulation — NTT+iNTT on many columns
    // =========================================================
    // Real prover: uploads all trace columns, does NTT, pointwise constraint
    // evaluation, iNTT, downloads. Compare host API (per-column round-trip)
    // vs batch device API (single upload, all kernels, single download).
    println!("\n--- Part 4: Prover simulation (NTT + iNTT on N columns) ---");
    println!("{:>6} {:>5}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}",
             "Size", "Cols", "Host(ms)", "Batch(ms)", "Up(ms)", "Kern(ms)", "Dn(ms)", "Speedup", "MB");

    for &(lg, ncols) in &[(20, 200), (22, 200), (22, 50), (24, 50)] {
        let n = 1usize << lg;
        let total_elems = n * ncols;
        let total_mb = (total_elems * 4) as f64 / (1024.0 * 1024.0);

        // Generate all columns as one flat buffer
        let mut all_data = Vec::with_capacity(total_elems);
        let mut state: u64 = 0xdeadbeef12345678;
        for _ in 0..total_elems {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            all_data.push(((state >> 33) as u32) % 0x78000001);
        }
        let original = all_data.clone();

        // --- Host API: N separate NTT+iNTT calls (each with HtoD + kernel + DtoH) ---
        // warmup
        {
            let mut col = all_data[..n].to_vec();
            ntt_cuda::NTT(DEFAULT_GPU, &mut col, NTTInputOutputOrder::NR);
        }

        let t = Instant::now();
        for c in 0..ncols {
            let start = c * n;
            let end = start + n;
            ntt_cuda::NTT(DEFAULT_GPU, &mut all_data[start..end], NTTInputOutputOrder::NR);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut all_data[start..end], NTTInputOutputOrder::RN);
        }
        let host_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Verify host round-trip
        assert_eq!(all_data, original, "Host NTT+iNTT round-trip failed");

        // --- Batch device API: 1 upload, NTT all, iNTT all, 1 download ---
        // warmup
        {
            let mut warmup_dev = DeviceBatchNTTData::upload(DEFAULT_GPU, &original[..n], n);
            warmup_dev.NTT_all(NTTInputOutputOrder::NR);
            let mut tmp = vec![0u32; n];
            warmup_dev.download(&mut tmp);
        }

        let t_upload = Instant::now();
        let mut dev = DeviceBatchNTTData::upload(DEFAULT_GPU, &original, n);
        let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;

        let t_kernel = Instant::now();
        dev.NTT_all(NTTInputOutputOrder::NR);
        dev.iNTT_all(NTTInputOutputOrder::RN);
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

        let mut result = vec![0u32; total_elems];
        // Pre-fault pages so DtoH doesn't trigger page faults
        for chunk in result.chunks_mut(1024) {
            chunk[0] = 1;
        }
        let t_download = Instant::now();
        dev.download(&mut result);
        let download_ms = t_download.elapsed().as_secs_f64() * 1000.0;
        let batch_ms = upload_ms + kernel_ms + download_ms;

        // Verify batch round-trip
        assert_eq!(result, original, "Batch NTT+iNTT round-trip failed for 2^{} x {}", lg, ncols);

        println!("2^{:>2}  {:>4}  {:>8.1}  {:>8.1}  {:>6.1}  {:>6.1}  {:>6.1}  {:>6.1}x  {:>5.0}",
                 lg, ncols, host_ms, batch_ms, upload_ms, kernel_ms, download_ms,
                 host_ms / batch_ms, total_mb);
    }

    println!("\nAll correctness checks passed.");
}
