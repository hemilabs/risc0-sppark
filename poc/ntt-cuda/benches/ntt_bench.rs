// BabyBear NTT benchmark on CUDA
// Measures total time (includes PCIe transfer) and reports both
// total and estimated kernel-only time.
use sppark::NTTInputOutputOrder;
use std::time::Instant;

const GPU: usize = 0;
const WARMUP: usize = 50;
const SAMPLES: usize = 15;

// Measured PCIe 4.0 x16 round-trip bandwidth: ~26.5 GB/s
const PCIE_BW_GBS: f64 = 26.5;

fn random_bb31_vec(n: usize) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let mut h = DefaultHasher::new();
        (i as u64 ^ 0xdeadbeef).hash(&mut h);
        v.push((h.finish() as u32) % 0x78000001);
    }
    v
}

fn main() {
    println!("====================================================");
    println!("  sppark CUDA NTT Benchmark — BabyBear (RTX 4090)");
    println!("====================================================");
    println!();
    println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "lg_n", "N", "Total(ms)", "PCIe(ms)", "Kern(ms)", "G bfly/s");

    for lg_n in [10u32, 12, 14, 16, 18, 20, 22, 24] {
        let n = 1usize << lg_n;
        let mut data = random_bb31_vec(n);

        // Warm up
        for _ in 0..WARMUP {
            ntt_cuda::NTT(GPU, &mut data, NTTInputOutputOrder::NN);
        }

        // Timed samples
        let mut best_ns = u128::MAX;
        for _ in 0..SAMPLES {
            let start = Instant::now();
            ntt_cuda::NTT(GPU, &mut data, NTTInputOutputOrder::NN);
            let elapsed = start.elapsed().as_nanos();
            if elapsed < best_ns {
                best_ns = elapsed;
            }
        }

        let total_ms = best_ns as f64 / 1e6;
        let bytes_rt = 2.0 * n as f64 * 4.0; // round-trip: HtoD + DtoH
        let pcie_ms = bytes_rt / (PCIE_BW_GBS * 1e9) * 1e3;
        let kern_ms = (total_ms - pcie_ms).max(0.01);
        let total_bfly = (n as f64 / 2.0) * lg_n as f64;
        let g_bfly = total_bfly / (kern_ms * 1e6);

        println!(
            "{:>6} {:>10} {:>10.3} {:>10.3} {:>10.3} {:>10.1}",
            lg_n, n, total_ms, pcie_ms, kern_ms, g_bfly
        );
    }
    println!();
}
