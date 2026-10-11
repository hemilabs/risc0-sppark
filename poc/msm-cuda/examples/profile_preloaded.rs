// Minimal binary for nsys/ncu profiling of preloaded MSM (RISC Zero pattern)
use ark_bn254::G1Affine;
use ark_ff::BigInteger256;
use msm_cuda::*;
use std::str::FromStr;

fn main() {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("23".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();
    let npoints = 1usize << npoints_npow;

    eprintln!("Generating {} points and scalars...", npoints);
    let (points, scalars) = util::generate_points_scalars::<G1Affine>(npoints);

    eprintln!("Pre-loading points to GPU...");
    let msm = PreloadedMsm::init_g1(&points);

    let scalars_bi = unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    };

    // Warmup
    eprintln!("Warmup...");
    for _ in 0..3 {
        let _ = msm.invoke_g1(scalars_bi);
    }

    // Timed runs
    let nruns = 10;
    eprintln!("Running {} iterations...", nruns);
    let start = std::time::Instant::now();
    for _ in 0..nruns {
        let _ = msm.invoke_g1(scalars_bi);
    }
    let elapsed = start.elapsed();
    eprintln!("Average: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / nruns as f64);

    drop(msm);
}
