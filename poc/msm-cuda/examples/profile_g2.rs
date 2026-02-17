// Quick profiling test: run preloaded G2 MSM at 2^23 a few times
use ark_bn254::G2Affine;
use ark_ff::BigInteger256;
use msm_cuda::*;

fn main() {
    let npow = 23;
    let npoints = 1usize << npow;
    eprintln!("Generating {} G2 points...", npoints);

    let (points, scalars) = util::generate_points_scalars::<G2Affine>(npoints);
    let scalars_bi = unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    };

    eprintln!("Initializing preloaded MSM (G2)...");
    let msm = PreloadedMsm::init_g2(&points);

    // Warmup (first call also precomputes digits)
    eprintln!("Warmup...");
    for _ in 0..3 {
        let _ = msm.invoke_g2(scalars_bi);
    }

    // Profiling runs
    let nruns = 10;
    eprintln!("\nG2 profiling ({} runs):", nruns);
    let start = std::time::Instant::now();
    for _ in 0..nruns {
        let _ = msm.invoke_g2(scalars_bi);
    }
    let elapsed = start.elapsed();
    eprintln!("\nG2 wall-clock average: {:.3} ms\n", elapsed.as_secs_f64() * 1000.0 / nruns as f64);

    drop(msm);
}
