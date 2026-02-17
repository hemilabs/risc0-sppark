// Quick profiling test: run preloaded G1 MSM at 2^23 a few times
use ark_bn254::G1Affine;
use ark_ff::BigInteger256;
use msm_cuda::*;

fn main() {
    let npow = 23;
    let npoints = 1usize << npow;
    eprintln!("Generating {} points...", npoints);

    let (points, scalars) = util::generate_points_scalars::<G1Affine>(npoints);
    let scalars_bi = unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    };

    eprintln!("Initializing preloaded MSM (GLV)...");
    let msm = PreloadedMsm::init_g1(&points);

    eprintln!("Running 3 invocations...");
    for i in 0..3 {
        let _result = msm.invoke_g1(scalars_bi);
        eprintln!("  Invocation {} done", i);
    }

    drop(msm);
    eprintln!("Done.");
}
