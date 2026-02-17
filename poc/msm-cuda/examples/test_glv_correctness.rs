// Correctness test for MSM paths vs arkworks reference
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::{msm::VariableBaseMSM, ProjectiveCurve};
use ark_ff::{BigInteger256, PrimeField};
use msm_cuda::*;

fn main() {
    for npow in [8, 12, 16] {
        let npoints = 1usize << npow;
        eprintln!("Testing correctness with {} points...", npoints);

        let (points, scalars) = util::generate_points_scalars::<G1Affine>(npoints);

        // Transmuted Montgomery-form scalars (what the FFI receives)
        let scalars_bi = unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        };

        // References:
        // Raw scalars (true scalar values)
        let scalars_repr: Vec<_> = scalars.iter().map(|s| s.into_repr()).collect();
        let ref_raw = VariableBaseMSM::multi_scalar_mul(&points, &scalars_repr);
        let ref_raw_affine = ref_raw.into_affine();

        // Montgomery-form scalars (what mont=false computes with)
        let ref_mont = VariableBaseMSM::multi_scalar_mul(&points, scalars_bi);
        let ref_mont_affine = ref_mont.into_affine();

        // Test 1: non-preloaded path (mult_pippenger_inf, mont=false)
        // This computes with Montgomery scalars directly
        let gpu_result = multi_scalar_mult_arkworks(&points, scalars_bi);
        let gpu_affine = gpu_result.into_affine();

        if gpu_affine == ref_mont_affine {
            eprintln!("  PASS non-preloaded (matches mont ref): 2^{}", npow);
        } else if gpu_affine == ref_raw_affine {
            eprintln!("  PASS non-preloaded (matches raw ref): 2^{}", npow);
        } else {
            eprintln!("  FAIL non-preloaded: 2^{}", npow);
            eprintln!("  GPU:       {:?}", gpu_affine);
            eprintln!("  Ref (mont):{:?}", ref_mont_affine);
            eprintln!("  Ref (raw): {:?}", ref_raw_affine);
            std::process::exit(1);
        }

        // Test 2: preloaded path (also uses mont=false)
        let msm = PreloadedMsm::init_g1(&points);
        let gpu_preloaded = msm.invoke_g1(scalars_bi);
        drop(msm);
        let pl_affine = gpu_preloaded.into_affine();

        if pl_affine == ref_mont_affine {
            eprintln!("  PASS preloaded (matches mont ref): 2^{}", npow);
        } else if pl_affine == ref_raw_affine {
            eprintln!("  PASS preloaded (matches raw ref): 2^{}", npow);
        } else {
            eprintln!("  FAIL preloaded: 2^{}", npow);
            eprintln!("  GPU:       {:?}", pl_affine);
            eprintln!("  Ref (mont):{:?}", ref_mont_affine);
            eprintln!("  Ref (raw): {:?}", ref_raw_affine);
            std::process::exit(1);
        }
    }

    eprintln!("All correctness tests PASSED!");
}
