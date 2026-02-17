use ark_bn254::{G1Affine, G2Affine};
use ark_ff::BigInteger256;

fn main() {
    println!("G1Affine:      {} bytes, align {}", std::mem::size_of::<G1Affine>(), std::mem::align_of::<G1Affine>());
    println!("G2Affine:      {} bytes, align {}", std::mem::size_of::<G2Affine>(), std::mem::align_of::<G2Affine>());
    println!("BigInteger256: {} bytes, align {}", std::mem::size_of::<BigInteger256>(), std::mem::align_of::<BigInteger256>());
}
