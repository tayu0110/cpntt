// In a normal Decimation In Frequency (DIF) FFT, the array starts the operation in normal order and are reordered into bit-reversal order according to the signal flow.
// However, the same result can be obtained by first re-ordering the array in bit-reversal order,
// then proceeding in order of reversal signal flow with multiplying twiddle factors that is a power of the bit-reversal order to each block.
//
// This method greatly reduces the amount of cache required for the twiddle factors and improves performance by making memory accesses continuous and localized.
// Similar results are obtained for the Decimation In Time (DIT) FFT.
//
// The normal DIF requires bit-reversal reordering after the operation (or before in the case of DIT),
// but when FFT and IFFT are executed in pairs, the bit-reversal reordering can be canceled by proceeding in the order of DIF and IDIT.
//
// In this implementation, the correct result can be obtained by proceeding in the order of DIT and IDIF.
// The implementation was based on the AtCoder Library (reference1), and reference2 was used to understand the semantics of the implementation.
//
// - reference1: https://github.com/atcoder/ac-library/blob/master/atcoder/convolution.hpp
// - reference2: https://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_2.html#sec1_2_1

pub mod convolution;
pub mod cooley_tukey;
mod fft_cache;
pub mod gentleman_sande;
pub mod montgomery;
pub mod utility;

use std::mem::transmute;

use cooley_tukey::cooley_tukey_radix_4_butterfly;
pub use fft_cache::FftCache;
use gentleman_sande::gentleman_sande_radix_4_butterfly_inv;
pub use montgomery::*;
pub use utility::*;

/// Apply Number Theoretic Transform to `slice`.
///
/// # Constraint
/// - `slice.len()` must be a power of two.
/// - `slice.len() <= 1 << (M::N - 1).trailing_zeros()` must be satisfied.
pub fn ntt_m32<M: Modulo>(slice: &mut [M32<M>]) {
    let n = slice.len();
    assert_eq!(n, 1 << n.trailing_zeros());
    assert!(n <= (1 << (M::N - 1).trailing_zeros()));

    unsafe { cooley_tukey_radix_4_butterfly(n, slice, &M::CACHE) }
}
/// Apply Inverse Number Theoretic Transform to `slice`.
///
/// # Constraint
/// - `slice.len()` must be a power of two.
/// - `slice.len() <= 1 << (M::N - 1).trailing_zeros()` must be satisfied.
pub fn intt_m32<M: Modulo>(slice: &mut [M32<M>]) {
    let n = slice.len();
    assert_eq!(n, 1 << n.trailing_zeros());
    assert!(n <= (1 << (M::N - 1).trailing_zeros()));

    unsafe { gentleman_sande_radix_4_butterfly_inv(n, slice, &M::CACHE) }

    let ninv = M32::new(n as u32).inv();
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if n >= 8 && is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
        let ninv = M32x8::<M>::splat(ninv);
        for v in slice.chunks_exact_mut(8) {
            unsafe {
                let res = M32x8::load(v.as_ptr()) * ninv;
                res.store(v.as_mut_ptr());
            }
        }
    } else {
        slice.iter_mut().for_each(|a| *a *= ninv);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        slice.iter_mut().for_each(|a| *a *= ninv);
    }
}

/// Apply Number Theoretic Transform to `slice`.
///
/// After performing this method, elements of `slice` keeps to be Montgomery representation.  
/// If you need thier normal representation, you can use `utility::m32tou32`.
///
/// # Constraint
/// - `slice.len()` must be a power of two.
/// - `slice.len() <= 1 << (M::N - 1).trailing_zeros()` must be satisfied.
pub fn ntt_u32<M: Modulo>(slice: &mut [u32]) {
    unsafe {
        utility::u32tom32::<M>(slice);
        let converted = transmute::<&mut [u32], &mut [M32<M>]>(slice);
        ntt_m32(converted);
    }
}

/// Apply Inverse Number Theoretic Transform to `slice`.
///
/// This method requests that elements of `slice` are already Montgomery representation.  
/// If you can apply this method to normal representation slice, you can use `utility::u32tom32` before execution.
///
/// # Constraint
/// - `slice.len()` must be a power of two.
/// - `slice.len() <= 1 << (M::N - 1).trailing_zeros()` must be satisfied.
pub fn intt_u32<M: Modulo>(slice: &mut [u32]) {
    unsafe {
        let converted = transmute::<&mut [u32], &mut [M32<M>]>(slice);
        intt_m32(converted);
        utility::m32tou32::<M>(converted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cooley_tukey::cooley_tukey_radix_4_butterfly_inv;
    use gentleman_sande::gentleman_sande_radix_4_butterfly;
    use montgomery::Mod998244353;

    #[test]
    fn self_change_ntt_test() {
        type Modint = M32<Mod998244353>;
        for i in 15..=20 {
            let mut data = (1..=1 << i).map(Modint::new).collect::<Vec<_>>();
            let expect = data.clone();
            ntt_m32::<Mod998244353>(&mut data);
            intt_m32::<Mod998244353>(&mut data);
            assert_eq!(data, expect);
        }
    }

    #[test]
    fn self_change_ntt_u32_test() {
        for i in 15..=20 {
            let mut data = (1u32..=1 << i).collect::<Vec<_>>();
            let expect = data.clone();
            ntt_u32::<Mod998244353>(&mut data);
            intt_u32::<Mod998244353>(&mut data);
            assert_eq!(data, expect);
        }
    }

    type Modint = M32<Mod998244353>;

    pub fn ntt_cooley_tukey_radix_4(a: &mut [Modint]) {
        let deg = a.len();
        let log = deg.trailing_zeros() as usize;
        debug_assert_eq!(a.len(), 1 << log);
        bit_reverse(deg, a);
        let cache = FftCache::<Mod998244353>::new();
        unsafe { gentleman_sande_radix_4_butterfly(deg, a, &cache) }
    }
    pub fn intt_cooley_tukey_radix_4(a: &mut [Modint]) {
        let deg = a.len();
        let log = deg.trailing_zeros() as usize;
        debug_assert_eq!(a.len(), 1 << log);
        bit_reverse(deg, a);
        let cache = FftCache::<Mod998244353>::new();
        unsafe { gentleman_sande_radix_4_butterfly_inv(deg, a, &cache) }
        let inv = Modint::new(deg as u32).inv();
        a.iter_mut().for_each(|c| *c *= inv)
    }

    #[test]
    fn cooley_tukey_radix_4_test() {
        for i in 0..=13 {
            let n = 1 << i;
            let data: Vec<Modint> = (1..=n).map(Modint::new).collect();
            let mut data1 = data.clone();
            ntt_cooley_tukey_radix_4(&mut data1);
            intt_cooley_tukey_radix_4(&mut data1);
            assert_eq!(data, data1);
        }
    }

    pub fn ntt_gentleman_sande_radix_4(a: &mut [Modint]) {
        let deg = a.len();
        let log = deg.trailing_zeros() as usize;
        debug_assert_eq!(a.len(), 1 << log);
        let cache = FftCache::new();
        unsafe { cooley_tukey_radix_4_butterfly(deg, a, &cache) }
        bit_reverse(deg, a);
    }

    pub fn intt_gentleman_sande_radix_4(a: &mut [Modint]) {
        let deg = a.len();
        let log = deg.trailing_zeros() as usize;
        debug_assert_eq!(a.len(), 1 << log);
        let cache = FftCache::new();
        unsafe { cooley_tukey_radix_4_butterfly_inv(deg, a, &cache) }
        bit_reverse(deg, a);
        let inv = Modint::new(deg as u32).inv();
        a.iter_mut().for_each(|c| *c *= inv)
    }

    #[test]
    fn gentleman_sande_radix_4_test() {
        for i in 0..=13 {
            let n = 1 << i;
            let data: Vec<Modint> = (1..=n).map(Modint::new).collect();
            let mut data1 = data.clone();
            ntt_gentleman_sande_radix_4(&mut data1);
            intt_gentleman_sande_radix_4(&mut data1);
            assert_eq!(data, data1);
        }
    }
}
