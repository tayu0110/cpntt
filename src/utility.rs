#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use super::montgomery::M32x8;
use super::montgomery::{Modulo, M32};

#[inline]
// reference: https://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_25.html#sec1_2_5
pub fn bit_reverse<T>(deg: usize, a: &mut [T]) {
    let nh = deg >> 1;
    let nh1 = nh + 1;
    let mut i = 0;
    for j in (0..nh).step_by(2) {
        if j < i {
            a.swap(i, j);
            a.swap(i + nh1, j + nh1);
        }
        a.swap(j + nh, i + 1);
        let mut k = nh >> 1;
        i ^= k;
        while k > i {
            k >>= 1;
            i ^= k;
        }
    }
}

/// # Safety
/// No constraint. For apply `#[target_feature(...)]`.
#[inline]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2", enable = "bmi1")]
pub unsafe fn u32tom32<M: Modulo>(a: &mut [u32]) {
    let mut it = a.chunks_exact_mut(8);
    for v in it.by_ref() {
        M32x8::<M>::convert_from_u32slice(v).store(v.as_mut_ptr() as _);
    }
    it.into_remainder()
        .iter_mut()
        .for_each(|a| *a = M32::<M>::from(*a).rawval());
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub fn u32tom32<M: Modulo>(a: &mut [u32]) {
    a.iter_mut().for_each(|a| *a = M32::<M>::from(*a).rawval());
}

/// # Safety
/// No constraint. For apply `#[target_feature(...)]`.
#[inline]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2", enable = "bmi1")]
pub unsafe fn m32tou32<M: Modulo>(a: &mut [M32<M>]) {
    let mut it = a.chunks_exact_mut(8);
    for v in it.by_ref() {
        M32x8::from_rawval(M32x8::load(v.as_ptr()).val()).store(v.as_mut_ptr());
    }
    it.into_remainder()
        .iter_mut()
        .for_each(|a| *a = M32::from_rawval(a.val()));
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub fn m32tou32<M: Modulo>(a: &mut [M32<M>]) {
    a.iter_mut().for_each(|a| *a = M32::from_rawval(a.val()));
}
