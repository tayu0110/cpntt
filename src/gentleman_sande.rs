#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use super::montgomery::M32x8;
use super::montgomery::{Modulo, M32};
use super::FftCache;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::arch::x86_64::{
    _mm256_permute2f128_si256, _mm256_permute4x64_epi64, _mm256_setr_epi32, _mm256_storeu_si256,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
};

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
macro_rules! radix4_inner {
    ($c0:expr, $c1:expr, $c2:expr, $c3:expr, $rot:expr, $rot2:expr, $rot3:expr, $imag:expr) => {{
        let (c0, c1, c2, c3) = ($c0, $c1, $c2, $c3);
        let c01 = c0 + c1;
        let c01n = c0 - c1;
        let c23 = c2 + c3;
        let c23nim = (c2 - c3) * $imag;
        (
            c01 + c23,
            (c01n + c23nim) * $rot,
            (c01 - c23) * $rot2,
            (c01n - c23nim) * $rot3,
        )
    }};
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline]
fn gentleman_sande_radix_2_kernel<M: Modulo>(
    deg: usize,
    width: usize,
    offset: usize,
    a: &mut [M32<M>],
    rate: &[M32<M>],
) {
    let mut rot = M32::one();
    let blocks = deg / width;
    for block in 0..blocks {
        let top = block * width;
        if rot == M32::one() {
            for now in top..top + offset {
                let c = a[now + offset];
                a[now + offset] = a[now] - c;
                a[now] += c;
            }
        } else {
            for now in top..top + offset {
                let c = a[now + offset];
                a[now + offset] = (a[now] - c) * rot;
                a[now] += c;
            }
        }
        if top + width != deg {
            rot *= rate[block.trailing_ones() as usize];
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline]
fn gentleman_sande_radix_4_kernel<M: Modulo>(
    deg: usize,
    width: usize,
    offset: usize,
    im: M32<M>,
    a: &mut [M32<M>],
    rate: &[M32<M>],
) {
    let mut rot = M32::one();
    let blocks = deg / width;
    for block in 0..blocks {
        let top = block * width;
        let rot2 = rot * rot;
        let rot3 = rot * rot2;
        for now in top..top + offset {
            let (c0, c1, c2, c3) = (
                a[now],
                a[now + offset],
                a[now + offset * 2],
                a[now + offset * 3],
            );
            let c01 = c0 + c1;
            let c01n = c0 - c1;
            let c23 = c2 + c3;
            let c23nim = (c2 - c3) * im;
            a[now] = c01 + c23;
            a[now + offset] = (c01n + c23nim) * rot;
            a[now + offset * 2] = (c01 - c23) * rot2;
            a[now + offset * 3] = (c01n - c23nim) * rot3;
        }
        if top + width != deg {
            rot *= rate[block.trailing_ones() as usize];
        }
    }
}

#[target_feature(enable = "avx2", enable = "bmi1")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn gentleman_sande_radix_2_kernel<M: Modulo>(
    deg: usize,
    width: usize,
    offset: usize,
    a: &mut [M32<M>],
    rate: &[M32<M>],
) {
    let mut rot = M32::one();
    let blocks = deg / width;
    if offset == 1 && blocks >= 8 {
        let mut r = [M32::zero(); 8];
        let vindex = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        for block in (0..blocks).step_by(8) {
            let top = block * width;
            for i in 0..8 {
                r[i] = rot;
                rot *= rate[(block + i).trailing_ones() as usize];
            }
            let rot = M32x8::load(r.as_ptr());
            let c0 = M32x8::gather(a[top..].as_ptr(), vindex);
            let c1 = M32x8::gather(a[top + 1..].as_ptr(), vindex);

            let (c0, c1) = (c0 + c1, (c0 - c1) * rot);
            let r0 = c0.unpacklo32(c1);
            let r1 = c0.unpackhi32(c1);
            _mm256_storeu_si256(
                a[top..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x20>(r0.rawval(), r1.rawval()),
            );
            _mm256_storeu_si256(
                a[top + 8..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x31>(r0.rawval(), r1.rawval()),
            );
        }
    } else {
        for block in 0..blocks {
            let top = block * width;
            for now in top..top + offset {
                let c = a[now + offset];
                a[now + offset] = (a[now] - c) * rot;
                a[now] += c;
            }
            rot *= rate[block.trailing_ones() as usize];
        }
    }
}

#[target_feature(enable = "avx2", enable = "bmi1")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn gentleman_sande_radix_4_kernel<M: Modulo>(
    deg: usize,
    width: usize,
    offset: usize,
    im: M32<M>,
    a: &mut [M32<M>],
    rate: &[M32<M>],
) {
    let mut rot = M32::one();
    let blocks = deg / width;
    let imag = M32x8::splat(im);

    if offset == 1 && blocks >= 8 {
        let mut r = [M32::zero(); 8];
        let vindex = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        for block in (0..blocks).step_by(8) {
            let top = block * width;
            for i in 0..8 {
                r[i] = rot;
                rot *= rate[(block + i).trailing_ones() as usize];
            }
            let rot = M32x8::load(r.as_ptr());
            let rot2 = rot * rot;
            let rot3 = rot * rot2;
            let (r0, r1, r2, r3) = radix4_inner!(
                M32x8::gather(a[top..].as_ptr(), vindex),
                M32x8::gather(a[top + 1..].as_ptr(), vindex),
                M32x8::gather(a[top + 2..].as_ptr(), vindex),
                M32x8::gather(a[top + 3..].as_ptr(), vindex),
                rot,
                rot2,
                rot3,
                imag
            );
            let r01lo = r0.unpacklo32(r1);
            let r01hi = r0.unpackhi32(r1);
            let r23lo = r2.unpacklo32(r3);
            let r23hi = r2.unpackhi32(r3);
            let r01lo = _mm256_permute4x64_epi64::<0b11_01_10_00>(r01lo.rawval());
            let r01hi = _mm256_permute4x64_epi64::<0b11_01_10_00>(r01hi.rawval());
            let r23lo = _mm256_permute4x64_epi64::<0b11_01_10_00>(r23lo.rawval());
            let r23hi = _mm256_permute4x64_epi64::<0b11_01_10_00>(r23hi.rawval());
            _mm256_storeu_si256(
                a[top..].as_mut_ptr() as _,
                _mm256_unpacklo_epi64(r01lo, r23lo),
            );
            _mm256_storeu_si256(
                a[top + 8..].as_mut_ptr() as _,
                _mm256_unpacklo_epi64(r01hi, r23hi),
            );
            _mm256_storeu_si256(
                a[top + 16..].as_mut_ptr() as _,
                _mm256_unpackhi_epi64(r01lo, r23lo),
            );
            _mm256_storeu_si256(
                a[top + 24..].as_mut_ptr() as _,
                _mm256_unpackhi_epi64(r01hi, r23hi),
            );
        }
    } else if offset == 2 && blocks >= 4 {
        let mut r = [M32::zero(); 8];
        let vindex = _mm256_setr_epi32(0, 1, 8, 9, 16, 17, 24, 25);
        for block in (0..blocks).step_by(4) {
            let top = block * width;
            for i in 0..4 {
                r[i * 2] = rot;
                r[i * 2 + 1] = rot;
                rot *= rate[(block + i).trailing_ones() as usize];
            }
            let rot = M32x8::load(r.as_ptr());
            let rot2 = rot * rot;
            let rot3 = rot * rot2;
            let (r0, r1, r2, r3) = radix4_inner!(
                M32x8::gather(a[top..].as_ptr(), vindex),
                M32x8::gather(a[top + 2..].as_ptr(), vindex),
                M32x8::gather(a[top + 4..].as_ptr(), vindex),
                M32x8::gather(a[top + 6..].as_ptr(), vindex),
                rot,
                rot2,
                rot3,
                imag
            );
            let r01lo = r0.unpacklo64(r1);
            let r01hi = r0.unpackhi64(r1);
            let r23lo = r2.unpacklo64(r3);
            let r23hi = r2.unpackhi64(r3);
            _mm256_storeu_si256(
                a[top..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x20>(r01lo.rawval(), r23lo.rawval()),
            );
            _mm256_storeu_si256(
                a[top + 8..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x20>(r01hi.rawval(), r23hi.rawval()),
            );
            _mm256_storeu_si256(
                a[top + 16..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x31>(r01lo.rawval(), r23lo.rawval()),
            );
            _mm256_storeu_si256(
                a[top + 24..].as_mut_ptr() as _,
                _mm256_permute2f128_si256::<0x31>(r01hi.rawval(), r23hi.rawval()),
            );
        }
    } else if offset == 4 && blocks >= 2 {
        let vindex = _mm256_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19);
        for block in (0..blocks).step_by(2) {
            let top = block * width;
            let next_rot = rot * rate[block.trailing_ones() as usize];
            {
                let rot = M32x8::load(
                    [rot, rot, rot, rot, next_rot, next_rot, next_rot, next_rot].as_ptr(),
                );
                let rot2 = rot * rot;
                let rot3 = rot * rot2;
                let (r0, r1, r2, r3) = radix4_inner!(
                    M32x8::gather(a[top..].as_ptr(), vindex),
                    M32x8::gather(a[top + 4..].as_ptr(), vindex),
                    M32x8::gather(a[top + 8..].as_ptr(), vindex),
                    M32x8::gather(a[top + 12..].as_ptr(), vindex),
                    rot,
                    rot2,
                    rot3,
                    imag
                );
                _mm256_storeu_si256(
                    a[top..].as_mut_ptr() as _,
                    _mm256_permute2f128_si256::<0x20>(r0.rawval(), r1.rawval()),
                );
                _mm256_storeu_si256(
                    a[top + 8..].as_mut_ptr() as _,
                    _mm256_permute2f128_si256::<0x20>(r2.rawval(), r3.rawval()),
                );
                _mm256_storeu_si256(
                    a[top + 16..].as_mut_ptr() as _,
                    _mm256_permute2f128_si256::<0x31>(r0.rawval(), r1.rawval()),
                );
                _mm256_storeu_si256(
                    a[top + 24..].as_mut_ptr() as _,
                    _mm256_permute2f128_si256::<0x31>(r2.rawval(), r3.rawval()),
                );
            }
            rot = next_rot * rate[(block + 1).trailing_ones() as usize];
        }
    } else if offset < 8 {
        for block in 0..blocks {
            let top = block * width;
            let rot2 = rot * rot;
            let rot3 = rot * rot2;
            for now in top..top + offset {
                let (r0, r1, r2, r3) = radix4_inner!(
                    a[now],
                    a[now + offset],
                    a[now + offset * 2],
                    a[now + offset * 3],
                    rot,
                    rot2,
                    rot3,
                    im
                );
                a[now] = r0;
                a[now + offset] = r1;
                a[now + offset * 2] = r2;
                a[now + offset * 3] = r3;
            }
            rot *= rate[block.trailing_ones() as usize];
        }
    } else {
        for now in (0..offset).step_by(8) {
            let c0 = M32x8::load(a[now..].as_ptr());
            let c1 = M32x8::load(a[now + offset..].as_ptr());
            let c2 = M32x8::load(a[now + offset * 2..].as_ptr());
            let c3 = M32x8::load(a[now + offset * 3..].as_ptr());
            let c01 = c0 + c1;
            let c01n = c0 - c1;
            let c23 = c2 + c3;
            let c23nim = (c2 - c3) * imag;
            (c01 + c23).store(a[now..].as_mut_ptr());
            (c01n + c23nim).store(a[now + offset..].as_mut_ptr());
            (c01 - c23).store(a[now + offset * 2..].as_mut_ptr());
            (c01n - c23nim).store(a[now + offset * 3..].as_mut_ptr());
        }
        let mut rot = M32x8::splat(rate[0]);
        for block in 1..blocks {
            let top = block * width;
            let rot2 = rot * rot;
            let rot3 = rot * rot2;
            let mut head = a[top..].as_mut_ptr();
            for _ in (top..top + offset).step_by(8) {
                let (c0a, c1a, c2a, c3a) = (
                    head,
                    head.add(offset),
                    head.add(offset * 2),
                    head.add(offset * 3),
                );
                let (r0, r1, r2, r3) = radix4_inner!(
                    M32x8::load(c0a),
                    M32x8::load(c1a),
                    M32x8::load(c2a),
                    M32x8::load(c3a),
                    rot,
                    rot2,
                    rot3,
                    imag
                );
                r0.store(c0a);
                r1.store(c1a);
                r2.store(c2a);
                r3.store(c3a);
                head = head.add(8);
            }
            rot = rot * M32x8::splat(rate[block.trailing_ones() as usize]);
        }
    }
}

/// # Safety
/// The length of `a` must be the power of 2.
#[inline]
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature(enable = "avx2", enable = "bmi1")
)]
pub unsafe fn gentleman_sande_radix_4_butterfly<M: Modulo>(
    deg: usize,
    a: &mut [M32<M>],
    cache: &FftCache<M>,
) {
    let log = deg.trailing_zeros();
    if log & 1 != 0 {
        let width = 1 << 1;
        let offset = width >> 1;
        gentleman_sande_radix_2_kernel(deg, width, offset, a, &cache.rate2);
    }
    for i in (log & 1..log).step_by(2) {
        let width = 1 << (i + 2);
        let offset = width >> 2;
        gentleman_sande_radix_4_kernel(deg, width, offset, cache.root[2], a, &cache.rate3);
    }
}

/// # Safety
/// The length of `a` must be the power of 2.
#[inline]
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature(enable = "avx2", enable = "bmi1")
)]
pub unsafe fn gentleman_sande_radix_4_butterfly_inv<M: Modulo>(
    deg: usize,
    a: &mut [M32<M>],
    cache: &FftCache<M>,
) {
    let log = deg.trailing_zeros();
    if log & 1 != 0 {
        let width = 1 << 1;
        let offset = width >> 1;
        gentleman_sande_radix_2_kernel(deg, width, offset, a, &cache.irate2);
    }
    for i in (log & 1..log).step_by(2) {
        let width = 1 << (i + 2);
        let offset = width >> 2;
        gentleman_sande_radix_4_kernel(deg, width, offset, cache.iroot[2], a, &cache.irate3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{montgomery::Mod998244353, utility::bit_reverse};

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
        for i in 1..=13 {
            let n = 1 << i;
            let data: Vec<Modint> = (1..=n).map(Modint::new).collect();
            let mut data1 = data.clone();
            ntt_cooley_tukey_radix_4(&mut data1);
            intt_cooley_tukey_radix_4(&mut data1);
            assert_eq!(data, data1);
        }
    }
}
