use super::{modulo::Modulo, M32, THRESHOLD};
use std::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_blend_epi32, _mm256_castps_si256,
    _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_i32gather_epi32,
    _mm256_loadu_si256, _mm256_max_epu32, _mm256_min_epu32, _mm256_mul_epu32, _mm256_mullo_epi32,
    _mm256_or_si256, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_shuffle_epi32,
    _mm256_shuffle_ps, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
};
use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

/// # Safety
/// - The argument must be a register containing eight u32s of Montgomery representation.
#[inline]
#[target_feature(enable = "avx", enable = "avx2")]
pub unsafe fn mreducex8<M: Modulo>(t: __m256i) -> __m256i {
    let t_ninv = _mm256_mullo_epi32(t, M::N_INVX8);
    let t_ninv_n_lo = _mm256_mul_epu32(t_ninv, M::NX8);
    let t_ninv_n_hi = _mm256_mul_epu32(_mm256_shuffle_epi32(t_ninv, 0b10_11_00_01), M::NX8);
    let mr = _mm256_blend_epi32(
        _mm256_shuffle_epi32(t_ninv_n_lo, 0b10_11_00_01),
        t_ninv_n_hi,
        0b10101010,
    );
    if M::N > THRESHOLD {
        let zero = _mm256_setzero_si256();
        let mask = _mm256_cmpeq_epi32(mr, zero);
        let mask = _mm256_and_si256(
            M::NX8,
            _mm256_xor_si256(mask, _mm256_cmpeq_epi32(mask, mask)),
        );
        _mm256_sub_epi32(mask, mr)
    } else {
        _mm256_sub_epi32(M::NX8, mr)
    }
}

/// # Safety
/// - The argument must be a register containing eight u32s of Montgomery representation.
#[inline]
#[target_feature(enable = "avx", enable = "avx2")]
pub unsafe fn mmulx8<M: Modulo>(a: __m256i, b: __m256i) -> __m256i {
    let t1 = _mm256_mul_epu32(a, b);
    let t2 = _mm256_mul_epu32(
        _mm256_shuffle_epi32(a, 0b11_11_01_01),
        _mm256_shuffle_epi32(b, 0b11_11_01_01),
    );
    let t_modinv = {
        let tmi1 = _mm256_mul_epu32(t1, M::N_INVX8);
        let tmi2 = _mm256_mul_epu32(t2, M::N_INVX8);
        _mm256_castps_si256(_mm256_shuffle_ps(
            _mm256_castsi256_ps(tmi1),
            _mm256_castsi256_ps(tmi2),
            0b10_00_10_00,
        ))
    };

    let c = _mm256_shuffle_epi32(
        _mm256_castps_si256(_mm256_shuffle_ps(
            _mm256_castsi256_ps(_mm256_mul_epu32(t_modinv, M::NX8)),
            _mm256_castsi256_ps(_mm256_mul_epu32(
                _mm256_shuffle_epi32(t_modinv, 0b11_11_01_01),
                M::NX8,
            )),
            0b11_01_11_01,
        )),
        0b11_01_10_00,
    );

    let t = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(t1),
        _mm256_castsi256_ps(t2),
        0b11_01_11_01,
    ));
    if M::N > THRESHOLD {
        let one = _mm256_cmpeq_epi32(c, c);
        let mask = _mm256_and_si256(
            M::NX8,
            _mm256_xor_si256(one, _mm256_cmpeq_epi32(_mm256_min_epu32(t, c), c)),
        );
        _mm256_shuffle_epi32(
            _mm256_add_epi32(mask, _mm256_sub_epi32(t, c)),
            0b11_01_10_00,
        )
    } else {
        _mm256_shuffle_epi32(
            _mm256_add_epi32(M::NX8, _mm256_sub_epi32(t, c)),
            0b11_01_10_00,
        )
    }
}

/// # Safety
/// - The argument must be a register containing eight u32s of Montgomery representation.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn mrestorex8<M: Modulo>(t: __m256i) -> __m256i {
    if M::N > THRESHOLD {
        t
    } else {
        let mask = _mm256_or_si256(_mm256_cmpgt_epi32(t, M::NX8), _mm256_cmpeq_epi32(t, M::NX8));
        _mm256_sub_epi32(t, _mm256_and_si256(mask, M::NX8))
    }
}

/// # Safety
/// - The argument must be a register containing eight u32s of Montgomery representation.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn maddx8<M: Modulo>(a: __m256i, b: __m256i) -> __m256i {
    if M::N > THRESHOLD {
        let diff = _mm256_sub_epi32(M::NX8, a);
        let mask = _mm256_cmpeq_epi32(diff, _mm256_max_epu32(diff, b));
        let val = _mm256_add_epi32(_mm256_sub_epi32(a, M::NX8), b);
        _mm256_add_epi32(val, _mm256_and_si256(mask, M::NX8))
    } else {
        let res = _mm256_add_epi32(a, b);
        let mask = _mm256_cmpeq_epi32(res, _mm256_max_epu32(res, M::N2X8));
        _mm256_sub_epi32(res, _mm256_and_si256(M::N2X8, mask))
    }
}

/// # Safety
/// - The argument must be a register containing eight u32s of Montgomery representation.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn msubx8<M: Modulo>(a: __m256i, b: __m256i) -> __m256i {
    if M::N > THRESHOLD {
        let mask = _mm256_cmpeq_epi32(b, _mm256_max_epu32(a, b));
        let c_neg = _mm256_sub_epi32(a, b);
        _mm256_add_epi32(c_neg, _mm256_and_si256(M::NX8, mask))
    } else {
        let mask = _mm256_cmpgt_epi32(b, a);
        _mm256_add_epi32(_mm256_sub_epi32(a, b), _mm256_and_si256(M::N2X8, mask))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct M32x8<M: Modulo> {
    val: __m256i,
    _phantom: PhantomData<fn() -> M>,
}

impl<M: Modulo> M32x8<M> {
    /// `slice` should **NOT** be Montgomery Representation.
    /// After `slice` is loaded, this method converts the values of `slice` into Montgomery Representation.
    pub(crate) unsafe fn convert_from_u32slice(slice: &[u32]) -> Self {
        Self {
            val: (Self::load(slice.as_ptr() as _) * Self::from_rawval(M::R2X8)).rawval(),
            _phantom: PhantomData,
        }
    }

    pub fn splat(val: M32<M>) -> Self {
        Self::from_rawval(unsafe { _mm256_set1_epi32(val.val as i32) })
    }

    /// `val` should have be converted to Montgomery Representation.
    /// Internally, this method simply loads `val` does no other operations.

    pub fn from_rawval(val: __m256i) -> Self {
        Self {
            val,
            _phantom: PhantomData,
        }
    }

    pub fn val(&self) -> __m256i {
        unsafe { mrestorex8::<M>(mreducex8::<M>(self.val)) }
    }

    pub fn rawval(&self) -> __m256i {
        self.val
    }

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn load(head: *const M32<M>) -> Self {
        Self {
            val: _mm256_loadu_si256(head as _),
            _phantom: PhantomData,
        }
    }

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn store(&self, head: *mut M32<M>) {
        unsafe { _mm256_storeu_si256(head as _, self.val) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn gather(head: *const M32<M>, vindex: __m256i) -> Self {
        Self::from_rawval(unsafe { _mm256_i32gather_epi32(head as _, vindex, 4) })
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn unpacklo64(self, other: Self) -> Self {
        Self::from_rawval(_mm256_unpacklo_epi64(self.val, other.val))
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn unpackhi64(self, other: Self) -> Self {
        Self::from_rawval(_mm256_unpackhi_epi64(self.val, other.val))
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn unpacklo32(self, other: Self) -> Self {
        Self::from_rawval(_mm256_unpacklo_epi32(self.val, other.val))
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn unpackhi32(self, other: Self) -> Self {
        Self::from_rawval(_mm256_unpackhi_epi32(self.val, other.val))
    }
}

impl<M: Modulo> Add for M32x8<M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            val: unsafe { maddx8::<M>(self.val, rhs.val) },
            _phantom: PhantomData,
        }
    }
}

impl<M: Modulo> Sub for M32x8<M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            val: unsafe { msubx8::<M>(self.val, rhs.val) },
            _phantom: PhantomData,
        }
    }
}

impl<M: Modulo> Mul for M32x8<M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            val: unsafe { mmulx8::<M>(self.val, rhs.val) },
            _phantom: PhantomData,
        }
    }
}

impl<M: Modulo> std::fmt::Debug for M32x8<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dest = [0u32; 8];
        unsafe {
            _mm256_storeu_si256(dest.as_mut_ptr() as *mut _, self.val);
            write!(f, "{:?}", dest)
        }
    }
}
