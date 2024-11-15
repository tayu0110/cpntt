use super::montgomery::{Modulo, M32};

/// AtCoder-Library like FftCache
/// reference: https://github.com/atcoder/ac-library/blob/master/atcoder/convolution.hpp
pub struct FftCache<M: Modulo> {
    pub root: [M32<M>; 35],
    pub iroot: [M32<M>; 35],
    pub rate2: [M32<M>; 35],
    pub irate2: [M32<M>; 35],
    pub rate3: [M32<M>; 35],
    pub irate3: [M32<M>; 35],
}

impl<M: Modulo> FftCache<M> {
    const RANK2: usize = (M::N - 1).trailing_zeros() as usize;
    pub const fn new() -> Self {
        let mut root = [M32::one(); 35];
        let mut iroot = [M32::one(); 35];
        let mut rate2 = [M32::one(); 35];
        let mut irate2 = [M32::one(); 35];
        let mut rate3 = [M32::one(); 35];
        let mut irate3 = [M32::one(); 35];

        root[Self::RANK2] = M32::<M>::nth_root(1 << Self::RANK2);
        iroot[Self::RANK2] = root[Self::RANK2].inv();
        let mut i = Self::RANK2;
        while i > 0 {
            i -= 1;
            root[i] = root[i + 1].mul_const(root[i + 1]);
            iroot[i] = iroot[i + 1].mul_const(iroot[i + 1]);
        }

        let mut prod = M32::one();
        let mut iprod = M32::one();
        let mut i = 0;
        while i < Self::RANK2.saturating_sub(1) {
            rate2[i] = root[i + 2].mul_const(prod);
            irate2[i] = iroot[i + 2].mul_const(iprod);
            prod = prod.mul_const(iroot[i + 2]);
            iprod = iprod.mul_const(root[i + 2]);
            i += 1;
        }

        let mut prod = M32::one();
        let mut iprod = M32::one();
        let mut i = 0;
        while i < Self::RANK2.saturating_sub(2) {
            rate3[i] = root[i + 3].mul_const(prod);
            irate3[i] = iroot[i + 3].mul_const(iprod);
            prod = prod.mul_const(iroot[i + 3]);
            iprod = iprod.mul_const(root[i + 3]);
            i += 1;
        }

        Self {
            root,
            iroot,
            rate2,
            irate2,
            rate3,
            irate3,
        }
    }

    #[inline]
    pub fn gen_rate(&self, log: usize) -> Vec<M32<M>> {
        if log == 2 {
            return self.rate2.into();
        } else if log == 3 {
            return self.rate3.into();
        }
        let mut rate = vec![M32::one(); Self::RANK2.saturating_sub(log - 1)];
        let mut prod = M32::one();
        for (i, rate) in rate.iter_mut().enumerate() {
            *rate = self.root[i + log] * prod;
            prod *= self.iroot[i + log];
        }
        rate
    }

    #[inline]
    pub fn gen_irate(&self, log: usize) -> Vec<M32<M>> {
        if log == 2 {
            return self.irate2.into();
        }
        if log == 3 {
            return self.irate3.into();
        }
        let mut irate = vec![M32::one(); Self::RANK2.saturating_sub(log - 1)];
        let mut iprod = M32::one();
        for (i, irate) in irate.iter_mut().enumerate() {
            *irate = self.iroot[i + log] * iprod;
            iprod *= self.root[i + log];
        }
        irate
    }
}

impl<M: Modulo> Default for FftCache<M> {
    fn default() -> Self {
        Self::new()
    }
}
