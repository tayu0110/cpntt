use cpntt::{
    bit_reverse,
    cooley_tukey::{cooley_tukey_radix_4_butterfly, cooley_tukey_radix_4_butterfly_inv},
    montgomery::{Mod998244353, M32},
    FftCache,
};
use criterion::{criterion_group, criterion_main, Criterion};

type Modint = M32<Mod998244353>;

pub fn ntt_cooley_tukey_radix_4(a: &mut [Modint]) {
    let deg = a.len();
    let log = deg.trailing_zeros() as usize;
    debug_assert_eq!(a.len(), 1 << log);
    let cache = FftCache::new();
    unsafe { cooley_tukey_radix_4_butterfly(deg, a, &cache) }
    bit_reverse(deg, a);
}

pub fn intt_cooley_tukey_radix_4(a: &mut [Modint]) {
    let deg = a.len();
    let log = deg.trailing_zeros() as usize;
    debug_assert_eq!(a.len(), 1 << log);
    let cache = FftCache::new();
    unsafe { cooley_tukey_radix_4_butterfly_inv(deg, a, &cache) }
    bit_reverse(deg, a);
    let inv = Modint::new(deg as u32).inv();
    a.iter_mut().for_each(|c| *c *= inv)
}

fn cooley_tukey_radix_4_bench(b: &mut Criterion) {
    b.bench_function("Gentleman Sande Radix-4", |b| {
        b.iter(|| {
            for i in 15..=20 {
                let n = 1 << i;
                let mut data: Vec<Modint> = (1..=n).map(Modint::new).collect();
                ntt_cooley_tukey_radix_4(&mut data);
                intt_cooley_tukey_radix_4(&mut data);
            }
        })
    });
}

criterion_group!(benches, cooley_tukey_radix_4_bench);
criterion_main!(benches);
