use cpntt::{
    intt_m32,
    montgomery::{Mod998244353, M32},
    ntt_m32,
};
use criterion::{criterion_group, criterion_main, Criterion};

fn simple_ntt_bench(b: &mut Criterion) {
    b.bench_function("NTT", |b| {
        b.iter(|| {
            type Modint = M32<Mod998244353>;
            for i in 15..=20 {
                let mut data = (0..1 << i).map(Modint::new).collect::<Vec<_>>();
                ntt_m32(&mut data);
                intt_m32(&mut data);
            }
        })
    });
}

criterion_group!(benches, simple_ntt_bench);
criterion_main!(benches);
