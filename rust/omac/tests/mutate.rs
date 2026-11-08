//! Corrupted input must give an error, never a panic (spec section 2: a
//! decoder rejects invalid data). CRCs are recomputed after mutating, so the
//! damage reaches the header, size-table and bitstream parsers instead of
//! stopping at the checksum. Run in debug mode, where integer overflow panics.

use std::panic::{catch_unwind, AssertUnwindSafe};

use omac::{decode_all, decode_range, encode, EncoderConfig, Header, StreamInfo};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn u32_at(d: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(d[i..i + 4].try_into().unwrap())
}

/// Recompute header and block CRCs wherever the structure is still readable.
fn fix_crcs(d: &mut [u8]) {
    let Ok(info) = StreamInfo::parse(d) else { return };
    let hs = info.header_size();
    if d.len() < hs {
        return;
    }
    let crc = crc32c::crc32c_append(crc32c::crc32c(&d[..32]), &d[40..hs]);
    d[32..36].copy_from_slice(&crc.to_le_bytes());
    let io = info.index_offset();
    for k in 0..info.num_blocks as usize {
        let a = u64::from_le_bytes(d[io + 8 * k..io + 8 * k + 8].try_into().unwrap()) as usize;
        let z = u64::from_le_bytes(d[io + 8 * k + 8..io + 8 * k + 16].try_into().unwrap()) as usize;
        if a.checked_add(4).is_some_and(|a4| a4 <= z) && z <= d.len() {
            let c = crc32c::crc32c(&d[a + 4..z]);
            d[a..a + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
    let _ = u32_at; // keep helper for debugging
}

fn entries() -> Vec<Vec<u8>> {
    let mut rng = Rng(12345);
    let mut out = vec![];
    for (frames, ch, bps, level, block) in
        [(700, 3, 16, 5, 256), (300, 1, 24, 8, 64), (513, 2, 8, 0, 128), (200, 4, 32, 6, 100)]
    {
        let hi = (1i64 << (bps - 1)) - 1;
        let mut walk = vec![0i64; ch];
        let x: Vec<i32> = (0..frames * ch)
            .map(|i| {
                let c = i % ch;
                walk[c] = (walk[c] + (rng.next() % 2001) as i64 - 1000).clamp(-hi - 1, hi);
                if c == 1 { (rng.next() as i64 % hi) as i32 } else { walk[c] as i32 }
            })
            .collect();
        let cfg = EncoderConfig { block_size: block, level, threads: 1 };
        out.push(encode(&x, ch, 16000, bps, None, &cfg).unwrap());
    }
    out
}

fn mutate(rng: &mut Rng, d: &mut Vec<u8>) {
    for _ in 0..1 + rng.below(4) {
        match rng.below(6) {
            0 => {
                let i = rng.below(d.len());
                d[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(d.len());
                d[i] = rng.next() as u8;
            }
            2 => {
                let i = rng.below(d.len());
                d[i] = [0x00, 0xff, 0x7f, 0x80][rng.below(4)];
            }
            3 => {
                let n = rng.below(d.len());
                d.truncate(n.max(1));
            }
            4 => {
                let i = rng.below(d.len());
                d.insert(i, rng.next() as u8);
            }
            _ => {
                let i = rng.below(d.len());
                let n = rng.below(16).min(d.len() - i);
                d.drain(i..i + n);
            }
        }
        if d.is_empty() {
            d.push(0);
        }
    }
}

#[test]
fn mutated_entries_never_panic() {
    let base = entries();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let iters: usize = std::env::var("OMAC_MUTATIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let mut decoded_ok = 0;
    for it in 0..iters {
        let mut d = base[it % base.len()].clone();
        mutate(&mut rng, &mut d);
        if it % 2 == 0 {
            fix_crcs(&mut d);
        }
        let verify = it % 3 != 0;
        let r = catch_unwind(AssertUnwindSafe(|| {
            let ok = decode_all(&d, verify, 1).is_ok();
            // Also the range path with a subset, as an indexed reader calls it.
            if let Ok(h) = Header::parse(&d) {
                let n = h.info.num_frames;
                let start = h.block_offsets[0] as usize;
                if start <= d.len() && n > 0 {
                    let mut out = vec![0f32; (n as usize).min(1 << 20)];
                    let hi = (out.len() as u64).min(n);
                    let _ = decode_range(&h.info, &d[start..], &h.block_offsets, 0, &[0], 0, hi, verify, 1, &mut out[..hi as usize]);
                }
            }
            ok
        }));
        match r {
            Ok(true) => decoded_ok += 1,
            Ok(false) => {}
            Err(_) => panic!("decoder panicked on mutation {it}: {:02x?}", &d[..d.len().min(64)]),
        }
    }
    // Some mutations hit only padding bits or produce other valid streams.
    assert!(decoded_ok < iters, "no mutation was rejected");
}

/// Inputs that once made the decoder panic (found by `cargo fuzz`), run the
/// way the fuzz target runs them.
#[test]
fn fuzz_regressions() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fuzz_regressions");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let mut d = std::fs::read(&path).unwrap();
        fix_crcs(&mut d);
        let r = catch_unwind(AssertUnwindSafe(|| decode_all(&d, true, 1).is_ok()));
        assert!(r.is_ok(), "decoder panicked on {}", path.display());
    }
}
