//! Fuzz the decoder: any input must decode or return an error, never panic.
//!
//!     cargo +nightly fuzz run decode -- -max_total_time=600
//!
//! Header and block CRCs are recomputed first, so the fuzzer explores the
//! parsers behind them instead of stopping at checksum failures.
#![no_main]

use libfuzzer_sys::fuzz_target;
use omac::{decode_all, decode_range, Header, StreamInfo};

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
        let at = |j: usize| u64::from_le_bytes(d[io + 8 * j..io + 8 * j + 8].try_into().unwrap()) as usize;
        let (a, z) = (at(k), at(k + 1));
        if a.checked_add(4).is_some_and(|a4| a4 <= z) && z <= d.len() {
            let c = crc32c::crc32c(&d[a + 4..z]);
            d[a..a + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut d = data.to_vec();
    fix_crcs(&mut d);
    // A valid entry may legitimately decode to far more than its size
    // (silence costs a few bytes per block), so cap the output, not the input.
    if let Ok(info) = StreamInfo::parse(&d) {
        if info.num_frames.saturating_mul(info.channels as u64) > 1 << 22 {
            return;
        }
    }
    let _ = decode_all(&d, true, 1);
    if let Ok(h) = Header::parse(&d) {
        let n = h.info.num_frames.min(1 << 16);
        let start = h.block_offsets[0] as usize;
        if n > 0 && start <= d.len() {
            let sel = [h.info.channels as usize - 1];
            let mut out = vec![0i32; n as usize];
            let _ = decode_range(&h.info, &d[start..], &h.block_offsets, 0, &sel, 0, n, false, 1, &mut out);
        }
    }
});
