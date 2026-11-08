use omac::{decode_all, decode_range, encode, EncoderConfig, Error, Header};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

fn clamp_bps(v: i64, bps: u32) -> i32 {
    let hi = (1i64 << (bps - 1)) - 1;
    v.clamp(-hi - 1, hi) as i32
}

/// Several signal kinds, one per channel, interleaved.
fn signal(frames: usize, channels: usize, bps: u32, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let amp = ((1i64 << (bps - 1)) - 1) as f64;
    let mut out = vec![0i32; frames * channels];
    for c in 0..channels {
        let mut walk = 0f64;
        for i in 0..frames {
            let v = match c % 6 {
                0 => rng.range(-(amp as i64) - 1, amp as i64), // full-scale noise
                1 => ((i as f64 * 0.01 * (c + 1) as f64).sin() * amp * 0.9) as i64,
                2 => {
                    walk = 0.999 * walk + rng.range(-1000, 1000) as f64 * amp / 32768.0;
                    walk as i64
                }
                3 => if (i / 37) % 2 == 0 { amp as i64 } else { -(amp as i64) - 1 }, // square
                4 => 0,
                _ => (rng.range(-500, 500) * 256).clamp(-(amp as i64), amp as i64), // wasted bits
            };
            out[i * channels + c] = clamp_bps(v, bps);
        }
    }
    out
}

fn roundtrip(frames: usize, channels: usize, bps: u32, block: u32, level: u8, threads: usize) {
    let x = signal(frames, channels, bps, (frames * 31 + channels) as u64 + bps as u64);
    let cfg = EncoderConfig { block_size: block, level, threads };
    let enc = encode(&x, channels, 16000, bps, None, &cfg).unwrap();
    let (h, y) = decode_all(&enc, true, threads).unwrap();
    assert_eq!(h.info.num_frames, frames as u64);
    assert_eq!(h.info.channels, channels as u32);
    assert!(x == y, "mismatch frames={frames} ch={channels} bps={bps} block={block} level={level}");
    // Same bytes whatever the thread count.
    let enc1 = encode(&x, channels, 16000, bps, None, &EncoderConfig { threads: 1, ..cfg }).unwrap();
    assert!(enc == enc1, "encoder output depends on threads");
}

#[test]
fn bit_depths_and_levels() {
    for bps in [4, 8, 16, 24, 32] {
        for level in [0, 5, 8] {
            roundtrip(10_000, 6, bps, 4096, level, 1);
        }
    }
}

#[test]
fn odd_lengths() {
    for frames in [0, 1, 2, 3, 15, 16, 17, 4095, 4096, 4097, 8191] {
        roundtrip(frames, 3, 16, 4096, 5, 1);
    }
    for block in [16, 17, 1000, 65536] {
        roundtrip(70_001, 2, 16, block, 5, 1);
    }
}

#[test]
fn many_channels_and_threads() {
    roundtrip(5000, 300, 16, 1024, 3, 4);
    roundtrip(20_000, 16, 24, 4096, 5, 8);
}

#[test]
fn windows_and_channel_subsets() {
    let (frames, c) = (50_000usize, 7usize);
    let x = signal(frames, c, 16, 7);
    let enc = encode(&x, c, 16000, 16, None, &EncoderConfig { block_size: 4096, ..Default::default() })
        .unwrap();
    let h = Header::parse(&enc).unwrap();
    let start = h.block_offsets[0] as usize;
    for (lo, hi) in [(0, 1), (0, 50_000), (4095, 4097), (12_345, 33_333), (49_999, 50_000), (100, 100)] {
        for sel in [vec![0], vec![6, 2], (0..c).collect::<Vec<_>>()] {
            for threads in [1, 3] {
                let mut out = vec![0i32; (hi - lo) * sel.len()];
                decode_range(&h.info, &enc[start..], &h.block_offsets, 0, &sel, lo as u64, hi as u64, true, threads, &mut out)
                    .unwrap();
                for i in lo..hi {
                    for (s, &ch) in sel.iter().enumerate() {
                        assert_eq!(out[(i - lo) * sel.len() + s], x[i * c + ch]);
                    }
                }
            }
        }
    }
    // Only the blocks of the window, as a range reader would fetch them.
    let (k0, k1) = (3usize, 5usize);
    let data = &enc[h.block_offsets[k0] as usize..h.block_offsets[k1 + 1] as usize];
    let (lo, hi) = (3 * 4096 + 10, 6 * 4096 - 1);
    let mut out = vec![0f32; hi - lo];
    decode_range(&h.info, data, &h.block_offsets[k0..=k1 + 1], k0 as u64, &[4], lo as u64, hi as u64, true, 1, &mut out)
        .unwrap();
    for i in lo..hi {
        assert_eq!(out[i - lo], x[i * c + 4] as f32 / 32768.0);
    }
}

#[test]
fn padded_flag() {
    let x = signal(1000, 2, 16, 3);
    let enc = encode(&x, 2, 8000, 16, Some(&[1000, 900]), &EncoderConfig::default()).unwrap();
    let h = Header::parse(&enc).unwrap();
    assert_eq!(h.valid_frames, vec![1000, 900]);
    assert_eq!(h.info.flags & 1, 1);
}

#[test]
fn rejects_bad_input_and_corruption() {
    assert!(encode(&[40000], 1, 8000, 16, None, &EncoderConfig::default()).is_err());
    let x = signal(20_000, 2, 16, 5);
    let enc = encode(&x, 2, 8000, 16, None, &EncoderConfig::default()).unwrap();
    // Flip one bit in every region of the entry: header, index, payloads.
    for pos in [2usize, 30, 50, 70, enc.len() / 2, enc.len() - 1] {
        let mut bad = enc.clone();
        bad[pos] ^= 0x10;
        assert!(decode_all(&bad, true, 1).is_err(), "corruption at {pos} not detected");
    }
    for cut in [0usize, 10, 39, 100, enc.len() - 1] {
        let r = decode_all(&enc[..cut], true, 1);
        assert!(r.is_err(), "truncation at {cut} not detected");
    }
    assert!(matches!(decode_all(b"fLaC....", true, 1), Err(Error::Truncated) | Err(Error::Invalid(_))));
}
