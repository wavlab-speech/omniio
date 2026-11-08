//! Reference encoder. Bitstream choices follow spec section 5; the search
//! (which predictor, order, precision and partitioning) is non-normative.

use rayon::prelude::*;

use crate::bits::BitWriter;
use crate::format::*;
use crate::lpc;
use crate::{zigzag, Error};

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Frames per block.
    pub block_size: u32,
    /// 0 (fastest) ..= 8 (smallest).
    pub level: u8,
    /// Encoder threads; 1 encodes on the calling thread. Output does not
    /// depend on it.
    pub threads: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig { block_size: 4096, level: 5, threads: 1 }
    }
}

struct Params {
    max_lpc: usize,
    /// How many LPC orders, best by estimate, to code exactly; 0 = all.
    orders_tried: usize,
    precs: &'static [u32],
    max_po: u32,
}

fn params(level: u8) -> Params {
    match level {
        0 => Params { max_lpc: 0, orders_tried: 0, precs: &[], max_po: 4 },
        1 | 2 => Params { max_lpc: 8, orders_tried: 1, precs: &[14], max_po: 6 },
        3..=5 => Params { max_lpc: 12, orders_tried: 2, precs: &[14], max_po: 8 },
        6 => Params { max_lpc: 32, orders_tried: 3, precs: &[14], max_po: 8 },
        7 => Params { max_lpc: 32, orders_tried: 0, precs: &[14], max_po: 8 },
        _ => Params { max_lpc: 32, orders_tried: 0, precs: &[12, 13, 14, 15], max_po: 8 },
    }
}

#[derive(Clone)]
enum Predictor {
    Verbatim,
    Fixed(usize),
    Lpc { coefs: Vec<i32>, prec: u32, shift: u32 },
}

impl Predictor {
    fn order(&self) -> usize {
        match self {
            Predictor::Verbatim => 0,
            Predictor::Fixed(p) => *p,
            Predictor::Lpc { coefs, .. } => coefs.len(),
        }
    }
    fn header_bits(&self, w: u32) -> u64 {
        8 + match self {
            Predictor::Verbatim => 0,
            Predictor::Fixed(_) => 3 + w as u64,
            Predictor::Lpc { coefs, prec, .. } => 14 + coefs.len() as u64 * *prec as u64 + w as u64,
        }
    }
}

/// `r[i-1] = v[i] - pred(i)` for i in 1..n (spec 5.2).
fn residual(v: &[i64], pred: &Predictor, out: &mut Vec<i64>) {
    let n = v.len();
    let p = pred.order();
    out.clear();
    for i in 1..n {
        let guess = if i < p {
            if i == 1 { v[0] } else { 2 * v[i - 1] - v[i - 2] }
        } else {
            match pred {
                Predictor::Fixed(0) => 0,
                Predictor::Fixed(1) => v[i - 1],
                Predictor::Fixed(2) => 2 * v[i - 1] - v[i - 2],
                Predictor::Fixed(3) => 3 * v[i - 1] - 3 * v[i - 2] + v[i - 3],
                Predictor::Fixed(4) => 4 * v[i - 1] - 6 * v[i - 2] + 4 * v[i - 3] - v[i - 4],
                Predictor::Lpc { coefs, shift, .. } => {
                    let mut s = 0i64;
                    for (j, &c) in coefs.iter().enumerate() {
                        s += c as i64 * v[i - 1 - j];
                    }
                    s >> shift
                }
                _ => unreachable!(),
            }
        };
        out.push(v[i] - guess);
    }
}

/// Partitioned Rice plan for residual `r[1..n]`: (bits, po, per-partition k).
fn plan_rice(res: &[i64], n: usize, max_po: u32) -> (u64, u32, Vec<u32>) {
    let mut top = 0;
    while top < max_po.min(MAX_PARTITION_ORDER) && n % (1usize << (top + 1)) == 0 {
        top += 1;
    }
    let parts = 1usize << top;
    let part = n >> top;
    let mut sums = vec![0u128; parts];
    let mut maxs = vec![0u64; parts];
    let mut cnts = vec![0u64; parts];
    for (idx, &r) in res.iter().enumerate() {
        let j = (idx + 1) / part; // sample index i = idx + 1
        let u = zigzag(r);
        sums[j] += u as u128;
        maxs[j] |= u;
        cnts[j] += 1;
    }
    let mut best: Option<(u64, u32, Vec<u32>)> = None;
    let mut po = top;
    loop {
        let mut bits = 4u64;
        let mut ks = Vec::with_capacity(sums.len());
        for j in 0..sums.len() {
            let (b, k) = best_k(sums[j], cnts[j], maxs[j]);
            bits += b;
            ks.push(k);
        }
        if best.as_ref().is_none_or(|b| bits < b.0) {
            best = Some((bits, po, ks));
        }
        if po == 0 {
            break;
        }
        po -= 1;
        let merge = |v: &[u128]| v.chunks(2).map(|c| c[0] + c[1]).collect::<Vec<_>>();
        sums = merge(&sums);
        maxs = maxs.chunks(2).map(|c| c[0] | c[1]).collect();
        cnts = cnts.chunks(2).map(|c| c[0] + c[1]).collect();
    }
    best.unwrap()
}

/// Estimated bits and parameter for one partition (31 = escape).
fn best_k(sum: u128, cnt: u64, maxu: u64) -> (u64, u32) {
    if cnt == 0 {
        return (5, 0);
    }
    let width = 64 - maxu.leading_zeros() as u64;
    let mut best = (5 + 6 + cnt * width, RICE_ESCAPE);
    let mean = sum / cnt as u128;
    let k0 = if mean == 0 { 0 } else { 127 - mean.leading_zeros() };
    for k in k0.saturating_sub(1)..=(k0 + 1).min(30) {
        let est = sum >> k;
        let bits = 5u128 + cnt as u128 * (k as u128 + 1) + est;
        if bits < best.0 as u128 {
            best = (bits as u64, k);
        }
    }
    best
}

fn write_residual(w: &mut BitWriter, res: &[i64], n: usize, po: u32, ks: &[u32]) {
    w.write(po as u64, 4);
    let part = n >> po;
    let mut idx = 0;
    for (j, &k) in ks.iter().enumerate() {
        let cnt = if j == 0 { part - 1 } else { part };
        let chunk = &res[idx..idx + cnt];
        idx += cnt;
        w.write(k as u64, 5);
        if k == RICE_ESCAPE {
            let maxu = chunk.iter().fold(0u64, |m, &r| m | zigzag(r));
            let width = 64 - maxu.leading_zeros();
            w.write(width as u64, 6);
            for &r in chunk {
                w.write_signed(r, width);
            }
        } else {
            for &r in chunk {
                w.write_rice(zigzag(r), k);
            }
        }
    }
}

struct Scratch {
    v: Vec<i64>,
    res: Vec<i64>,
    win: Vec<f64>,
    win_n: usize,
    xw: Vec<f64>,
}

impl Scratch {
    fn new() -> Self {
        Scratch { v: vec![], res: vec![], win: vec![], win_n: 0, xw: vec![] }
    }
}

/// Code one channel of one block (spec section 5).
fn encode_payload(x: &[i64], bps: u32, prm: &Params, s: &mut Scratch) -> Vec<u8> {
    let n = x.len();
    let mut w = BitWriter::with_capacity(n * bps as usize / 8 + 16);
    if x.iter().all(|&v| v == x[0]) {
        w.write(TYPE_CONSTANT, 2);
        w.write(0, 6);
        w.write_signed(x[0], bps);
        return w.finish();
    }
    let or = x.iter().fold(0i64, |a, &v| a | v);
    let wasted = (or.trailing_zeros()).min(bps - 1);
    let wbits = bps - wasted;
    s.v.clear();
    s.v.extend(x.iter().map(|&v| v >> wasted));

    // (bits, predictor, rice plan)
    let mut best: (u64, Predictor, u32, Vec<u32>) =
        (8 + n as u64 * wbits as u64, Predictor::Verbatim, 0, vec![]);
    let consider = |pred: Predictor, s: &mut Scratch, best: &mut (u64, Predictor, u32, Vec<u32>)| {
        residual(&s.v, &pred, &mut s.res);
        let (rb, po, ks) = plan_rice(&s.res, n, prm.max_po);
        let bits = pred.header_bits(wbits) + rb;
        if bits < best.0 {
            *best = (bits, pred, po, ks);
        }
    };
    for p in 0..=MAX_FIXED_ORDER {
        consider(Predictor::Fixed(p), s, &mut best);
    }

    let max_order = prm.max_lpc.min(n.saturating_sub(1));
    if max_order > 0 {
        if s.win_n != n {
            s.win = lpc::tukey(n, 0.5);
            s.win_n = n;
        }
        s.xw.clear();
        s.xw.extend(s.v.iter().zip(&s.win).map(|(&v, &w)| v as f64 * w));
        let autoc = lpc::autocorrelation(&s.xw, max_order);
        let sols = lpc::levinson(&autoc, max_order);
        let mut orders: Vec<usize> = (0..sols.len()).collect();
        if prm.orders_tried > 0 && sols.len() > prm.orders_tried {
            let prec = prm.precs[0] as f64;
            let est = |i: usize| {
                let e = (sols[i].1 / n as f64).max(1e-9);
                0.5 * e.log2() * n as f64 + (i + 1) as f64 * prec
            };
            orders.sort_by(|&a, &b| est(a).total_cmp(&est(b)));
            orders.truncate(prm.orders_tried);
        }
        for &oi in &orders {
            for &prec in prm.precs {
                if let Some((coefs, shift)) = lpc::quantize(&sols[oi].0, prec) {
                    consider(Predictor::Lpc { coefs, prec, shift }, s, &mut best);
                }
            }
        }
    }

    let (_, pred, po, ks) = best;
    match &pred {
        Predictor::Verbatim => {
            w.write(TYPE_VERBATIM, 2);
            w.write(wasted as u64, 6);
            for &v in &s.v {
                w.write_signed(v, wbits);
            }
            return w.finish();
        }
        Predictor::Fixed(p) => {
            w.write(TYPE_FIXED, 2);
            w.write(wasted as u64, 6);
            w.write(*p as u64, 3);
        }
        Predictor::Lpc { coefs, prec, shift } => {
            w.write(TYPE_LPC, 2);
            w.write(wasted as u64, 6);
            w.write(coefs.len() as u64 - 1, 5);
            w.write(*prec as u64 - 1, 4);
            w.write(*shift as u64, 5);
            for &c in coefs {
                w.write_signed(c as i64, *prec);
            }
        }
    }
    w.write_signed(s.v[0], wbits);
    residual(&s.v, &pred, &mut s.res);
    write_residual(&mut w, &s.res, n, po, &ks);
    w.finish()
}

fn encode_block(
    samples: &[i32],
    channels: usize,
    bps: u32,
    prm: &Params,
    s: &mut Scratch,
) -> Vec<u8> {
    let n = samples.len() / channels;
    let mut x = Vec::with_capacity(n);
    let payloads: Vec<Vec<u8>> = (0..channels)
        .map(|c| {
            x.clear();
            x.extend((0..n).map(|i| samples[i * channels + c] as i64));
            encode_payload(&x, bps, prm, s)
        })
        .collect();
    let total = 4 + 4 * channels + payloads.iter().map(Vec::len).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&[0; 4]);
    for p in &payloads {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
    }
    for p in &payloads {
        out.extend_from_slice(p);
    }
    let crc = crc32c::crc32c(&out[4..]);
    out[..4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Encode interleaved samples (`frames x channels`, row-major) as one entry.
///
/// `valid_frames[c]` (default: all frames) marks zero padding at the end of
/// channel `c` (spec 3.2); the caller pads, this only records it.
pub fn encode(
    samples: &[i32],
    channels: usize,
    sample_rate: u32,
    bits_per_sample: u32,
    valid_frames: Option<&[u64]>,
    cfg: &EncoderConfig,
) -> Result<Vec<u8>, Error> {
    if channels == 0 || samples.len() % channels != 0 {
        return Err(Error::Invalid("samples length is not a multiple of channels".into()));
    }
    let frames = (samples.len() / channels) as u64;
    let mut info =
        StreamInfo::new(channels as u32, bits_per_sample, sample_rate, cfg.block_size, frames)?;
    let lo = -(1i64 << (bits_per_sample - 1));
    let hi = (1i64 << (bits_per_sample - 1)) - 1;
    if let Some(bad) = samples.iter().find(|&&v| (v as i64) < lo || (v as i64) > hi) {
        return Err(Error::Invalid(format!(
            "sample {bad} does not fit in {bits_per_sample} bits"
        )));
    }
    let valid: Vec<u64> = match valid_frames {
        None => vec![frames; channels],
        Some(v) => {
            if v.len() != channels || v.iter().any(|&f| f > frames) {
                return Err(Error::Invalid(
                    "valid_frames needs one value <= num_frames per channel".into(),
                ));
            }
            v.to_vec()
        }
    };
    if valid.iter().any(|&v| v < frames) {
        info.flags |= FLAG_PADDED;
    }

    let prm = params(cfg.level);
    let b = cfg.block_size as usize * channels;
    let chunks: Vec<&[i32]> = samples.chunks(b).collect();
    let blocks: Vec<Vec<u8>> = if cfg.threads <= 1 {
        let mut s = Scratch::new();
        chunks.iter().map(|c| encode_block(c, channels, bits_per_sample, &prm, &mut s)).collect()
    } else {
        crate::decode::pool(cfg.threads).install(|| {
            chunks
                .par_iter()
                .map_init(Scratch::new, |s, c| encode_block(c, channels, bits_per_sample, &prm, s))
                .collect()
        })
    };

    let hs = info.header_size();
    let mut tables = Vec::with_capacity(hs - FIXED_HEADER_SIZE);
    for v in &valid {
        tables.extend_from_slice(&v.to_le_bytes());
    }
    let mut off = hs as u64;
    tables.extend_from_slice(&off.to_le_bytes());
    for blk in &blocks {
        off += blk.len() as u64;
        tables.extend_from_slice(&off.to_le_bytes());
    }
    let mut fixed = info.to_bytes();
    let crc = header_crc(&fixed, &tables);
    fixed[32..36].copy_from_slice(&crc.to_le_bytes());

    let mut out = Vec::with_capacity(off as usize);
    out.extend_from_slice(&fixed);
    out.extend_from_slice(&tables);
    for blk in &blocks {
        out.extend_from_slice(blk);
    }
    Ok(out)
}
