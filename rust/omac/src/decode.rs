//! Decoder (spec sections 4 and 5).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::bits::BitReader;
use crate::format::*;
use crate::Error;

/// Thread pool with `threads` workers, created once per size per process.
pub(crate) fn pool(threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS.get_or_init(Default::default).lock().unwrap();
    pools
        .entry(threads)
        .or_insert_with(|| {
            Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap())
        })
        .clone()
}

/// Output sample type. `scale` is `2^-(bps-1)`, `shift` is `16 - bps`.
pub trait Sample: Copy + Send + Sync + Default {
    fn convert(v: i32, scale: f32, shift: u32) -> Self;
}

impl Sample for i32 {
    #[inline]
    fn convert(v: i32, _: f32, _: u32) -> Self {
        v
    }
}

impl Sample for f32 {
    #[inline]
    fn convert(v: i32, scale: f32, _: u32) -> Self {
        v as f32 * scale
    }
}

/// Left-aligned to 16 bits; only for `bits_per_sample <= 16`.
impl Sample for i16 {
    #[inline]
    fn convert(v: i32, _: f32, shift: u32) -> Self {
        (v << shift) as i16
    }
}

fn read_residual(r: &mut BitReader, n: usize, res: &mut Vec<i64>) -> Result<(), Error> {
    let po = r.read(4)? as u32;
    if n % (1usize << po) != 0 {
        return Err(Error::Corrupt("partition order does not divide block length"));
    }
    let part = n >> po;
    res.clear();
    res.resize(n.saturating_sub(1), 0);
    let mut at = 0;
    for j in 0..(1usize << po) {
        let cnt = if j == 0 { part - 1 } else { part };
        let dst = &mut res[at..at + cnt];
        at += cnt;
        let k = r.read(5)? as u32;
        if k == RICE_ESCAPE {
            let width = r.read(6)? as u32;
            for d in dst.iter_mut() {
                *d = r.read_signed(width)?;
            }
        } else {
            r.read_rice_run(k, dst)?;
        }
    }
    Ok(())
}

#[inline]
fn check(x: i64, lo: i64, hi: i64) -> Result<i32, Error> {
    if x < lo || x > hi {
        return Err(Error::Corrupt("decoded sample out of range"));
    }
    Ok(x as i32)
}

/// Apply the predictor to residuals `res = r[1..n]`, given `out[0] = x[0]`.
fn predict_fixed(p: usize, res: &[i64], out: &mut [i32], lo: i64, hi: i64) -> Result<(), Error> {
    let n = out.len();
    for i in 1..n {
        let x = |j: usize| out[i - j] as i64;
        let guess = if i < p {
            if i == 1 { x(1) } else { 2 * x(1) - x(2) }
        } else {
            match p {
                0 => 0,
                1 => x(1),
                2 => 2 * x(1) - x(2),
                3 => 3 * x(1) - 3 * x(2) + x(3),
                _ => 4 * x(1) - 6 * x(2) + 4 * x(3) - x(4),
            }
        };
        let v = guess.checked_add(res[i - 1]).ok_or(Error::Corrupt("residual overflow"))?;
        out[i] = check(v, lo, hi)?;
    }
    Ok(())
}

fn predict_lpc(
    coefs: &[i64],
    shift: u32,
    res: &[i64],
    out: &mut [i32],
    lo: i64,
    hi: i64,
) -> Result<(), Error> {
    let n = out.len();
    let p = coefs.len();
    let warm = p.min(n);
    for i in 1..warm {
        let guess = if i == 1 { out[0] as i64 } else { 2 * out[i - 1] as i64 - out[i - 2] as i64 };
        let v = guess.checked_add(res[i - 1]).ok_or(Error::Corrupt("residual overflow"))?;
        out[i] = check(v, lo, hi)?;
    }
    if n <= p {
        return Ok(());
    }
    // coefs reversed so that the history window out[i-p..i] lines up with it.
    let mut rev = [0i64; MAX_LPC_ORDER];
    for (r, c) in rev.iter_mut().zip(coefs.iter().rev()) {
        *r = *c;
    }
    // The dot product fits in i32 when sum|coef| * 2^(w-1) < 2^31, w being
    // the coded width (|x| <= 2^(w-1)). Checked per payload, so this is exact.
    let w_minus_1 = (hi + 1).trailing_zeros();
    let sum_abs: i64 = coefs.iter().map(|c| c.abs()).sum();
    let narrow = w_minus_1 < 31 && (sum_abs << w_minus_1) < (1i64 << 31);
    macro_rules! dispatch {
        ($($n:literal)*) => {
            if narrow {
                let mut rev32 = [0i32; MAX_LPC_ORDER];
                for (a, b) in rev32.iter_mut().zip(&rev) {
                    *a = *b as i32;
                }
                match p {
                    $($n => lpc_loop32::<$n>(&rev32[..$n], shift, res, out, lo, hi),)*
                    _ => unreachable!(),
                }
            } else {
                match p {
                    $($n => lpc_loop::<$n>(&rev[..$n], shift, res, out, lo, hi),)*
                    _ => unreachable!(),
                }
            }
        };
    }
    dispatch!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

/// `lpc_loop` with 32-bit products, for payloads where they cannot overflow.
#[inline(always)]
fn lpc_loop32<const P: usize>(
    rev: &[i32],
    shift: u32,
    res: &[i64],
    out: &mut [i32],
    lo: i64,
    hi: i64,
) -> Result<(), Error> {
    let rev: &[i32; P] = rev.try_into().unwrap();
    let mut prev = out[P - 1];
    for i in P..out.len() {
        let hist: &[i32; P] = out[i - P..i].try_into().unwrap();
        let mut s = 0i32;
        for j in 0..P - 1 {
            s += hist[j] * rev[j];
        }
        s += prev * rev[P - 1];
        let v = ((s >> shift) as i64).checked_add(res[i - 1]).ok_or(Error::Corrupt("residual overflow"))?;
        prev = check(v, lo, hi)?;
        out[i] = prev;
    }
    Ok(())
}

/// LPC prediction for `i >= P`, with the order known at compile time so the
/// dot product is unrolled.
#[inline(always)]
fn lpc_loop<const P: usize>(
    rev: &[i64],
    shift: u32,
    res: &[i64],
    out: &mut [i32],
    lo: i64,
    hi: i64,
) -> Result<(), Error> {
    let rev: &[i64; P] = rev.try_into().unwrap();
    // The newest sample is kept in a register and added last, so the
    // loop-carried dependency is one multiply-add, not a store, a reload and
    // the whole dot product.
    let mut prev = out[P - 1] as i64;
    for i in P..out.len() {
        let hist: &[i32; P] = out[i - P..i].try_into().unwrap();
        let mut s = 0i64;
        for j in 0..P - 1 {
            s += hist[j] as i64 * rev[j];
        }
        s += prev * rev[P - 1];
        let v = (s >> shift).checked_add(res[i - 1]).ok_or(Error::Corrupt("residual overflow"))?;
        out[i] = check(v, lo, hi)?;
        prev = v;
    }
    Ok(())
}

/// Decode one channel payload of `out.len()` samples (spec section 5).
pub fn decode_payload(
    payload: &[u8],
    bps: u32,
    out: &mut [i32],
    res: &mut Vec<i64>,
) -> Result<(), Error> {
    let n = out.len();
    let mut r = BitReader::new(payload);
    let typ = r.read(2)?;
    let wasted = r.read(6)? as u32;
    if wasted >= bps {
        return Err(Error::Corrupt("wasted bits >= bits_per_sample"));
    }
    let w = bps - wasted;
    let lo = -(1i64 << (w - 1));
    let hi = (1i64 << (w - 1)) - 1;
    match typ {
        TYPE_CONSTANT => {
            let v = r.read_signed(w)? as i32;
            out.fill(v);
        }
        TYPE_VERBATIM => {
            for o in out.iter_mut() {
                *o = r.read_signed(w)? as i32;
            }
        }
        TYPE_FIXED => {
            let p = r.read(3)? as usize;
            if p > MAX_FIXED_ORDER {
                return Err(Error::Corrupt("fixed order > 4"));
            }
            out[0] = r.read_signed(w)? as i32;
            read_residual(&mut r, n, res)?;
            predict_fixed(p, res, out, lo, hi)?;
        }
        _ => {
            let p = r.read(5)? as usize + 1;
            let prec = r.read(4)? as u32 + 1;
            if prec > MAX_PREC {
                return Err(Error::Corrupt("LPC precision > 15"));
            }
            let shift = r.read(5)? as u32;
            let mut coefs = [0i64; MAX_LPC_ORDER];
            for c in coefs[..p].iter_mut() {
                *c = r.read_signed(prec)?;
            }
            out[0] = r.read_signed(w)? as i32;
            read_residual(&mut r, n, res)?;
            predict_lpc(&coefs[..p], shift, res, out, lo, hi)?;
        }
    }
    if wasted > 0 {
        for o in out.iter_mut() {
            *o <<= wasted;
        }
    }
    Ok(())
}

/// Byte ranges of the selected channels' payloads inside a block.
fn payload_ranges(
    block: &[u8],
    channels: usize,
    sel: &[usize],
    verify: bool,
) -> Result<Vec<(usize, usize)>, Error> {
    let table = 4 + 4 * channels;
    if block.len() < table {
        return Err(Error::Truncated);
    }
    if verify {
        let stored = u32::from_le_bytes(block[..4].try_into().unwrap());
        if crc32c::crc32c(&block[4..]) != stored {
            return Err(Error::Corrupt("block CRC mismatch"));
        }
    }
    let mut starts = Vec::with_capacity(channels + 1);
    let mut pos = table;
    for c in 0..channels {
        starts.push(pos);
        let sz = u32::from_le_bytes(block[4 + 4 * c..8 + 4 * c].try_into().unwrap()) as usize;
        if sz == 0 {
            return Err(Error::Corrupt("empty channel payload"));
        }
        pos += sz;
    }
    if pos != block.len() {
        return Err(Error::Corrupt("payload sizes do not add up to the block size"));
    }
    starts.push(pos);
    Ok(sel.iter().map(|&c| (starts[c], starts[c + 1])).collect())
}

struct Job<'a, T> {
    block: &'a [u8],
    n: usize,
    r0: usize,
    r1: usize,
    out: &'a mut [T],
}

fn run_job<T: Sample>(
    job: Job<T>,
    info: &StreamInfo,
    sel: &[usize],
    verify: bool,
    buf: &mut (Vec<i32>, Vec<i64>),
) -> Result<(), Error> {
    let (scale, shift) = conv(info.bits_per_sample);
    let nsel = sel.len();
    let ranges = payload_ranges(job.block, info.channels as usize, sel, verify)?;
    buf.0.resize(job.n, 0);
    for (s, &(a, b)) in ranges.iter().enumerate() {
        decode_payload(&job.block[a..b], info.bits_per_sample, &mut buf.0, &mut buf.1)?;
        for (i, &v) in buf.0[job.r0..job.r1].iter().enumerate() {
            job.out[i * nsel + s] = T::convert(v, scale, shift);
        }
    }
    Ok(())
}

/// The same block, channels decoded in parallel (for windows of few blocks).
fn run_job_par<T: Sample>(
    job: Job<T>,
    info: &StreamInfo,
    sel: &[usize],
    verify: bool,
) -> Result<(), Error> {
    let (scale, shift) = conv(info.bits_per_sample);
    let nsel = sel.len();
    let ranges = payload_ranges(job.block, info.channels as usize, sel, verify)?;
    let decoded: Vec<Vec<i32>> = ranges
        .par_iter()
        .map_init(Vec::new, |res, &(a, b)| {
            let mut out = vec![0i32; job.n];
            decode_payload(&job.block[a..b], info.bits_per_sample, &mut out, res).map(|_| out)
        })
        .collect::<Result<_, _>>()?;
    for (s, ch) in decoded.iter().enumerate() {
        for (i, &v) in ch[job.r0..job.r1].iter().enumerate() {
            job.out[i * nsel + s] = T::convert(v, scale, shift);
        }
    }
    Ok(())
}

fn conv(bps: u32) -> (f32, u32) {
    ((-(bps as i32 - 1) as f32).exp2(), 16u32.saturating_sub(bps))
}

/// Decode frames `[frame_lo, frame_hi)` of channels `sel` into `out`
/// (`(frame_hi - frame_lo) x sel.len()`, row-major).
///
/// `data` holds the entry bytes from `offsets[0]` on; `offsets` are the
/// entry's `block_offset[first_block ..= first_block + m]` for the `m`
/// blocks in `data`, which must cover the window.
#[allow(clippy::too_many_arguments)]
pub fn decode_range<T: Sample>(
    info: &StreamInfo,
    data: &[u8],
    offsets: &[u64],
    first_block: u64,
    sel: &[usize],
    frame_lo: u64,
    frame_hi: u64,
    verify: bool,
    threads: usize,
    out: &mut [T],
) -> Result<(), Error> {
    let nsel = sel.len();
    if frame_lo > frame_hi || frame_hi > info.num_frames {
        return Err(Error::Invalid(format!(
            "frames [{frame_lo}, {frame_hi}) outside [0, {})",
            info.num_frames
        )));
    }
    if let Some(&c) = sel.iter().find(|&&c| c >= info.channels as usize) {
        return Err(Error::Invalid(format!("channel {c} >= channels {}", info.channels)));
    }
    if std::mem::size_of::<T>() == 2 && info.bits_per_sample > 16 {
        return Err(Error::Invalid("int16 output needs bits_per_sample <= 16".into()));
    }
    if Some(out.len() as u64) != (frame_hi - frame_lo).checked_mul(nsel as u64) {
        return Err(Error::Invalid("output buffer has the wrong size".into()));
    }
    if frame_lo == frame_hi || nsel == 0 {
        return Ok(());
    }
    let b = info.block_size as u64;
    let (k0, k1) = (frame_lo / b, (frame_hi - 1) / b);
    if k0 < first_block || k1 + 1 >= first_block.saturating_add(offsets.len() as u64) {
        return Err(Error::Invalid("offsets do not cover the requested frames".into()));
    }
    let base = offsets[0];
    let mut jobs = Vec::with_capacity((k1 - k0 + 1) as usize);
    let mut rest = out;
    for k in k0..=k1 {
        let i = (k - first_block) as usize;
        let (a, z) = (offsets[i], offsets[i + 1]);
        if a < base || z < a || z - base > data.len() as u64 {
            return Err(Error::Invalid("block offsets outside the data".into()));
        }
        let r0 = (frame_lo.max(k * b) - k * b) as usize;
        let r1 = (frame_hi.min((k + 1) * b) - k * b) as usize;
        let (mine, tail) = rest.split_at_mut((r1 - r0) * nsel);
        rest = tail;
        jobs.push(Job {
            block: &data[(a - base) as usize..(z - base) as usize],
            n: info.block_frames(k),
            r0,
            r1,
            out: mine,
        });
    }
    if threads <= 1 {
        let mut buf = (Vec::new(), Vec::new());
        jobs.into_iter().try_for_each(|j| run_job(j, info, sel, verify, &mut buf))
    } else if jobs.len() < threads && nsel > 1 {
        pool(threads).install(|| jobs.into_iter().try_for_each(|j| run_job_par(j, info, sel, verify)))
    } else {
        pool(threads).install(|| {
            jobs.into_par_iter()
                .try_for_each_init(|| (Vec::new(), Vec::new()), |buf, j| run_job(j, info, sel, verify, buf))
        })
    }
}

/// Decode a whole entry: header and every frame of every channel.
pub fn decode_all(data: &[u8], verify: bool, threads: usize) -> Result<(Header, Vec<i32>), Error> {
    let h = Header::parse(data)?;
    let info = h.info;
    if *h.block_offsets.last().unwrap() != data.len() as u64 {
        return Err(Error::Invalid("entry size does not match the block index".into()));
    }
    let c = info.channels as usize;
    let sel: Vec<usize> = (0..c).collect();
    let total = (info.num_frames as usize)
        .checked_mul(c)
        .ok_or(Error::Invalid("entry too large to decode in memory".into()))?;
    let mut out = vec![0i32; total];
    let start = h.block_offsets[0] as usize;
    decode_range(
        &info,
        &data[start..],
        &h.block_offsets,
        0,
        &sel,
        0,
        info.num_frames,
        verify,
        threads,
        &mut out,
    )?;
    Ok((h, out))
}

