//! Encoder-side linear prediction analysis (non-normative, spec section 9).

/// Tukey(0.5) window of length n, as FLAC uses by default.
pub fn tukey(n: usize, p: f64) -> Vec<f64> {
    let mut w = vec![1.0; n];
    let np = ((p / 2.0) * n as f64) as usize;
    if np > 1 {
        for i in 0..np {
            let v = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / (np - 1) as f64).cos();
            w[i] = v;
            w[n - 1 - i] = v;
        }
    }
    w
}

/// Autocorrelation lags 0..=max_lag of the windowed signal.
pub fn autocorrelation(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    (0..=max_lag)
        .map(|l| {
            if l >= n {
                0.0
            } else {
                x[..n - l].iter().zip(&x[l..]).map(|(a, b)| a * b).sum()
            }
        })
        .collect()
}

/// Levinson-Durbin. Returns, for each order 1..=max_order that could be
/// computed, the predictor `c` with `x[i] ~ sum_j c[j] * x[i-1-j]` and the
/// prediction error energy.
pub fn levinson(autoc: &[f64], max_order: usize) -> Vec<(Vec<f64>, f64)> {
    let mut out = Vec::with_capacity(max_order);
    let mut err = autoc[0];
    if err <= 0.0 {
        return out;
    }
    let mut lpc = vec![0.0f64; max_order];
    for i in 0..max_order {
        let mut r = -autoc[i + 1];
        for j in 0..i {
            r -= lpc[j] * autoc[i - j];
        }
        r /= err;
        lpc[i] = r;
        let mut j = 0;
        while j < (i >> 1) {
            let tmp = lpc[j];
            lpc[j] += r * lpc[i - 1 - j];
            lpc[i - 1 - j] += r * tmp;
            j += 1;
        }
        if i & 1 == 1 {
            lpc[j] += lpc[j] * r;
        }
        err *= 1.0 - r * r;
        out.push((lpc[..=i].iter().map(|v| -v).collect(), err.max(0.0)));
        if err <= 0.0 || !err.is_finite() {
            break;
        }
    }
    out
}

/// Quantize to `prec`-bit signed integers with a right shift, carrying the
/// rounding error forward. None if the coefficients are too large to code.
pub fn quantize(c: &[f64], prec: u32) -> Option<(Vec<i32>, u32)> {
    let cmax = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if cmax <= 0.0 || !cmax.is_finite() {
        return None;
    }
    // cmax < 2^e
    let e = cmax.log2().floor() as i32 + 1;
    let shift = prec as i32 - 1 - e;
    if shift < 0 {
        return None;
    }
    let shift = shift.min(31) as u32;
    let qmax = (1i64 << (prec - 1)) - 1;
    let qmin = -(1i64 << (prec - 1));
    let scale = (1u64 << shift) as f64;
    let mut carry = 0.0;
    let q = c
        .iter()
        .map(|&v| {
            carry += v * scale;
            let r = carry.round().clamp(qmin as f64, qmax as f64);
            carry -= r;
            r as i32
        })
        .collect();
    Some((q, shift))
}
