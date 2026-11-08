//! OMAC (Omniio Multichannel Audio Codec; unrelated to the cryptographic
//! One-Key MAC): lossless multichannel audio codec with random access in time and
//! in channels. The bitstream is specified in `docs/omac.md`.

pub mod bits;
pub mod decode;
pub mod encode;
pub mod format;
mod lpc;
#[cfg(feature = "python")]
mod python;

pub use decode::{decode_all, decode_range, Sample};
pub use encode::{encode, EncoderConfig};
pub use format::{Header, StreamInfo};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Input ended before a field did.
    Truncated,
    /// Bytes are not a valid entry (CRC, range or structure).
    Corrupt(&'static str),
    /// A field or argument is out of range.
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Truncated => write!(f, "OMAC data is truncated"),
            Error::Corrupt(m) => write!(f, "corrupt OMAC data: {m}"),
            Error::Invalid(m) => write!(f, "invalid OMAC data or argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Zigzag map of a signed residual (spec 5.3).
#[inline]
pub(crate) fn zigzag(r: i64) -> u64 {
    ((r << 1) ^ (r >> 63)) as u64
}

#[inline]
pub(crate) fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}
