//! Python bindings (module `omac`).

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::{Error, Header, Sample, StreamInfo};

fn err(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Encode an int32 array of shape (frames, channels) to an OMAC entry.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, bits_per_sample, block_size=4096, level=5, valid_frames=None, threads=1))]
fn encode<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, i32>,
    sample_rate: u32,
    bits_per_sample: u32,
    block_size: u32,
    level: u8,
    valid_frames: Option<Vec<u64>>,
    threads: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let channels = samples.shape()[1];
    let owned;
    let data: &[i32] = match samples.as_slice() {
        Ok(s) => s,
        Err(_) => {
            owned = samples.as_array().iter().copied().collect::<Vec<_>>();
            &owned
        }
    };
    let cfg = crate::EncoderConfig { block_size, level, threads };
    let out = py
        .detach(|| crate::encode(data, channels, sample_rate, bits_per_sample, valid_frames.as_deref(), &cfg))
        .map_err(err)?;
    Ok(PyBytes::new(py, &out))
}

fn header_dict<'py>(py: Python<'py>, info: &StreamInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("channels", info.channels)?;
    d.set_item("bits_per_sample", info.bits_per_sample)?;
    d.set_item("sample_rate", info.sample_rate)?;
    d.set_item("block_size", info.block_size)?;
    d.set_item("num_blocks", info.num_blocks)?;
    d.set_item("num_frames", info.num_frames)?;
    d.set_item("padded", info.flags & crate::format::FLAG_PADDED != 0)?;
    Ok(d)
}

/// Parse the 40-byte fixed header.
#[pyfunction]
fn read_info<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    header_dict(py, &StreamInfo::parse(data).map_err(err)?)
}

/// Parse the full header (fixed fields, valid_frames, block index), checking its CRC.
#[pyfunction]
fn read_header<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let h = Header::parse(data).map_err(err)?;
    let d = header_dict(py, &h.info)?;
    d.set_item("valid_frames", h.valid_frames)?;
    d.set_item("block_offsets", h.block_offsets)?;
    Ok(d)
}

fn run<'py, T: Sample + numpy::Element>(
    py: Python<'py>,
    info: &StreamInfo,
    data: &[u8],
    offsets: &[u64],
    first_block: u64,
    sel: &[usize],
    lo: u64,
    hi: u64,
    verify: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = (hi.saturating_sub(lo)) as usize;
    let mut out = vec![T::default(); rows * sel.len()];
    py.detach(|| crate::decode_range(info, data, offsets, first_block, sel, lo, hi, verify, threads, &mut out))
        .map_err(err)?;
    let arr: Bound<'py, PyArray2<T>> = out.into_pyarray(py).reshape([rows, sel.len()])?;
    Ok(arr.into_any())
}

/// Decode frames [frame_lo, frame_hi) of the selected channels.
///
/// `info` is the entry's 40-byte fixed header; `data` holds the entry from
/// `offsets[0]` on, and `offsets` are `block_offset[first_block ..= first_block + m]`.
/// Returns (frames, channels) as float32 (in [-1, 1)), int32 or int16.
#[pyfunction]
#[pyo3(signature = (info, data, offsets, first_block, frame_lo, frame_hi, channels=None, dtype="float32", verify=true, threads=1))]
fn decode_range<'py>(
    py: Python<'py>,
    info: &[u8],
    data: &[u8],
    offsets: Vec<u64>,
    first_block: u64,
    frame_lo: u64,
    frame_hi: u64,
    channels: Option<Vec<usize>>,
    dtype: &str,
    verify: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let info = StreamInfo::parse(info).map_err(err)?;
    let sel = channels.unwrap_or_else(|| (0..info.channels as usize).collect());
    let args = (&info, data, &offsets[..], first_block, &sel[..], frame_lo, frame_hi, verify, threads);
    match dtype {
        "float32" => run::<f32>(py, args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8),
        "int32" => run::<i32>(py, args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8),
        "int16" => run::<i16>(py, args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8),
        _ => Err(PyValueError::new_err(format!("dtype must be float32, int32 or int16, not {dtype}"))),
    }
}

/// Decode a whole entry. Returns (array, header dict).
#[pyfunction]
#[pyo3(signature = (data, channels=None, dtype="float32", verify=true, threads=1))]
fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    channels: Option<Vec<usize>>,
    dtype: &str,
    verify: bool,
    threads: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let h = Header::parse(data).map_err(err)?;
    if *h.block_offsets.last().unwrap() != data.len() as u64 {
        return Err(err(Error::Invalid("entry size does not match the block index".into())));
    }
    let start = h.block_offsets[0] as usize;
    let arr = decode_range(
        py, &data[..40], &data[start..], h.block_offsets.clone(), 0, 0, h.info.num_frames, channels, dtype, verify, threads,
    )?;
    let d = header_dict(py, &h.info)?;
    d.set_item("valid_frames", h.valid_frames)?;
    d.set_item("block_offsets", h.block_offsets)?;
    Ok((arr, d))
}

#[pymodule]
fn omac(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_range, m)?)?;
    m.add_function(wrap_pyfunction!(read_info, m)?)?;
    m.add_function(wrap_pyfunction!(read_header, m)?)?;
    Ok(())
}
