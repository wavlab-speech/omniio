//! Entry header, channel lengths and block index (spec section 3).

use crate::Error;

pub const MAGIC: &[u8; 4] = b"OMAC";
pub const VERSION: u16 = 1;
pub const FIXED_HEADER_SIZE: usize = 40;
pub const FLAG_PADDED: u16 = 1;

pub const MIN_BLOCK: u32 = 16;
pub const MAX_BLOCK: u32 = 65536;
pub const MIN_BPS: u32 = 4;
pub const MAX_BPS: u32 = 32;

pub const TYPE_CONSTANT: u64 = 0;
pub const TYPE_VERBATIM: u64 = 1;
pub const TYPE_FIXED: u64 = 2;
pub const TYPE_LPC: u64 = 3;
pub const MAX_FIXED_ORDER: usize = 4;
pub const MAX_LPC_ORDER: usize = 32;
pub const MAX_PREC: u32 = 15;
pub const MAX_PARTITION_ORDER: u32 = 15;
pub const RICE_ESCAPE: u32 = 31;

/// The fields of the 40-byte fixed header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub flags: u16,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub sample_rate: u32,
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_frames: u64,
}

impl StreamInfo {
    pub fn new(
        channels: u32,
        bits_per_sample: u32,
        sample_rate: u32,
        block_size: u32,
        num_frames: u64,
    ) -> Result<Self, Error> {
        let num_blocks = num_frames.div_ceil(block_size.max(1) as u64);
        let info = StreamInfo {
            flags: 0,
            channels,
            bits_per_sample,
            sample_rate,
            block_size,
            num_blocks: u32::try_from(num_blocks)
                .map_err(|_| Error::Invalid("more than 2^32 - 1 blocks".into()))?,
            num_frames,
        };
        info.validate()?;
        Ok(info)
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.flags & !FLAG_PADDED != 0 {
            return Err(Error::Invalid(format!("unknown flags {:#x}", self.flags)));
        }
        if !(1..=65535).contains(&self.channels) {
            return Err(Error::Invalid(format!("channels {} not in 1..65535", self.channels)));
        }
        if !(MIN_BPS..=MAX_BPS).contains(&self.bits_per_sample) {
            return Err(Error::Invalid(format!(
                "bits_per_sample {} not in {MIN_BPS}..{MAX_BPS}",
                self.bits_per_sample
            )));
        }
        if self.sample_rate == 0 {
            return Err(Error::Invalid("sample_rate is 0".into()));
        }
        if !(MIN_BLOCK..=MAX_BLOCK).contains(&self.block_size) {
            return Err(Error::Invalid(format!(
                "block_size {} not in {MIN_BLOCK}..{MAX_BLOCK}",
                self.block_size
            )));
        }
        if self.num_frames.div_ceil(self.block_size as u64) != self.num_blocks as u64 {
            return Err(Error::Invalid("num_blocks does not match num_frames".into()));
        }
        Ok(())
    }

    /// Frames in block `k`.
    pub fn block_frames(&self, k: u64) -> usize {
        let b = self.block_size as u64;
        (self.num_frames - k * b).min(b) as usize
    }

    /// Offset of `block_offset[0]` in the entry.
    pub fn index_offset(&self) -> usize {
        FIXED_HEADER_SIZE + 8 * self.channels as usize
    }

    /// Size of everything before block 0.
    pub fn header_size(&self) -> usize {
        self.index_offset() + 8 * (self.num_blocks as usize + 1)
    }

    pub fn to_bytes(&self) -> [u8; FIXED_HEADER_SIZE] {
        let mut h = [0u8; FIXED_HEADER_SIZE];
        h[0..4].copy_from_slice(MAGIC);
        h[4..6].copy_from_slice(&VERSION.to_le_bytes());
        h[6..8].copy_from_slice(&self.flags.to_le_bytes());
        h[8..10].copy_from_slice(&(self.channels as u16).to_le_bytes());
        h[10] = self.bits_per_sample as u8;
        h[12..16].copy_from_slice(&self.sample_rate.to_le_bytes());
        h[16..20].copy_from_slice(&self.block_size.to_le_bytes());
        h[20..24].copy_from_slice(&self.num_blocks.to_le_bytes());
        h[24..32].copy_from_slice(&self.num_frames.to_le_bytes());
        h
    }

    /// Parse and validate the fixed header (does not check `header_crc`).
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.len() < FIXED_HEADER_SIZE {
            return Err(Error::Truncated);
        }
        if &data[0..4] != MAGIC {
            return Err(Error::Invalid("not an OMAC entry (bad magic)".into()));
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != VERSION {
            return Err(Error::Invalid(format!("unsupported OMAC version {version}")));
        }
        if data[11] != 0 || data[36..40] != [0, 0, 0, 0] {
            return Err(Error::Invalid("reserved header bytes are not zero".into()));
        }
        let info = StreamInfo {
            flags: u16::from_le_bytes([data[6], data[7]]),
            channels: u16::from_le_bytes([data[8], data[9]]) as u32,
            bits_per_sample: data[10] as u32,
            sample_rate: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            block_size: u32::from_le_bytes(data[16..20].try_into().unwrap()),
            num_blocks: u32::from_le_bytes(data[20..24].try_into().unwrap()),
            num_frames: u64::from_le_bytes(data[24..32].try_into().unwrap()),
        };
        info.validate()?;
        Ok(info)
    }
}

/// Header CRC over bytes `[0, 32)` and `[40, header_size)` of the entry.
pub fn header_crc(fixed: &[u8], tables: &[u8]) -> u32 {
    let c = crc32c::crc32c(&fixed[..32]);
    crc32c::crc32c_append(c, tables)
}

/// Full header: fixed fields, channel lengths and block index.
#[derive(Clone, Debug)]
pub struct Header {
    pub info: StreamInfo,
    pub valid_frames: Vec<u64>,
    pub block_offsets: Vec<u64>,
}

impl Header {
    /// Parse everything before block 0 and check `header_crc`.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let info = StreamInfo::parse(data)?;
        let hs = info.header_size();
        if data.len() < hs {
            return Err(Error::Truncated);
        }
        let stored = u32::from_le_bytes(data[32..36].try_into().unwrap());
        if header_crc(&data[..FIXED_HEADER_SIZE], &data[FIXED_HEADER_SIZE..hs]) != stored {
            return Err(Error::Corrupt("header CRC mismatch"));
        }
        let u64s = |lo: usize, n: usize| -> Vec<u64> {
            (0..n)
                .map(|i| u64::from_le_bytes(data[lo + 8 * i..lo + 8 * i + 8].try_into().unwrap()))
                .collect()
        };
        let c = info.channels as usize;
        let valid_frames = u64s(FIXED_HEADER_SIZE, c);
        let block_offsets = u64s(info.index_offset(), info.num_blocks as usize + 1);
        if valid_frames.iter().any(|&v| v > info.num_frames) {
            return Err(Error::Invalid("valid_frames exceeds num_frames".into()));
        }
        let padded = valid_frames.iter().any(|&v| v < info.num_frames);
        if padded != (info.flags & FLAG_PADDED != 0) {
            return Err(Error::Invalid("PADDED flag does not match valid_frames".into()));
        }
        if block_offsets[0] != hs as u64 {
            return Err(Error::Invalid("block_offset[0] does not follow the header".into()));
        }
        let min_block = 4 + 5 * c as u64; // crc, sizes, >= 1 byte per payload
        // checked: a corrupt offset near u64::MAX must not wrap around.
        if block_offsets.windows(2).any(|w| w[0].checked_add(min_block).is_none_or(|m| w[1] < m)) {
            return Err(Error::Invalid("block offsets not increasing".into()));
        }
        Ok(Header { info, valid_frames, block_offsets })
    }
}
