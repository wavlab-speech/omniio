//! MSB-first bit writer and reader (spec section 2).

use crate::Error;

pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter { buf: Vec::with_capacity(bytes), acc: 0, nbits: 0 }
    }

    /// Write the low `n` bits of `value`, `n <= 32`.
    #[inline]
    fn put(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (value & ((1u64 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.buf.push((self.acc >> self.nbits) as u8);
        }
    }

    /// Write the low `n` bits of `value`, `n <= 64`.
    #[inline]
    pub fn write(&mut self, value: u64, n: u32) {
        if n > 32 {
            self.put(value >> 32, n - 32);
            self.put(value, 32);
        } else {
            self.put(value, n);
        }
    }

    /// Write `v` as an `n`-bit two's complement field.
    #[inline]
    pub fn write_signed(&mut self, v: i64, n: u32) {
        self.write(v as u64, n);
    }

    #[inline]
    pub fn write_rice(&mut self, u: u64, k: u32) {
        let mut q = u >> k;
        while q >= 32 {
            self.put(0, 32);
            q -= 32;
        }
        // q zeros, a one, then the low k bits.
        self.put(1, q as u32 + 1);
        self.put(u, k);
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put(0, pad);
        }
        self.buf
    }
}

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Left-aligned bit buffer; the top `bits` bits are valid, `bits <= 63`.
    acc: u64,
    bits: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, acc: 0, bits: 0 }
    }

    #[inline]
    fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let w = u64::from_be_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
            // Bits below `bits + 8 * nbytes` are later bytes, so OR-ing them
            // in again on the next refill writes the same values.
            self.acc |= w >> self.bits;
            let nbytes = (63 - self.bits) >> 3;
            self.pos += nbytes as usize;
            self.bits += nbytes * 8;
        } else {
            // Stop below 64 valid bits: callers shift by up to `bits`, and a
            // shift by 64 overflows (a panic in debug, a no-op in release).
            while self.bits < 56 && self.pos < self.data.len() {
                self.acc |= (self.data[self.pos] as u64) << (56 - self.bits);
                self.pos += 1;
                self.bits += 8;
            }
        }
    }

    /// Read `n <= 56` bits.
    #[inline]
    fn get(&mut self, n: u32) -> Result<u64, Error> {
        if n == 0 {
            return Ok(0);
        }
        if self.bits < n {
            self.refill();
            if self.bits < n {
                return Err(Error::Truncated);
            }
        }
        let v = self.acc >> (64 - n);
        self.acc <<= n;
        self.bits -= n;
        Ok(v)
    }

    /// Read `n <= 64` bits.
    #[inline]
    pub fn read(&mut self, n: u32) -> Result<u64, Error> {
        if n > 32 {
            let hi = self.get(n - 32)?;
            let lo = self.get(32)?;
            Ok((hi << 32) | lo)
        } else {
            self.get(n)
        }
    }

    /// Read an `n`-bit two's complement field, `1 <= n <= 64`.
    #[inline]
    pub fn read_signed(&mut self, n: u32) -> Result<i64, Error> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.read(n)?;
        let s = 64 - n;
        Ok(((v << s) as i64) >> s)
    }

    /// Count zero bits up to and including the terminating one bit.
    #[inline]
    pub fn read_unary(&mut self) -> Result<u64, Error> {
        let mut q = 0u64;
        loop {
            if self.bits == 0 {
                self.refill();
                if self.bits == 0 {
                    return Err(Error::Truncated);
                }
            }
            let lz = self.acc.leading_zeros();
            if lz < self.bits {
                q += lz as u64;
                self.acc <<= lz + 1;
                self.bits -= lz + 1;
                return Ok(q);
            }
            q += self.bits as u64;
            self.acc = 0;
            self.bits = 0;
        }
    }

    /// Rice code with parameter `k <= 30`.
    #[inline(always)]
    pub fn read_rice(&mut self, k: u32) -> Result<u64, Error> {
        if self.bits < 32 {
            self.refill();
        }
        // Fast path: the whole code is in the buffer. `lz < bits` also rules
        // out reading the not-yet-counted lookahead bits.
        let lz = self.acc.leading_zeros();
        let len = lz + 1 + k;
        if len <= self.bits {
            let a = (self.acc << lz) << 1;
            let low = if k == 0 { 0 } else { a >> (64 - k) };
            self.acc = a << k;
            self.bits -= len;
            return Ok(((lz as u64) << k) | low);
        }
        self.read_rice_slow(k)
    }

    /// Decode `out.len()` Rice codes with parameter `k <= 30`, unzigzagged.
    /// Same result as calling `read_rice` in a loop, with the bit buffer
    /// kept in locals.
    pub fn read_rice_run(&mut self, k: u32, out: &mut [i64]) -> Result<(), Error> {
        let data = self.data;
        let (mut acc, mut bits, mut pos) = (self.acc, self.bits, self.pos);
        let mut i = 0;
        while i < out.len() {
            // Branch-free refill while 8 bytes remain: tops the buffer up to
            // 56..63 bits every code (predictable, unlike `bits < 32`).
            if pos + 8 <= data.len() {
                let w = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
                acc |= w >> bits;
                pos += ((63 - bits) >> 3) as usize;
                bits |= 56;
            } else if bits < 32 {
                self.acc = acc;
                self.bits = bits;
                self.pos = pos;
                self.refill();
                acc = self.acc;
                bits = self.bits;
                pos = self.pos;
            }
            let lz = acc.leading_zeros();
            let len = lz + 1 + k;
            if len <= bits {
                let a = (acc << lz) << 1;
                // (a >> 1) >> (63 - k) is a >> (64 - k) for k > 0, and 0 for k = 0.
                let low = (a >> 1) >> (63 - k);
                acc = a << k;
                bits -= len;
                out[i] = crate::unzigzag(((lz as u64) << k) | low);
                i += 1;
            } else {
                // Long code or the end of the data: one code the slow way.
                self.acc = acc;
                self.bits = bits;
                self.pos = pos;
                out[i] = crate::unzigzag(self.read_rice_slow(k)?);
                i += 1;
                acc = self.acc;
                bits = self.bits;
                pos = self.pos;
            }
        }
        self.acc = acc;
        self.bits = bits;
        self.pos = pos;
        Ok(())
    }

    #[cold]
    fn read_rice_slow(&mut self, k: u32) -> Result<u64, Error> {
        let q = self.read_unary()?;
        if k == 0 {
            return Ok(q);
        }
        if q >= (1u64 << (64 - k)) {
            return Err(Error::Corrupt("rice quotient overflows 64 bits"));
        }
        let low = self.get(k)?;
        Ok((q << k) | low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_fields() {
        let mut w = BitWriter::with_capacity(16);
        w.write(0b101, 3);
        w.write_signed(-5, 7);
        w.write(u64::MAX >> 1, 63);
        w.write_rice(1000, 3);
        w.write_rice(0, 0);
        w.write_rice(77, 30);
        w.write_signed(i32::MIN as i64, 32);
        let buf = w.finish();
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read(3).unwrap(), 0b101);
        assert_eq!(r.read_signed(7).unwrap(), -5);
        assert_eq!(r.read(63).unwrap(), u64::MAX >> 1);
        assert_eq!(r.read_rice(3).unwrap(), 1000);
        assert_eq!(r.read_rice(0).unwrap(), 0);
        assert_eq!(r.read_rice(30).unwrap(), 77);
        assert_eq!(r.read_signed(32).unwrap(), i32::MIN as i64);
    }
}
