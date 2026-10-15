//! Wire codec: serde JSON + length-prefixed framing.
//!
//! Layout per frame: `u32 BE length` + `length` bytes of JSON — always a serialized
//! command or `UiEvent` value.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("frame exceeds 16 MiB cap: {0} bytes")]
    Oversized(u32),
}

const MAX_FRAME: u32 = 16 * 1024 * 1024;

/// Encode a serializable event into `len + payload` bytes.
pub fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let payload = serde_json::to_vec(value)?;
    let len = payload.len() as u32;
    if len > MAX_FRAME {
        return Err(CodecError::Oversized(len));
    }
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Streaming decoder — feed bytes, drain complete frames.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pop the next complete frame, if any.
    pub fn next_frame<T: serde::de::DeserializeOwned>(&mut self) -> Result<Option<T>, CodecError> {
        if self.buf.len() < 4 {
            return Ok(None);
        }
        let len = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]);
        if len > MAX_FRAME {
            return Err(CodecError::Oversized(len));
        }
        if self.buf.len() < 4 + len as usize {
            return Ok(None);
        }
        let payload = self
            .buf
            .drain(..4 + len as usize)
            .skip(4)
            .collect::<Vec<u8>>();
        Ok(Some(serde_json::from_slice(&payload)?))
    }
}
