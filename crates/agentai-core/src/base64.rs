//! Base64 (standard alphabet, with padding), the encoding of [`MediaSource::Base64`]
//! and of data URLs.
//!
//! [`MediaSource::Base64`]: crate::message::MediaSource::Base64

use crate::http::HttpError;
use ::base64::Engine;
use ::base64::engine::general_purpose::STANDARD;

/// Encodes bytes.
pub fn encode(bytes: impl AsRef<[u8]>) -> String {
    STANDARD.encode(bytes)
}

/// Decodes base64 text. Invalid input is [`HttpError::InvalidMedia`].
pub fn decode(data: &str) -> Result<Vec<u8>, HttpError> {
    STANDARD
        .decode(data)
        .map_err(|e| HttpError::InvalidMedia(e.to_string()))
}
