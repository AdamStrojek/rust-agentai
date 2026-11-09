//! HTTP helpers shared by the providers.

use super::AudioError;
use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpRequest, HttpResponse};
use base64::Engine;

/// Adds the bearer token, unless the key is empty.
pub(super) fn with_bearer(request: HttpRequest, api_key: &str) -> HttpRequest {
    if api_key.is_empty() {
        request
    } else {
        request.with_header("Authorization", format!("Bearer {api_key}"))
    }
}

/// Sends the request; a non-2xx status becomes [`AudioError::Api`].
pub(super) async fn send(
    client: &dyn HttpClient,
    request: HttpRequest,
) -> Result<HttpResponse, AudioError> {
    let response = client.send(request).await?;
    if !response.is_success() {
        return Err(api_error(&response));
    }
    Ok(response)
}

/// Wraps the body of a response as audio of the given type.
pub(super) fn audio_media(
    response: &HttpResponse,
    content_type: &str,
) -> Result<Media, AudioError> {
    if response.body.is_empty() {
        return Err(AudioError::InvalidResponse(
            "the response has no audio".to_string(),
        ));
    }
    Ok(Media::from_bytes(content_type, &response.body))
}

/// Bytes of a media: decoded base64, or downloaded from its URL.
pub(super) async fn media_bytes(
    client: &dyn HttpClient,
    media: &Media,
) -> Result<Vec<u8>, AudioError> {
    match &media.source {
        MediaSource::Base64(data) => base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| AudioError::InvalidResponse(format!("invalid base64 audio: {e}"))),
        MediaSource::Url(url) => Ok(send(client, HttpRequest::get(url)).await?.body),
    }
}

/// Escapes everything but the unreserved characters of RFC 3986.
pub(super) fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Reads the message of the error shapes of OpenAI, oMLX/FastAPI and ElevenLabs.
fn api_error(response: &HttpResponse) -> AudioError {
    let message = response
        .json::<serde_json::Value>()
        .ok()
        .and_then(|json| {
            let detail = &json["detail"];
            json["error"]["message"]
                .as_str()
                .or_else(|| detail.as_str())
                .or_else(|| detail["message"].as_str())
                .or_else(|| detail[0]["msg"].as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| response.text());
    AudioError::Api {
        status: response.status,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::json_response;
    use serde_json::{Value, json};

    #[test]
    fn api_error_reads_every_error_shape() {
        let message = |body: Value| match api_error(&json_response(400, body)) {
            AudioError::Api { message, .. } => message,
            other => panic!("{other:?}"),
        };
        assert_eq!(message(json!({"error": {"message": "openai"}})), "openai");
        assert_eq!(message(json!({"detail": "omlx"})), "omlx");
        assert_eq!(message(json!({"detail": {"message": "eleven"}})), "eleven");
        assert_eq!(
            message(json!({"detail": [{"msg": "validation"}]})),
            "validation"
        );
        assert_eq!(message(json!({"other": 1})), r#"{"other":1}"#);
    }

    #[test]
    fn empty_audio_body_is_invalid() {
        let empty = HttpResponse::new(200, Vec::new());
        assert!(matches!(
            audio_media(&empty, "audio/mpeg"),
            Err(AudioError::InvalidResponse(_))
        ));
        let full = HttpResponse::new(200, b"hi".to_vec());
        assert_eq!(
            audio_media(&full, "audio/mpeg").unwrap(),
            Media::from_bytes("audio/mpeg", b"hi")
        );
    }

    #[test]
    fn percent_encode_escapes_everything_but_unreserved() {
        assert_eq!(percent_encode("a-b_c.d~1"), "a-b_c.d~1");
        assert_eq!(percent_encode("a+b/c=d e"), "a%2Bb%2Fc%3Dd%20e");
    }
}
