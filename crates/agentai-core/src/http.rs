//! HTTP client abstraction.

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

/// HTTP method of a [`HttpRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    Get,
    Post,
}

/// A request sent with a [`HttpClient`]. The body is always plain bytes, so a client only
/// has to send them.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

impl HttpRequest {
    /// Creates a request without headers and body.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Creates a `GET` request.
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(Method::Get, url)
    }

    /// Creates a `POST` request.
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Method::Post, url)
    }

    /// Adds a header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the body.
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Sets a JSON body and its `Content-Type`.
    pub fn with_json(self, body: &impl Serialize) -> Result<Self, HttpError> {
        let body = serde_json::to_vec(body).map_err(HttpError::Json)?;
        Ok(self
            .with_header("Content-Type", "application/json")
            .with_body(body))
    }

    /// Sets a `multipart/form-data` body and its `Content-Type`.
    pub fn with_multipart(self, form: Multipart) -> Self {
        let (content_type, body) = form.into_body();
        self.with_header("Content-Type", content_type)
            .with_body(body)
    }

    /// Returns the value of the first header with this name, ignoring case.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }
}

/// A response returned by a [`HttpClient`], with the whole body read.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Creates a response without headers.
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    /// Adds a header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Returns `true` for a 2xx status.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Returns the value of the first header with this name, ignoring case.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }

    /// Parses the body as JSON.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T, HttpError> {
        serde_json::from_slice(&self.body).map_err(HttpError::Json)
    }

    /// Returns the body as text, replacing invalid UTF-8.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Errors of a [`HttpClient`] and of building or reading a request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HttpError {
    /// The request could not be sent or the response could not be read (connection,
    /// timeout, TLS, ...).
    #[error("HTTP request failed: {0}")]
    Transport(Box<dyn std::error::Error + Send + Sync>),
    /// A JSON body could not be written or read.
    #[error("invalid JSON body: {0}")]
    Json(serde_json::Error),
}

impl HttpError {
    /// Creates a [`HttpError::Transport`] from any error.
    pub fn transport(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Transport(err.into())
    }
}

/// Sends HTTP requests.
///
/// Crates such as `agentai-image` send their requests through this trait instead of a
/// particular HTTP client. [`ReqwestClient`] (feature `reqwest`) implements it.
///
/// The returned future is `Send`. Clients whose futures are not `Send` (for example in the
/// browser) will need a relaxed bound, which comes with WebAssembly support.
#[async_trait]
pub trait HttpClient: Send + Sync + 'static {
    /// Sends the request and reads the whole response. A response with an error status is
    /// still `Ok`; only failures to send or read are errors.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;
}

/// A `multipart/form-data` body, built in memory.
///
/// ```rust
/// # use agentai_core::{HttpRequest, Multipart};
/// let request = HttpRequest::post("https://example.com/upload").with_multipart(
///     Multipart::new()
///         .with_text("prompt", "a cat")
///         .with_file("image", "cat.png", "image/png", vec![0x89, b'P', b'N', b'G']),
/// );
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Multipart {
    parts: Vec<FormPart>,
}

#[derive(Debug, Clone, PartialEq)]
struct FormPart {
    name: String,
    file: Option<(String, String)>,
    data: Vec<u8>,
}

impl Multipart {
    /// Creates an empty form.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a text field.
    pub fn with_text(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parts.push(FormPart {
            name: name.into(),
            file: None,
            data: value.into().into_bytes(),
        });
        self
    }

    /// Adds a file field.
    pub fn with_file(
        mut self,
        name: impl Into<String>,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        self.parts.push(FormPart {
            name: name.into(),
            file: Some((filename.into(), content_type.into())),
            data: data.into(),
        });
        self
    }

    /// Returns the `Content-Type` header value and the body.
    fn into_body(self) -> (String, Vec<u8>) {
        let boundary = self.boundary();
        let mut body = Vec::new();
        for part in self.parts {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            let name = escape(&part.name);
            match part.file {
                Some((filename, content_type)) => body.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{}\"\r\n\
                         Content-Type: {content_type}\r\n\r\n",
                        escape(&filename)
                    )
                    .as_bytes(),
                ),
                None => body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            body.extend_from_slice(&part.data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    /// A boundary that appears in no part. Chosen without randomness or a clock, so it works
    /// in every environment.
    fn boundary(&self) -> String {
        (0u64..)
            .map(|n| format!("agentai-boundary-{n:016x}"))
            .find(|boundary| {
                !self
                    .parts
                    .iter()
                    .any(|part| contains(&part.data, boundary.as_bytes()))
            })
            .expect("some boundary is not in the data")
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Quotes and line breaks would end the header value early.
fn escape(value: &str) -> String {
    value
        .replace('"', "%22")
        .replace('\r', "%0D")
        .replace('\n', "%0A")
}

#[cfg(feature = "reqwest")]
mod reqwest_client {
    use super::*;

    /// [`HttpClient`] on top of `reqwest`.
    #[derive(Debug, Clone, Default)]
    pub struct ReqwestClient {
        client: reqwest::Client,
    }

    impl ReqwestClient {
        /// Creates a client with the default `reqwest` settings.
        pub fn new() -> Self {
            Self::default()
        }

        /// Uses a configured `reqwest` client (timeouts, proxy, ...).
        pub fn from_client(client: reqwest::Client) -> Self {
            Self { client }
        }
    }

    #[async_trait]
    impl HttpClient for ReqwestClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let method = match request.method {
                Method::Get => reqwest::Method::GET,
                Method::Post => reqwest::Method::POST,
            };
            let mut builder = self.client.request(method, &request.url);
            for (name, value) in request.headers {
                builder = builder.header(name, value);
            }
            if let Some(body) = request.body {
                builder = builder.body(body);
            }
            let response = builder.send().await.map_err(HttpError::transport)?;
            let status = response.status().as_u16();
            let headers = response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    Some((name.to_string(), value.to_str().ok()?.to_string()))
                })
                .collect();
            let body = response.bytes().await.map_err(HttpError::transport)?;
            Ok(HttpResponse {
                status,
                headers,
                body: body.to_vec(),
            })
        }
    }
}

#[cfg(feature = "reqwest")]
pub use reqwest_client::ReqwestClient;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_body_has_the_fields_and_files_between_boundaries() {
        let request = HttpRequest::post("u").with_multipart(
            Multipart::new().with_text("prompt", "a cat").with_file(
                "image[]",
                "a.png",
                "image/png",
                b"PNG".to_vec(),
            ),
        );
        let boundary = "agentai-boundary-0000000000000000";
        assert_eq!(
            request.header("content-type"),
            Some(format!("multipart/form-data; boundary={boundary}").as_str())
        );
        let expected = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"prompt\"\r\n\r\n\
             a cat\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\n\
             Content-Type: image/png\r\n\r\n\
             PNG\r\n\
             --{boundary}--\r\n"
        );
        assert_eq!(String::from_utf8(request.body.unwrap()).unwrap(), expected);
    }

    #[test]
    fn multipart_boundary_never_appears_in_the_data() {
        let data = "agentai-boundary-0000000000000000";
        let request =
            HttpRequest::post("u").with_multipart(Multipart::new().with_text("prompt", data));
        let content_type = request.header("Content-Type").unwrap();
        assert!(content_type.ends_with("agentai-boundary-0000000000000001"));
    }
}
