//! [`HttpClient`] implemented with `reqwest`.

use super::{HttpClient, HttpError, HttpRequest, HttpResponse, Method};
use async_trait::async_trait;

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
            .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
            .collect();
        let body = response.bytes().await.map_err(HttpError::transport)?;
        Ok(HttpResponse {
            status,
            headers,
            body: body.to_vec(),
        })
    }
}
