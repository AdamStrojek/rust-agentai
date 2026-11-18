//! A [`HttpClient`] for tests of providers (feature `test-support`).

use super::{HttpClient, HttpError, HttpRequest, HttpResponse};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Recorded {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}

/// Records the requests and answers with prepared responses, in order. Clones share the
/// record, so one clone can go to a provider while the test keeps another to read the requests.
///
/// ```rust
/// # use agentai_core::mock::{MockHttp, json_response, body_json};
/// # use agentai_core::{HttpClient, HttpRequest};
/// # async fn run() {
/// let http = MockHttp::new(vec![json_response(200, serde_json::json!({"id": "j1"}))]);
/// let client: Box<dyn HttpClient> = Box::new(http.clone());
/// let request = HttpRequest::post("http://host/jobs")
///     .with_json(&serde_json::json!({"prompt": "a cat"}))
///     .unwrap();
/// client.send(request).await.unwrap();
/// assert_eq!(body_json(&http.request(0))["prompt"], "a cat");
/// # }
/// ```
#[derive(Clone, Default)]
pub struct MockHttp(Arc<Recorded>);

impl MockHttp {
    /// Creates a client that answers with `responses`, in order. A request after the last
    /// response panics.
    pub fn new(responses: Vec<HttpResponse>) -> Self {
        Self(Arc::new(Recorded {
            requests: Mutex::default(),
            responses: Mutex::new(responses.into()),
        }))
    }

    /// The request sent as number `index`, counting from 0. Panics when there is none.
    pub fn request(&self, index: usize) -> HttpRequest {
        self.0.requests.lock().unwrap()[index].clone()
    }

    /// How many requests were sent.
    pub fn request_count(&self) -> usize {
        self.0.requests.lock().unwrap().len()
    }
}

#[async_trait]
impl HttpClient for MockHttp {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.0.requests.lock().unwrap().push(request);
        Ok(self
            .0
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("no response left"))
    }
}

/// A response with a JSON body.
pub fn json_response(status: u16, body: Value) -> HttpResponse {
    HttpResponse::new(status, body.to_string())
}

/// The body of a request as JSON. Panics when there is no body or it is not JSON.
pub fn body_json(request: &HttpRequest) -> Value {
    serde_json::from_slice(request.body.as_ref().expect("the request has no body"))
        .expect("the body is not JSON")
}
