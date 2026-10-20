use super::*;
use crate::tool::{Tool, ToolResult};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Minimal OpenAI-compatible server that replays canned responses and records request bodies.
struct MockLlm {
    base_url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockLlm {
    async fn start(responses: Vec<(u16, String)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1/", listener.local_addr().unwrap());
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let queue = queue.clone();
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let body = loop {
                        let n = socket.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        let text = String::from_utf8_lossy(&buf).to_string();
                        if let Some((head, body)) = text.split_once("\r\n\r\n") {
                            let len = head
                                .lines()
                                .find_map(|l| {
                                    l.to_ascii_lowercase()
                                        .strip_prefix("content-length:")
                                        .map(|v| v.trim().parse::<usize>().unwrap())
                                })
                                .unwrap_or(0);
                            if body.len() >= len {
                                break body.to_string();
                            }
                        }
                    };
                    recorded.lock().unwrap().push(body);
                    let (status, payload) = queue
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("mock LLM has no more responses");
                    let response = format!(
                        "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                    let _ = socket.shutdown().await;
                });
            }
        });
        Self { base_url, requests }
    }

    fn agent(&self) -> Agent {
        Agent::new_with_url(&self.base_url, "test-key", "system").unwrap()
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn text_reply(text: &str) -> (u16, String) {
    let body = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    (200, body.to_string())
}

fn tool_call_reply() -> (u16, String) {
    let body = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "echo", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    (200, body.to_string())
}

/// Toolbox with one tool, `echo`, that always fails with the error produced by `make_error`.
struct FailingToolBox {
    make_error: fn() -> ToolError,
}

#[async_trait]
impl ToolBox for FailingToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(vec![Tool::new("echo")])
    }

    async fn call_tool(&self, _tool_name: String, _arguments: Value) -> ToolResult {
        Err((self.make_error)())
    }
}

#[tokio::test]
async fn recoverable_tool_error_lets_the_model_retry() {
    let llm = MockLlm::start(vec![tool_call_reply(), text_reply("recovered")]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::InvalidArguments("timezone is unknown".to_string()),
    };

    let answer: String = llm
        .agent()
        .run("m", "question", Some(&toolbox))
        .await
        .unwrap();

    assert_eq!(answer, "recovered");
    let requests = llm.requests();
    assert_eq!(requests.len(), 2);
    // The second request carries the tool error as the tool result
    assert!(requests[1].contains("invalid arguments: timezone is unknown"));
}

#[tokio::test]
async fn fatal_tool_error_ends_the_run() {
    let llm = MockLlm::start(vec![tool_call_reply()]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::Configuration("missing API key".to_string()),
    };

    let err = llm
        .agent()
        .run::<String>("m", "question", Some(&toolbox))
        .await
        .unwrap_err();

    match err {
        AgentError::Tool { tool, source } => {
            assert_eq!(tool, "echo");
            assert!(matches!(source, ToolError::Configuration(_)));
            assert!(!source.is_recoverable());
        }
        other => panic!("expected AgentError::Tool, got {other:?}"),
    }
    // The run stopped: the model was not asked again
    assert_eq!(llm.requests().len(), 1);
}

#[tokio::test]
async fn invalid_answer_keeps_the_raw_text() {
    #[derive(Deserialize, JsonSchema, Debug)]
    struct Answer {
        #[allow(dead_code)]
        number: u32,
    }
    let llm = MockLlm::start(vec![text_reply("not json at all")]).await;

    let err = llm
        .agent()
        .run::<Answer>("m", "question", None)
        .await
        .unwrap_err();

    match err {
        AgentError::InvalidAnswer { raw, .. } => assert_eq!(raw, "not json at all"),
        other => panic!("expected AgentError::InvalidAnswer, got {other:?}"),
    }
}

#[tokio::test]
async fn running_out_of_iterations_is_reported() {
    let llm = MockLlm::start(vec![tool_call_reply(); 5]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::Failed("still broken".to_string()),
    };

    let err = llm
        .agent()
        .run::<String>("m", "question", Some(&toolbox))
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::MaxIterationsReached(5)));
}

#[tokio::test]
async fn provider_errors_know_if_they_are_transient() {
    let llm = MockLlm::start(vec![
        (502, r#"{"error":"bad gateway"}"#.to_string()),
        (401, r#"{"error":"unauthorized"}"#.to_string()),
    ])
    .await;

    let err = llm.agent().run::<String>("m", "q", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Llm(_)));
    assert!(err.is_transient());

    let err = llm.agent().run::<String>("m", "q", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Llm(_)));
    assert!(!err.is_transient());
}

#[test]
fn recoverable_and_fatal_tool_errors() {
    assert!(ToolError::NotFound("x".into()).is_recoverable());
    assert!(ToolError::InvalidArguments("x".into()).is_recoverable());
    assert!(ToolError::Failed("x".into()).is_recoverable());
    assert!(!ToolError::Configuration("x".into()).is_recoverable());
    assert!(!ToolError::fatal("x").is_recoverable());
}

fn request_json(llm: &MockLlm, index: usize) -> Value {
    serde_json::from_str(&llm.requests()[index]).unwrap()
}

#[tokio::test]
async fn max_iterations_can_be_configured() {
    let llm = MockLlm::start(vec![tool_call_reply(); 2]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::Failed("still broken".to_string()),
    };

    let err = llm
        .agent()
        .with_max_iterations(2)
        .run::<String>("m", "question", Some(&toolbox))
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::MaxIterationsReached(2)));
    assert_eq!(llm.requests().len(), 2);
}

#[tokio::test]
async fn zero_iterations_do_not_call_the_model() {
    let llm = MockLlm::start(vec![]).await;

    let err = llm
        .agent()
        .with_max_iterations(0)
        .run::<String>("m", "question", None)
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::MaxIterationsReached(0)));
    assert!(llm.requests().is_empty());
}

#[tokio::test]
async fn default_options_send_the_default_temperature() {
    let llm = MockLlm::start(vec![text_reply("hi")]).await;

    let agent = llm.agent();
    assert_eq!(agent.max_iterations(), DEFAULT_MAX_ITERATIONS);
    let _: String = agent.clone().run("m", "question", None).await.unwrap();

    assert_eq!(
        request_json(&llm, 0)["temperature"],
        json!(DEFAULT_TEMPERATURE)
    );
}

#[tokio::test]
async fn chat_options_are_sent_with_every_request() {
    use genai::chat::ToolChoice;

    let llm = MockLlm::start(vec![text_reply("hi")]).await;
    let options = ChatOptions::default()
        .with_temperature(0.7)
        .with_max_tokens(123)
        .with_tool_choice(ToolChoice::Required);

    let _: String = llm
        .agent()
        .with_chat_options(options)
        .run("m", "question", None)
        .await
        .unwrap();

    let request = request_json(&llm, 0);
    assert_eq!(request["temperature"], json!(0.7));
    let max_tokens = request
        .get("max_tokens")
        .or_else(|| request.get("max_completion_tokens"));
    assert_eq!(max_tokens, Some(&json!(123)));
    assert_eq!(request["tool_choice"], json!("required"));
}

#[tokio::test]
async fn structured_output_keeps_custom_options() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        value: u32,
    }

    let llm = MockLlm::start(vec![text_reply(r#"{"value": 7}"#)]).await;

    let answer: Answer = llm
        .agent()
        .with_chat_options(ChatOptions::default().with_temperature(0.0))
        .run("m", "question", None)
        .await
        .unwrap();

    assert_eq!(answer.value, 7);
    let request = request_json(&llm, 0);
    assert_eq!(request["temperature"], json!(0.0));
    assert!(request.get("response_format").is_some());
}

struct StrictTools;

#[crate::tool::toolbox]
impl StrictTools {
    /// Strict echo.
    #[tool(strict)]
    fn strict_echo(
        &self,
        /// Text to echo
        text: String,
    ) -> ToolResult {
        Ok(text)
    }

    /// Plain echo.
    #[tool]
    fn plain_echo(
        &self,
        /// Text to echo
        text: String,
    ) -> ToolResult {
        Ok(text)
    }
}

#[tokio::test]
async fn strict_tools_are_sent_as_strict() {
    let llm = MockLlm::start(vec![text_reply("hi")]).await;

    let _: String = llm
        .agent()
        .run("m", "question", Some(&StrictTools))
        .await
        .unwrap();

    let request = request_json(&llm, 0);
    let tools = request["tools"].as_array().unwrap();
    let function = |name: &str| {
        tools
            .iter()
            .map(|t| &t["function"])
            .find(|f| f["name"] == json!(name))
            .unwrap()
            .clone()
    };

    let strict = function("strict_echo");
    assert_eq!(strict["strict"], json!(true));
    assert_eq!(strict["parameters"]["additionalProperties"], json!(false));
    assert_eq!(strict["parameters"]["required"], json!(["text"]));
    assert_eq!(function("plain_echo")["strict"], json!(false));
}
