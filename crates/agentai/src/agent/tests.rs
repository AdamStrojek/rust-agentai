use super::*;
use crate::message::Message;
use crate::tool::{Tool, ToolResult};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
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
                    let content_type = if payload.starts_with("data:") {
                        "text/event-stream"
                    } else {
                        "application/json"
                    };
                    let response = format!(
                        "HTTP/1.1 {status} Mock\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
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
        Agent::new_with_url(&self.base_url, "test-key", "system")
            .unwrap()
            .with_model("m")
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

    let result: RunResult<String> = llm
        .agent()
        .with_toolbox(toolbox)
        .run("question")
        .await
        .unwrap();

    assert_eq!(result.answer, "recovered");
    assert_eq!(result.iterations, 2);
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
        .with_toolbox(toolbox)
        .run::<String>("question")
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

    let err = llm.agent().run::<Answer>("question").await.unwrap_err();

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
        .with_toolbox(toolbox)
        .run::<String>("question")
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

    let err = llm.agent().run::<String>("q").await.unwrap_err();
    assert!(matches!(err, AgentError::Llm(_)));
    assert!(err.is_transient());

    let err = llm.agent().run::<String>("q").await.unwrap_err();
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
        .with_toolbox(toolbox)
        .run::<String>("question")
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
        .run::<String>("question")
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
    let _: RunResult<String> = agent.run("question").await.unwrap();

    assert_eq!(
        request_json(&llm, 0)["temperature"],
        json!(DEFAULT_TEMPERATURE)
    );
}

#[tokio::test]
async fn chat_options_are_sent_with_every_request() {
    let llm = MockLlm::start(vec![text_reply("hi")]).await;
    let options = ChatOptions::default()
        .with_temperature(0.7)
        .with_max_tokens(123)
        .with_tool_choice(ToolChoice::Required);

    let _: RunResult<String> = llm
        .agent()
        .with_chat_options(options)
        .run("question")
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

    let result: RunResult<Answer> = llm
        .agent()
        .with_chat_options(ChatOptions::default().with_temperature(0.0))
        .run("question")
        .await
        .unwrap();

    assert_eq!(result.answer.value, 7);
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

    let _: RunResult<String> = llm
        .agent()
        .with_toolbox(StrictTools)
        .run("question")
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

/// Toolbox with one tool, `echo`, that returns "echoed".
struct EchoToolBox;

#[async_trait]
impl ToolBox for EchoToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(vec![Tool::new("echo")])
    }

    async fn call_tool(&self, _tool_name: String, _arguments: Value) -> ToolResult {
        Ok("echoed".to_string())
    }
}

fn sse(chunks: Vec<Value>) -> (u16, String) {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    (200, body)
}

fn sse_chunk(delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
    })
}

fn sse_usage() -> Value {
    json!({
        "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

fn streamed_text_reply(parts: &[&str]) -> (u16, String) {
    let mut chunks: Vec<Value> = parts
        .iter()
        .map(|part| sse_chunk(json!({ "role": "assistant", "content": part }), None))
        .collect();
    chunks.push(sse_chunk(json!({}), Some("stop")));
    chunks.push(sse_usage());
    sse(chunks)
}

fn streamed_tool_call_reply() -> (u16, String) {
    sse(vec![
        sse_chunk(
            json!({ "role": "assistant", "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": { "name": "echo", "arguments": "" }
            }]}),
            None,
        ),
        sse_chunk(
            json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{}" } }]}),
            None,
        ),
        sse_chunk(json!({}), Some("tool_calls")),
        sse_usage(),
    ])
}

fn truncated_reply(text: &str) -> (u16, String) {
    let body = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "length"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    (200, body.to_string())
}

/// Messages as JSON, to compare them.
fn messages_json(messages: &[Message]) -> Value {
    serde_json::to_value(messages).unwrap()
}

#[tokio::test]
async fn run_returns_only_the_new_messages() {
    let llm = MockLlm::start(vec![tool_call_reply(), text_reply("done")]).await;
    let history = vec![
        Message::user("earlier question"),
        Message::assistant("earlier answer"),
    ];

    let result: RunResult<String> = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run("question")
        .with_history(&history)
        .await
        .unwrap();

    assert_eq!(result.answer, "done");
    // prompt, assistant tool call, tool response, final answer
    assert_eq!(result.new_messages.len(), 4);
    let new = messages_json(&result.new_messages);
    assert!(!new.to_string().contains("earlier"));
    assert!(!new.to_string().contains("\"system\""));
    assert_eq!(result.usage.total_tokens, 4);
    assert_eq!(result.iterations, 2);
    assert!(matches!(result.stop_reason, Some(StopReason::Completed)));

    // The request carries the system prompt and the history before the prompt
    let messages = request_json(&llm, 0)["messages"].clone();
    assert_eq!(messages[0]["role"], json!("system"));
    assert_eq!(messages[0]["content"], json!("system"));
    assert_eq!(messages[1]["content"], json!("earlier question"));
    assert_eq!(messages[2]["content"], json!("earlier answer"));
    assert_eq!(messages[3]["content"], json!("question"));
}

#[tokio::test]
async fn appended_messages_continue_the_conversation() {
    let llm = MockLlm::start(vec![text_reply("first"), text_reply("second")]).await;
    let agent = llm.agent();
    let mut history = Vec::new();

    let result: RunResult<String> = agent.run("one").with_history(&history).await.unwrap();
    history.extend(result.new_messages);
    let result: RunResult<String> = agent.run("two").with_history(&history).await.unwrap();
    history.extend(result.new_messages);

    assert_eq!(history.len(), 4);
    let contents: Vec<Value> = request_json(&llm, 1)["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["content"].clone())
        .collect();
    assert_eq!(
        contents,
        vec![json!("system"), json!("one"), json!("first"), json!("two")]
    );
}

#[tokio::test]
async fn run_options_replace_the_agent_settings() {
    let llm = MockLlm::start(vec![tool_call_reply(), text_reply("unused")]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::Failed("still broken".to_string()),
    };

    let err = llm
        .agent()
        .run::<String>("question")
        .with_options(
            RunOptions::default()
                .with_model("other-model")
                .with_max_iterations(1)
                .with_toolbox(toolbox),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::MaxIterationsReached(1)));
    let request = request_json(&llm, 0);
    assert_eq!(request["model"], json!("other-model"));
    assert_eq!(request["tools"][0]["function"]["name"], json!("echo"));
}

#[tokio::test]
async fn a_run_without_a_model_fails_before_calling_the_model() {
    let llm = MockLlm::start(vec![]).await;
    let agent = Agent::new_with_url(&llm.base_url, "test-key", "system").unwrap();

    let err = agent.run::<String>("question").await.unwrap_err();
    assert!(matches!(err, AgentError::MissingModel));
    let err = agent.run_stream::<String>("question").await.err().unwrap();
    assert!(matches!(err, AgentError::MissingModel));
    assert!(llm.requests().is_empty());

    // A model set for the run is enough
    let llm = MockLlm::start(vec![text_reply("hi")]).await;
    let agent = Agent::new_with_url(&llm.base_url, "test-key", "system").unwrap();
    let result: RunResult<String> = agent.run("question").with_model("m").await.unwrap();
    assert_eq!(result.answer, "hi");
}

#[tokio::test]
async fn truncated_answer_is_reported() {
    let llm = MockLlm::start(vec![truncated_reply(r#"{"value":"#)]).await;

    let err = llm.agent().run::<String>("question").await.unwrap_err();

    match err {
        AgentError::MaxTokens { raw } => assert_eq!(raw, r#"{"value":"#),
        other => panic!("expected AgentError::MaxTokens, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_sends_deltas_tool_events_and_the_result() {
    use futures_util::StreamExt;

    let llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;

    let mut stream = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap();
    let mut events = Vec::new();
    let mut result = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::TextDelta(text) => events.push(format!("text:{text}")),
            AgentEvent::ToolCall(call) => events.push(format!("call:{}", call.name)),
            AgentEvent::ToolResult {
                tool,
                content,
                is_error,
                ..
            } => events.push(format!("result:{tool}:{content}:{is_error}")),
            AgentEvent::Finished(res) => result = Some(res),
            other => panic!("unexpected event {other:?}"),
        }
    }

    assert_eq!(
        events,
        vec![
            "call:echo",
            "result:echo:echoed:false",
            "text:Hel",
            "text:lo"
        ]
    );
    let result = result.expect("the stream ends with Finished");
    assert_eq!(result.answer, "Hello");
    assert_eq!(result.new_messages.len(), 4);
    assert_eq!(result.usage.total_tokens, 4);
    assert_eq!(request_json(&llm, 0)["stream"], json!(true));
}

#[tokio::test]
async fn streamed_and_awaited_runs_produce_the_same_messages() {
    let awaited_llm = MockLlm::start(vec![tool_call_reply(), text_reply("Hello")]).await;
    let awaited: RunResult<String> = awaited_llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run("question")
        .await
        .unwrap();

    let streamed_llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;
    let streamed = streamed_llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .finish()
        .await
        .unwrap();

    assert_eq!(
        messages_json(&streamed.new_messages),
        messages_json(&awaited.new_messages)
    );
    // Both send the same conversation to the model in the second call
    assert_eq!(
        request_json(&streamed_llm, 1)["messages"],
        request_json(&awaited_llm, 1)["messages"]
    );
}

#[tokio::test]
async fn streamed_structured_output_is_parsed_at_the_end() {
    #[derive(Deserialize, JsonSchema, Debug)]
    struct Answer {
        value: u32,
    }
    let llm = MockLlm::start(vec![streamed_text_reply(&[r#"{"val"#, r#"ue": 7}"#])]).await;

    let result = llm
        .agent()
        .run_stream::<Answer>("question")
        .await
        .unwrap()
        .finish()
        .await
        .unwrap();

    assert_eq!(result.answer.value, 7);
}

#[tokio::test]
async fn fatal_tool_error_ends_the_stream() {
    use futures_util::StreamExt;

    let llm = MockLlm::start(vec![streamed_tool_call_reply()]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::Configuration("missing API key".to_string()),
    };

    let mut stream = llm
        .agent()
        .with_toolbox(toolbox)
        .run_stream::<String>("q")
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Ok(AgentEvent::ToolCall(_)))
    ));
    assert!(matches!(
        stream.next().await,
        Some(Ok(AgentEvent::ToolResult { is_error: true, .. }))
    ));
    assert!(matches!(
        stream.next().await,
        Some(Err(AgentError::Tool { .. }))
    ));
    assert!(stream.next().await.is_none());
}

#[test]
fn runs_and_streams_can_be_moved_to_other_tasks() {
    fn assert_send_static<T: Send + 'static>(_: &T) {}
    fn assert_send_static_type<T: Send + 'static>() {}
    let agent = Agent::new_with_url("http://localhost/", "key", "system").unwrap();
    let history = vec![Message::user("hi")];
    let future =
        std::future::IntoFuture::into_future(agent.run::<String>("q").with_history(&history));
    assert_send_static(&future);
    let future = std::future::IntoFuture::into_future(
        agent.run_stream::<String>("q").with_history(&history),
    );
    assert_send_static(&future);
    assert_send_static_type::<AgentStream<String>>();
    fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
    assert_send_sync_clone::<Agent>();
}
