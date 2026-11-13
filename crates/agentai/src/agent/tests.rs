use super::*;
use crate::listener::{EventListener, ListenerError, Printer, RunEvent};
use crate::message::{Message, ToolCall};
use crate::tool::{Tool, ToolOutput, ToolResult, ToolReturn};
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
        Agent::from_client(openai_client(&self.base_url), "m", "system")
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

/// Client that sends every model name to `base_url` with the OpenAI Chat Completions API.
fn openai_client(base_url: &str) -> Client {
    use genai::adapter::AdapterKind;
    use genai::resolver::{AuthData, Endpoint};
    crate::tls::install_crypto_provider();
    Client::builder()
        .with_adapter_kind(AdapterKind::OpenAI)
        .append_provider_config(
            AdapterKind::OpenAI,
            (
                Endpoint::from_owned(base_url.to_string()),
                AuthData::from_single("test-key"),
            ),
        )
        .build()
        .unwrap()
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

    async fn call_tool(&self, _call: &ToolCall) -> ToolResult<ToolReturn> {
        Err((self.make_error)())
    }
}

#[tokio::test]
async fn recoverable_tool_error_lets_the_model_retry() {
    let llm = MockLlm::start(vec![tool_call_reply(), text_reply("recovered")]).await;
    let toolbox = FailingToolBox {
        make_error: || ToolError::InvalidArguments("timezone is unknown".to_string()),
    };

    let result: AgentResult<String> = llm
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
async fn chat_options_are_sent_with_every_request() {
    let llm = MockLlm::start(vec![text_reply("hi")]).await;
    let options = ChatOptions::default()
        .with_temperature(0.7)
        .with_max_tokens(123)
        .with_tool_choice(ToolChoice::Required);

    let _: AgentResult<String> = llm
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

    let result: AgentResult<Answer> = llm
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

    let _: AgentResult<String> = llm
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

    async fn call_tool(&self, _call: &ToolCall) -> ToolResult<ToolReturn> {
        Ok("echoed".into())
    }
}

/// Toolbox with one tool, `echo`, that returns text and an image.
struct ImageToolBox;

#[async_trait]
impl ToolBox for ImageToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(vec![Tool::new("echo")])
    }

    async fn call_tool(&self, _call: &ToolCall) -> ToolResult<ToolReturn> {
        Ok(ToolOutput::from_text("image-1.png")
            .with_media(crate::Media::from_base64("image/png", "aGk="))
            .into())
    }
}

#[tokio::test]
async fn tool_images_reach_openai_compatible_models_as_user_attachments() {
    let llm = MockLlm::start(vec![tool_call_reply(), text_reply("done")]).await;

    let result: AgentResult<String> = llm
        .agent()
        .with_toolbox(ImageToolBox)
        .run("question")
        .await
        .unwrap();

    // The OpenAI chat API takes no images in tool results: the text goes in the tool message
    // and the image in a user message right after it
    let messages = request_json(&llm, 1)["messages"]
        .as_array()
        .unwrap()
        .clone();
    let tool_index = messages
        .iter()
        .position(|m| m["role"] == json!("tool"))
        .unwrap();
    assert_eq!(messages[tool_index]["content"], json!("image-1.png"));
    let attachment = &messages[tool_index + 1];
    assert_eq!(attachment["role"], json!("user"));
    let image_urls: Vec<&Value> = attachment["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|part| part["type"] == json!("image_url"))
        .map(|part| &part["image_url"]["url"])
        .collect();
    assert_eq!(image_urls, [&json!("data:image/png;base64,aGk=")]);
    // The run's messages keep the image, for providers that accept it and for the caller
    let response = result
        .new_messages
        .iter()
        .flat_map(|m| &m.parts)
        .find_map(|part| match part {
            crate::Part::ToolResponse(response) => Some(response),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        response.content,
        ToolOutput::from_text("image-1.png")
            .with_media(crate::Media::from_base64("image/png", "aGk="))
            .parts
    );
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

    let result: AgentResult<String> = llm
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

    let result: AgentResult<String> = agent.run("one").with_history(&history).await.unwrap();
    history.extend(result.new_messages);
    let result: AgentResult<String> = agent.run("two").with_history(&history).await.unwrap();
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
    let agent = Agent::default();

    let err = agent.run::<String>("question").await.unwrap_err();
    assert!(matches!(err, AgentError::MissingModel));
    let err = agent.run_stream::<String>("question").await.err().unwrap();
    assert!(matches!(err, AgentError::MissingModel));
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
    let awaited: AgentResult<String> = awaited_llm
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
    let agent = Agent::from_client(openai_client("http://localhost/"), "m", "system");
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

/// Toolbox that keeps the calls it gets.
struct RecordingToolBox(Arc<Mutex<Vec<ToolCall>>>);

#[async_trait]
impl ToolBox for RecordingToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(vec![Tool::new("echo")])
    }

    async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolReturn> {
        self.0.lock().unwrap().push(call.clone());
        Ok("echoed".into())
    }
}

#[tokio::test]
async fn the_toolbox_gets_the_call_with_the_id_from_the_model() {
    let reply = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "call_a", "type": "function",
                     "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"}},
                    {"id": "call_b", "type": "function",
                     "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"}},
                ]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    let llm = MockLlm::start(vec![(200, reply.to_string()), text_reply("done")]).await;
    let calls = Arc::new(Mutex::new(Vec::new()));

    llm.agent()
        .with_toolbox(RecordingToolBox(calls.clone()))
        .run::<String>("question")
        .await
        .unwrap();

    // Two calls with the same arguments differ only by their ids
    let calls = calls.lock().unwrap();
    let calls: Vec<(&str, &str, &Value)> = calls
        .iter()
        .map(|c| (c.id.as_str(), c.name.as_str(), &c.arguments))
        .collect();
    let args = json!({"text": "hi"});
    assert_eq!(
        calls,
        [("call_a", "echo", &args), ("call_b", "echo", &args)]
    );
}

fn truncated_tool_call_reply() -> (u16, String) {
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
            "finish_reason": "length"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    (200, body.to_string())
}

#[tokio::test]
async fn truncated_tool_calls_are_not_run() {
    let llm = MockLlm::start(vec![truncated_tool_call_reply()]).await;
    let calls = Arc::new(Mutex::new(Vec::new()));

    let err = llm
        .agent()
        .with_toolbox(RecordingToolBox(calls.clone()))
        .run::<String>("question")
        .await
        .unwrap_err();

    match err {
        AgentError::MaxTokens { raw } => assert_eq!(raw, ""),
        other => panic!("expected AgentError::MaxTokens, got {other:?}"),
    }
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(llm.requests().len(), 1);
}

#[tokio::test]
async fn truncated_answer_without_text_does_not_use_up_iterations() {
    let body = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "" },
            "finish_reason": "length"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    let llm = MockLlm::start(vec![(200, body.to_string())]).await;

    let err = llm.agent().run::<String>("question").await.unwrap_err();

    assert!(matches!(err, AgentError::MaxTokens { .. }));
    assert_eq!(llm.requests().len(), 1);
}

#[tokio::test]
async fn print_to_writes_text_and_tool_calls() {
    let llm = MockLlm::start(vec![
        sse(vec![
            sse_chunk(
                json!({ "role": "assistant", "content": "Let me check." }),
                None,
            ),
            sse_chunk(
                json!({ "tool_calls": [{
                    "index": 0, "id": "call_1", "type": "function",
                    "function": { "name": "echo", "arguments": "{}" }
                }]}),
                None,
            ),
            sse_chunk(json!({}), Some("tool_calls")),
            sse_usage(),
        ]),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;
    let mut out = Vec::new();

    let result = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .print_to(&mut out)
        .await
        .unwrap();

    assert_eq!(result.answer, "Hello");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "Let me check.\n[tool call] echo({})\nHello\n"
    );
}

#[tokio::test]
async fn printer_options_choose_what_is_written() {
    let llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["done"]),
    ])
    .await;
    let mut out = Vec::new();

    llm.agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .listen(
            Printer::new(&mut out)
                .with_tool_calls(false)
                .with_tool_results(true),
        )
        .await
        .unwrap();

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "[tool result] echo: echoed\ndone\n"
    );
}

/// Writer that always fails.
struct BrokenWriter;

impl std::io::Write for BrokenWriter {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("disk full"))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn print_to_reports_write_errors() {
    let llm = MockLlm::start(vec![streamed_text_reply(&["Hello"])]).await;

    let err = llm
        .agent()
        .run_stream::<String>("question")
        .await
        .unwrap()
        .print_to(BrokenWriter)
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::Listener(_)));
}

#[tokio::test]
async fn print_to_async_writes_to_a_futures_writer() {
    let llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;
    let mut out = futures_util::io::Cursor::new(Vec::new());

    let result = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .print_to_async(&mut out)
        .await
        .unwrap();

    assert_eq!(result.answer, "Hello");
    assert_eq!(
        String::from_utf8(out.into_inner()).unwrap(),
        "[tool call] echo({})\nHello\n"
    );
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn print_to_tokio_writes_to_a_tokio_writer() {
    let llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;
    let mut out: Vec<u8> = Vec::new();

    let result = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .print_to_tokio(&mut out)
        .await
        .unwrap();

    assert_eq!(result.answer, "Hello");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "[tool call] echo({})\nHello\n"
    );
}

/// Listener that records what it sees, like a GUI would.
#[derive(Default)]
struct Recorder {
    events: Vec<String>,
}

impl EventListener for Recorder {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        self.events.push(match event {
            RunEvent::TextDelta(text) => format!("text:{text}"),
            RunEvent::ToolCall(call) => format!("call:{}", call.name),
            RunEvent::ToolResult { tool, content, .. } => format!("result:{tool}:{content}"),
            RunEvent::Finished {
                new_messages,
                iterations,
                ..
            } => format!("finished:{}:{iterations}", new_messages.len()),
            _ => "other".to_string(),
        });
        Ok(())
    }
}

#[tokio::test]
async fn listen_passes_every_event_to_the_listener() {
    let llm = MockLlm::start(vec![
        streamed_tool_call_reply(),
        streamed_text_reply(&["Hel", "lo"]),
    ])
    .await;
    let mut recorder = Recorder::default();

    let result = llm
        .agent()
        .with_toolbox(EchoToolBox)
        .run_stream::<String>("question")
        .await
        .unwrap()
        .listen(&mut recorder)
        .await
        .unwrap();

    assert_eq!(result.answer, "Hello");
    assert_eq!(
        recorder.events,
        vec![
            "call:echo",
            "result:echo:echoed",
            "text:Hel",
            "text:lo",
            "finished:4:2"
        ]
    );
}

/// Listener that fails on the first text.
struct FailingListener;

impl EventListener for FailingListener {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        match event {
            RunEvent::TextDelta(_) => Err("window closed".into()),
            _ => Ok(()),
        }
    }
}

#[tokio::test]
async fn a_listener_error_stops_the_run() {
    let llm = MockLlm::start(vec![streamed_text_reply(&["Hello"])]).await;

    let err = llm
        .agent()
        .run_stream::<String>("question")
        .await
        .unwrap()
        .listen(FailingListener)
        .await
        .unwrap_err();

    match err {
        AgentError::Listener(source) => assert_eq!(source.to_string(), "window closed"),
        other => panic!("expected AgentError::Listener, got {other:?}"),
    }
}

#[tokio::test]
async fn images_in_the_prompt_are_sent_and_kept_in_the_messages() {
    let llm = MockLlm::start(vec![
        text_reply("a logo"),
        streamed_text_reply(&["a ", "logo"]),
    ])
    .await;
    let agent = llm.agent();
    let prompt = crate::Prompt::text("What is this?")
        .with_image_url("https://example.com/logo.png")
        .with_image_bytes("image/png", b"hi");

    let result: AgentResult<String> = agent.run(prompt.clone()).await.unwrap();
    let stream = agent.run_stream::<String>(prompt.clone()).await.unwrap();
    let streamed = stream.finish().await.unwrap();

    let expected = json!([
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/logo.png"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
    ]);
    for (index, new_messages) in [result.new_messages, streamed.new_messages]
        .iter()
        .enumerate()
    {
        let user = &request_json(&llm, index)["messages"][1];
        assert_eq!(user["role"], json!("user"));
        assert_eq!(user["content"], expected);
        assert_eq!(new_messages[0].parts, prompt.parts());
    }
}

fn named_tool_call_reply(id: &str, name: &str, arguments: Value) -> (u16, String) {
    let body = json!({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments.to_string() }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });
    (200, body.to_string())
}

fn streamed_named_tool_call_reply(id: &str, name: &str, arguments: Value) -> (u16, String) {
    sse(vec![
        sse_chunk(
            json!({ "role": "assistant", "tool_calls": [{
                "index": 0, "id": id, "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() }
            }]}),
            None,
        ),
        sse_chunk(json!({}), Some("tool_calls")),
        sse_usage(),
    ])
}

/// Toolbox with one tool, `slow_job`, that returns a deferred task waiting on `gate`.
struct GatedDeferredToolBox {
    gate: Arc<tokio::sync::Notify>,
}

#[crate::tool::toolbox]
impl GatedDeferredToolBox {
    /// Starts a background job that finishes once `gate` is notified.
    #[tool]
    fn slow_job(&self) -> ToolResult<ToolReturn> {
        let gate = self.gate.clone();
        Ok(ToolReturn::deferred("job started", async move {
            gate.notified().await;
            Ok(ToolOutput::from_text("job finished"))
        }))
    }
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn deferred_task_can_be_awaited_during_a_streamed_run() {
    use crate::tool::{TokioDeferredTaskToolBox, ToolBoxSet};
    use futures_util::StreamExt;
    use std::time::Duration;

    let gate = Arc::new(tokio::sync::Notify::new());
    let deferred =
        TokioDeferredTaskToolBox::new().with_default_timeout(Duration::from_millis(10));
    let mut tools = ToolBoxSet::new();
    tools.add_tool(GatedDeferredToolBox { gate: gate.clone() });
    tools.add_tool(deferred.clone());

    let llm = MockLlm::start(vec![
        // 1. Model starts the deferred job
        streamed_named_tool_call_reply("call_1", "slow_job", json!({})),
        // 2. Model waits while the task is still blocked -> non-destructive timeout
        streamed_named_tool_call_reply("call_2", "wait_deferred_task", json!({"task_id": "call_1"})),
        // 3. After the gate opens, model waits again and gets the final output
        streamed_named_tool_call_reply("call_3", "wait_deferred_task", json!({"task_id": "call_1"})),
        // 4. Final answer
        streamed_text_reply(&["All done"]),
    ])
    .await;

    let mut stream = llm
        .agent()
        .with_toolbox(tools)
        .run_stream::<String>("run job")
        .await
        .unwrap();

    let mut final_result = None;
    let mut wait_results = 0;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::DeferredTask(task) => deferred.spawn(task),
            AgentEvent::ToolResult { tool, content, .. } if tool == "wait_deferred_task" => {
                wait_results += 1;
                if wait_results == 1 {
                    assert_eq!(content.text().as_deref(), Some("Task 'call_1' is still running."));
                    // Unblock the background task before the second wait_deferred_task call
                    gate.notify_one();
                } else {
                    assert_eq!(content.text().as_deref(), Some("job finished"));
                }
            }
            AgentEvent::Finished(result) => final_result = Some(result),
            _ => {}
        }
    }

    let result = final_result.unwrap();
    assert_eq!(result.answer, "All done");
    assert!(result.deferred_tasks.is_empty());
    assert!(result.unstarted_deferred_tasks.is_empty());
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn deferred_task_outlives_run_and_notifies_next_run_without_prompt() {
    use crate::tool::{TokioDeferredTaskToolBox, ToolBoxSet};
    use futures_util::StreamExt;

    let gate = Arc::new(tokio::sync::Notify::new());
    let deferred = TokioDeferredTaskToolBox::new();
    let mut tools = ToolBoxSet::new();
    tools.add_tool(GatedDeferredToolBox { gate: gate.clone() });
    tools.add_tool(deferred.clone());

    let llm = MockLlm::start(vec![
        // Run 1: starts the deferred task and answers while it is still running
        streamed_named_tool_call_reply("call_1", "slow_job", json!({})),
        streamed_text_reply(&["Working in the background"]),
        // Run 2: starts with only the notification, fetches the result and answers
        named_tool_call_reply("call_2", "get_deferred_task", json!({"task_id": "call_1"})),
        text_reply("Background result: job finished"),
    ])
    .await;

    let agent = llm.agent().with_toolbox(tools);
    let mut history = Vec::new();

    // Run 1
    let mut stream = agent.run_stream::<String>("start it").await.unwrap();
    let mut run1_result = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::DeferredTask(task) => deferred.spawn(task),
            AgentEvent::Finished(result) => run1_result = Some(result),
            _ => {}
        }
    }
    let run1 = run1_result.unwrap();
    assert_eq!(run1.answer, "Working in the background");
    assert_eq!(run1.deferred_tasks.len(), 1);
    assert!(!run1.deferred_tasks[0].is_finished());
    history.extend(run1.new_messages);

    // Task completes between runs
    gate.notify_one();
    while !run1.deferred_tasks[0].is_finished() {
        tokio::task::yield_now().await;
    }

    // Run 2 starts with an empty prompt and the deferred task handles from Run 1
    let run2: AgentResult<String> = agent
        .run(crate::Prompt::new())
        .with_history(&history)
        .with_deferred_tasks(run1.deferred_tasks)
        .await
        .unwrap();

    assert_eq!(run2.answer, "Background result: job finished");
    assert!(run2.deferred_tasks.is_empty());
    // The first new message of Run 2 is the user notification for call_1
    assert_eq!(
        run2.new_messages[0],
        Message::user("Deferred task call_1 (slow_job) has finished.")
    );
    // And the tool result of get_deferred_task was sent to the model in request #3
    let last_request = request_json(&llm, 3);
    assert!(last_request.to_string().contains("job finished"));
}

#[tokio::test]
async fn multiple_deferred_task_notifications_share_one_message() {
    use crate::tool::DeferredTask;

    let (completed_task_1, completed_handle_1) = DeferredTask::new(
        &ToolCall::new("call_1", "first_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("first result")) }),
    );
    completed_task_1.run().await;

    let (cancelled_task, cancelled_handle) = DeferredTask::new(
        &ToolCall::new("call_2", "cancelled_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("unused")) }),
    );
    drop(cancelled_task);

    let (completed_task_3, completed_handle_3) = DeferredTask::new(
        &ToolCall::new("call_3", "third_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("third result")) }),
    );
    completed_task_3.run().await;

    let llm = MockLlm::start(vec![text_reply("Acknowledged notifications")]).await;
    let result: AgentResult<String> = llm
        .agent()
        .run(crate::Prompt::new())
        .with_deferred_tasks(vec![
            completed_handle_1,
            cancelled_handle,
            completed_handle_3,
        ])
        .await
        .unwrap();

    assert_eq!(
        result.new_messages[0],
        Message::user(
            "Deferred task call_1 (first_job) has finished.\n\
             Deferred task call_2 (cancelled_job) was cancelled before completing.\n\
             Deferred task call_3 (third_job) has finished."
        )
    );
    assert_eq!(result.new_messages.len(), 2);
}

#[tokio::test]
async fn non_streaming_run_returns_unstarted_deferred_tasks() {
    let gate = Arc::new(tokio::sync::Notify::new());
    let llm = MockLlm::start(vec![
        named_tool_call_reply("call_1", "slow_job", json!({})),
        text_reply("started"),
    ])
    .await;

    let mut result: AgentResult<String> = llm
        .agent()
        .with_toolbox(GatedDeferredToolBox { gate: gate.clone() })
        .run("start")
        .await
        .unwrap();

    assert_eq!(result.answer, "started");
    assert_eq!(result.unstarted_deferred_tasks.len(), 1);
    assert_eq!(result.deferred_tasks.len(), 1);
    assert!(!result.deferred_tasks[0].is_finished());

    let task = result.unstarted_deferred_tasks.pop().unwrap();
    assert_eq!(task.id, "call_1");
    assert_eq!(task.tool, "slow_job");

    let run_fut = tokio::spawn(task.run());
    gate.notify_one();
    run_fut.await.unwrap();
    assert!(result.deferred_tasks[0].is_finished());
    let output = result.deferred_tasks[0].take_result().unwrap().unwrap();
    assert_eq!(output.text().as_deref(), Some("job finished"));
    assert!(result.deferred_tasks[0].is_consumed());
}

#[tokio::test]
async fn empty_prompt_without_finished_deferred_tasks_returns_error() {
    let llm = MockLlm::start(vec![]).await;
    let agent = llm.agent();

    // 1. Empty prompt with no deferred tasks
    let err = agent.run::<String>(crate::Prompt::new()).await.unwrap_err();
    assert!(matches!(err, AgentError::EmptyPrompt));

    // 2. Empty string prompt with no deferred tasks
    let err = agent.run::<String>("").await.unwrap_err();
    assert!(matches!(err, AgentError::EmptyPrompt));

    // 3. Whitespace-only prompt with no deferred tasks
    let err = agent.run::<String>("   ").await.unwrap_err();
    assert!(matches!(err, AgentError::EmptyPrompt));

    // 4. Empty prompt with an unfinished deferred task
    let (_task, handle) = crate::tool::DeferredTask::new(
        &ToolCall::new("c1", "slow_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("done")) }),
    );
    let err = agent
        .run::<String>(crate::Prompt::new())
        .with_deferred_tasks(vec![handle])
        .await
        .unwrap_err();
    assert!(matches!(err, AgentError::EmptyPrompt));

    // 5. Streamed empty prompt also yields EmptyPrompt error
    use futures_util::StreamExt;
    let mut stream = agent
        .run_stream::<String>(crate::Prompt::new())
        .await
        .unwrap();
    let stream_err = stream.next().await.unwrap().unwrap_err();
    assert!(matches!(stream_err, AgentError::EmptyPrompt));

    // Ensure the model was never called
    assert!(llm.requests().is_empty());
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn failed_deferred_tasks_always_reach_the_model_as_recoverable_errors() {
    use crate::tool::{TokioDeferredTaskToolBox, ToolBox};
    use std::time::Duration;

    let errors = [
        ToolError::NotFound("x".into()),
        ToolError::InvalidArguments("bad".into()),
        ToolError::Failed("boom".into()),
        ToolError::Configuration("no key".into()),
        ToolError::fatal(std::io::Error::other("io")),
    ];

    let deferred = TokioDeferredTaskToolBox::new();
    for (index, error) in errors.into_iter().enumerate() {
        let id = format!("call_{index}");
        let (task, _handle) = crate::tool::DeferredTask::new(
            &ToolCall::new(&id, "slow_job", json!({})),
            Box::pin(async move { Err(error) }),
        );
        deferred.spawn(task);

        let call = ToolCall::new(
            "fetch",
            "wait_deferred_task",
            json!({"task_id": id, "timeout_seconds": 5}),
        );
        let err = tokio::time::timeout(Duration::from_secs(10), deferred.call_tool(&call))
            .await
            .unwrap()
            .unwrap_err();
        assert!(
            matches!(err, ToolError::Failed(_)),
            "task {id} failed with {err:?}, expected a recoverable ToolError::Failed"
        );
        assert!(err.is_recoverable());
    }
}

#[tokio::test]
async fn dropped_deferred_task_marks_handle_cancelled_and_notifies_model() {
    let llm = MockLlm::start(vec![text_reply("Acknowledged cancellation")]).await;
    let agent = llm.agent();

    let (task, handle) = crate::tool::DeferredTask::new(
        &ToolCall::new("c1", "slow_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("done")) }),
    );
    assert!(!handle.is_finished());
    assert!(!handle.is_cancelled());

    // Dropping the task without running it marks the shared state as cancelled
    drop(task);
    assert!(handle.is_finished());
    assert!(handle.is_cancelled());
    assert!(handle.take_result().is_none());

    // A run with an empty prompt and a cancelled handle sends a cancellation notification
    let result: AgentResult<String> = agent
        .run(crate::Prompt::new())
        .with_deferred_tasks(vec![handle])
        .await
        .unwrap();

    assert_eq!(result.answer, "Acknowledged cancellation");
    assert!(result.deferred_tasks.is_empty());
    assert_eq!(
        result.new_messages[0],
        Message::user("Deferred task c1 (slow_job) was cancelled before completing.")
    );
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn panicking_deferred_task_cancels_handle_and_wakes_waiters_with_recoverable_error() {
    use crate::tool::{TokioDeferredTaskToolBox, ToolBox};
    use std::time::Duration;

    let deferred = TokioDeferredTaskToolBox::new();
    let (task, handle) = crate::tool::DeferredTask::new(
        &ToolCall::new("panic_1", "slow_job", json!({})),
        Box::pin(async {
            panic!("boom inside background task");
        }),
    );
    deferred.spawn(task);

    let call = ToolCall::new(
        "fetch",
        "wait_deferred_task",
        json!({"task_id": "panic_1", "timeout_seconds": 5}),
    );
    let err = tokio::time::timeout(Duration::from_secs(2), deferred.call_tool(&call))
        .await
        .expect("wait_deferred_task should be woken when task panics, not time out")
        .unwrap_err();
    assert!(matches!(err, ToolError::Failed(_)));
    assert!(err.is_recoverable());
    assert!(handle.is_cancelled());
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn duplicate_deferred_task_id_does_not_overwrite_first_task() {
    use crate::tool::{TokioDeferredTaskToolBox, ToolBox};
    use std::time::Duration;

    let deferred = TokioDeferredTaskToolBox::new();
    let (task1, handle1) = crate::tool::DeferredTask::new(
        &ToolCall::new("dup_id", "slow_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("first result")) }),
    );
    let (task2, handle2) = crate::tool::DeferredTask::new(
        &ToolCall::new("dup_id", "slow_job", json!({})),
        Box::pin(async { Ok(ToolOutput::from_text("second result")) }),
    );

    deferred.spawn(task1);
    deferred.spawn(task2);

    // The duplicate second task is dropped immediately and marked cancelled
    assert!(handle2.is_cancelled());

    // Waiting on dup_id returns the first task's result
    let call = ToolCall::new(
        "fetch",
        "wait_deferred_task",
        json!({"task_id": "dup_id", "timeout_seconds": 5}),
    );
    let ret = tokio::time::timeout(Duration::from_secs(2), deferred.call_tool(&call))
        .await
        .unwrap()
        .unwrap();
    let output = match ret {
        ToolReturn::Done(output) => output,
        ToolReturn::Deferred { .. } => unreachable!("expected ToolReturn::Done"),
    };
    assert_eq!(output.text().as_deref(), Some("first result"));
    assert!(handle1.is_consumed());
}
