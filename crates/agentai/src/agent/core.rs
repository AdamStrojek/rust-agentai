//! The agent loop shared by every way of running an agent.

use super::{AgentError, AgentEvent, ChatOptions, LlmError, RunResult, StopReason, Usage};
use crate::message::{Message, ToolCall, ToolResponse};
use crate::tool::ToolBox;
use async_stream::try_stream;
use futures_util::{Stream, StreamExt, pin_mut};
use genai::Client;
use genai::chat::{ChatRequest, ChatStreamEvent, JsonSpec, StreamEnd};
use log::{debug, trace};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::{Value, from_str, json};
use std::sync::Arc;

/// Agent configuration with the overrides of one run applied.
pub(super) struct ResolvedRun {
    pub(super) client: Client,
    pub(super) system: String,
    pub(super) model: String,
    pub(super) toolbox: Option<Arc<dyn ToolBox>>,
    pub(super) max_iterations: usize,
    pub(super) chat_options: ChatOptions,
    /// JSON schema of the answer for structured output, `None` for a text answer
    pub(super) response_format: Option<JsonSpec>,
}

/// One model call, reduced to what the loop needs.
struct Step {
    tool_calls: Vec<ToolCall>,
    /// Assistant message to store when the model asked for tools. Keeps text, reasoning and
    /// provider continuation data next to the tool calls.
    tool_use_message: Option<Message>,
    texts: Vec<String>,
    usage: Usage,
    stop_reason: Option<StopReason>,
    reasoning_content: Option<String>,
}

/// Runs the agent loop. The history is only read; the messages of this run are returned in
/// [`RunResult::new_messages`]. With `streaming`, the model answers are streamed and text and
/// reasoning deltas are sent as events.
pub(super) fn run_loop<D>(
    run: ResolvedRun,
    history: Vec<Message>,
    prompt: String,
    streaming: bool,
) -> impl Stream<Item = Result<AgentEvent<D>, AgentError>> + Send + 'static
where
    D: DeserializeOwned + JsonSchema + Send + 'static,
{
    try_stream! {
        debug!("Agent Question: {prompt}");
        let mut new_messages = vec![Message::user(prompt)];
        let mut usage = Usage::default();

        // Agent options, plus the response format for structured output
        let mut chat_opts = run.chat_options.to_genai();
        let is_answer_string = run.response_format.is_none();
        if let Some(format) = run.response_format.clone() {
            chat_opts = chat_opts.with_response_format(format);
        }
        if streaming {
            // Without captures the end of the stream holds no tool calls, usage or reasoning
            chat_opts = chat_opts
                .with_capture_usage(true)
                .with_capture_content(true)
                .with_capture_tool_calls(true)
                .with_capture_reasoning_content(true);
        }

        for iteration in 0..run.max_iterations {
            debug!("Agent iteration: {iteration}");
            let mut chat_req = ChatRequest::from_messages(
                history
                    .iter()
                    .chain(&new_messages)
                    .cloned()
                    .map(Message::into_genai)
                    .collect(),
            );
            if !run.system.is_empty() {
                chat_req = chat_req.with_system(run.system.clone());
            }
            if let Some(toolbox) = &run.toolbox {
                let definitions =
                    toolbox
                        .tools_definitions()
                        .map_err(|source| AgentError::Tool {
                            tool: String::new(),
                            source,
                        })?;
                chat_req = chat_req.with_tools(definitions.into_iter().map(crate::tool::Tool::into_genai));
            }

            let step = if streaming {
                let mut stream = run
                    .client
                    .exec_chat_stream(&run.model, chat_req, Some(&chat_opts))
                    .await
                    .map_err(|err| AgentError::Llm(LlmError::from_genai(err)))?
                    .stream;
                let mut end = None;
                while let Some(event) = stream.next().await {
                    match event.map_err(|err| AgentError::Llm(LlmError::from_genai(err)))? {
                        ChatStreamEvent::Chunk(chunk) => yield AgentEvent::TextDelta(chunk.content),
                        ChatStreamEvent::ReasoningChunk(chunk) => {
                            yield AgentEvent::ReasoningDelta(chunk.content)
                        }
                        ChatStreamEvent::End(stream_end) => end = Some(stream_end),
                        _ => {}
                    }
                }
                step_from_stream_end(end.unwrap_or_default())
            } else {
                exec_step(&run, chat_req, &chat_opts).await?
            };
            usage += step.usage;

            // Some models return text (e.g. "Let me check...") together with tool calls; the
            // tool calls win.
            if !step.tool_calls.is_empty() {
                if let Some(message) = step.tool_use_message {
                    new_messages.push(message);
                }
                let mut fatal: Option<AgentError> = None;
                for tool_call in step.tool_calls {
                    if fatal.is_some() {
                        // The run is stopping. Every tool call still needs a response, or the
                        // provider would reject the conversation.
                        new_messages.push(Message::tool_response(ToolResponse::new(
                            &tool_call,
                            "Tool was not executed because an earlier tool call failed",
                        )));
                        continue;
                    }
                    yield AgentEvent::ToolCall(tool_call.clone());
                    let outcome = call_tool(run.toolbox.as_deref(), &tool_call).await;
                    let ToolOutcome { content, is_error, .. } = outcome;
                    fatal = outcome.fatal;
                    new_messages.push(Message::tool_response(ToolResponse::new(&tool_call, content.clone())));
                    yield AgentEvent::ToolResult {
                        call_id: tool_call.id,
                        tool: tool_call.name,
                        content,
                        is_error,
                    };
                }
                if let Some(err) = fatal {
                    Err(err)?;
                }
                continue;
            }

            if step.texts.is_empty() {
                // Nothing usable in this response (for example only reasoning), ask again
                debug!("Agent iteration {iteration}: response has no text and no tool calls");
                continue;
            }
            // Structured output may be split across several text parts
            let raw = step.texts.join("");
            debug!("Agent Answer: {raw}");
            if step.stop_reason.as_ref().is_some_and(StopReason::is_max_tokens) {
                Err(AgentError::MaxTokens { raw: raw.clone() })?;
            }
            new_messages.push(Message::assistant(raw.clone()));
            let answer = parse_answer::<D>(raw, is_answer_string)?;
            yield AgentEvent::Finished(RunResult {
                answer,
                new_messages,
                usage,
                stop_reason: step.stop_reason,
                reasoning_content: step.reasoning_content,
                iterations: iteration + 1,
            });
            return;
        }

        Err(AgentError::MaxIterationsReached(run.max_iterations))?;
    }
}

/// Drives a run to the end and returns its result.
pub(super) async fn finish<D>(
    events: impl Stream<Item = Result<AgentEvent<D>, AgentError>>,
) -> Result<RunResult<D>, AgentError> {
    pin_mut!(events);
    while let Some(event) = events.next().await {
        if let AgentEvent::Finished(result) = event? {
            return Ok(result);
        }
    }
    unreachable!("the agent loop ends with a result or an error")
}

/// Sends one request to the model.
async fn exec_step(
    run: &ResolvedRun,
    chat_req: ChatRequest,
    chat_opts: &genai::chat::ChatOptions,
) -> Result<Step, AgentError> {
    let resp = run
        .client
        .exec_chat(&run.model, chat_req, Some(chat_opts))
        .await
        .map_err(|err| AgentError::Llm(LlmError::from_genai(err)))?;
    let tool_calls: Vec<ToolCall> = resp
        .tool_calls()
        .into_iter()
        .cloned()
        .map(ToolCall::from_genai)
        .collect();
    let usage = Usage::from_genai(&resp.usage);
    let stop_reason = resp.stop_reason.clone().map(StopReason::from_genai);
    let reasoning_content = resp.reasoning_content.clone();
    let (tool_use_message, texts) = if tool_calls.is_empty() {
        (None, resp.content.into_texts())
    } else {
        (
            resp.into_assistant_message_for_tool_use()
                .map(Message::from_genai),
            Vec::new(),
        )
    };
    Ok(Step {
        tool_calls,
        tool_use_message,
        texts,
        usage,
        stop_reason,
        reasoning_content,
    })
}

/// Builds a step from the end of a streamed answer.
fn step_from_stream_end(end: StreamEnd) -> Step {
    let tool_calls: Vec<ToolCall> = end
        .captured_tool_calls()
        .map(|calls| {
            calls
                .into_iter()
                .cloned()
                .map(ToolCall::from_genai)
                .collect()
        })
        .unwrap_or_default();
    let usage = end
        .captured_usage
        .as_ref()
        .map(Usage::from_genai)
        .unwrap_or_default();
    let stop_reason = end.captured_stop_reason.clone().map(StopReason::from_genai);
    let reasoning_content = end.captured_reasoning_content.clone();
    let (tool_use_message, texts) = if tool_calls.is_empty() {
        (None, end.into_texts().unwrap_or_default())
    } else {
        (
            end.into_assistant_message_for_tool_use()
                .map(Message::from_genai),
            Vec::new(),
        )
    };
    Step {
        tool_calls,
        tool_use_message,
        texts,
        usage,
        stop_reason,
        reasoning_content,
    }
}

/// What a tool call produced.
struct ToolOutcome {
    /// Sent to the model as the tool result.
    content: String,
    is_error: bool,
    /// Set for a fatal error: the run stops once every tool call has a response.
    fatal: Option<AgentError>,
}

impl ToolOutcome {
    fn recoverable(content: String) -> Self {
        Self {
            content,
            is_error: true,
            fatal: None,
        }
    }
}

async fn call_tool(toolbox: Option<&dyn ToolBox>, tool_call: &ToolCall) -> ToolOutcome {
    trace!(
        "Tool request: {} with arguments: {}",
        tool_call.name, tool_call.arguments
    );
    let Some(toolbox) = toolbox else {
        debug!(
            "Model requested tool '{}' but no toolbox was provided",
            tool_call.name
        );
        return ToolOutcome::recoverable(format!(
            "Tool '{}' is not available: no tools were provided",
            tool_call.name
        ));
    };
    match toolbox
        .call_tool(tool_call.name.clone(), tool_call.arguments.clone())
        .await
    {
        Ok(result) => {
            trace!("Tool result: {result}");
            ToolOutcome {
                content: result,
                is_error: false,
                fatal: None,
            }
        }
        Err(err) if err.is_recoverable() => {
            // Recoverable errors go back to the model, so it can react to them. Some MCP
            // servers return important information as errors.
            trace!("Tool error: {err}");
            ToolOutcome::recoverable(err.to_string())
        }
        Err(source) => {
            // Retrying cannot help, so the run stops
            debug!("Fatal error in tool '{}': {source}", tool_call.name);
            ToolOutcome {
                content: format!("Tool failed: {source}"),
                is_error: true,
                fatal: Some(AgentError::Tool {
                    tool: tool_call.name.clone(),
                    source,
                }),
            }
        }
    }
}

/// JSON schema of `D` as a response format.
pub(super) fn response_format<D: JsonSchema>() -> Result<JsonSpec, AgentError> {
    let mut response_schema =
        serde_json::to_value(schema_for!(D)).map_err(|err| AgentError::Schema(err.to_string()))?;
    let obj = response_schema.as_object_mut().ok_or_else(|| {
        AgentError::Schema("JSON schema of the answer type is not an object".to_string())
    })?;
    // Schemars attaches additional fields and not every LLM accepts them (Gemini)
    obj.remove("$schema");
    obj.remove("title");
    Ok(JsonSpec::new("ResponseFormat", json!(obj)))
}

fn parse_answer<D: DeserializeOwned>(raw: String, is_answer_string: bool) -> Result<D, AgentError> {
    let resp = if is_answer_string {
        // `D: DeserializeOwned` cannot be returned as a `String` directly, so the text is
        // quoted as a JSON string and deserialized back.
        Value::String(raw.clone()).to_string()
    } else {
        raw.clone()
    };
    from_str(&resp).map_err(|source| AgentError::InvalidAnswer { raw, source })
}
