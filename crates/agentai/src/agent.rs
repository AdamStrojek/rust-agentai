//! # Core components of AI Agents
//!
//! This module contains core components that you can use in your AI Agents
//!
//! To read more about Agents look into [crate::agent::Agent]
//!
//! To read more about structured output look into [crate::structured_output]
//!
//! To read more about tool look into [crate::tool]

use crate::tool::{ToolBox, ToolError};
use genai::Client;
/// Provider (genai adapter) selector used by [`Agent::new_with_provider`], re-exported from genai.
pub use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, JsonSpec, ToolCall, ToolResponse};
use genai::resolver::{AuthData, Endpoint};
use log::{debug, trace};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::{Value, from_str, json};
use std::any::TypeId;
use thiserror::Error;

#[cfg(test)]
mod tests;

/// Errors returned by [`Agent`].
///
/// Match on the variants to decide what to do: retry a transient provider failure (see
/// [`AgentError::is_transient`]), show the raw answer of the model for
/// [`InvalidAnswer`](AgentError::InvalidAnswer), and so on.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum AgentError {
    /// The genai client could not be built (TLS, configuration).
    #[error("failed to create the LLM client: {0}")]
    ClientInit(#[source] genai::Error),
    /// The request to the model failed (network, HTTP status, provider error).
    #[error("LLM request failed: {0}")]
    Llm(#[source] genai::Error),
    /// The JSON schema of the answer type could not be generated or is not an object.
    #[error("invalid answer schema: {0}")]
    Schema(String),
    /// The final answer of the model could not be parsed into the requested type.
    #[error("failed to parse the model answer: {source}")]
    InvalidAnswer {
        /// Text returned by the model.
        raw: String,
        #[source]
        source: serde_json::Error,
    },
    /// The loop ended without a final answer. Holds the number of iterations.
    #[error("no final answer after {0} iterations")]
    MaxIterationsReached(usize),
    /// A tool failed with a fatal error ([`ToolError::is_recoverable`] is `false`),
    /// which stops the run.
    #[error("tool '{tool}' failed: {source}")]
    Tool {
        /// Name of the tool that failed. For errors raised before a call, such as
        /// [`ToolBox::tools_definitions`], it is empty.
        tool: String,
        #[source]
        source: ToolError,
    },
}

impl AgentError {
    /// Returns `true` when trying again may succeed: the provider answered with HTTP 408, 425,
    /// 429 or 5xx, or the request failed because of a timeout or a connection problem.
    pub fn is_transient(&self) -> bool {
        let Self::Llm(err) = self else {
            return false;
        };
        if let Some(status) = err.status() {
            return status.is_server_error() || matches!(status.as_u16(), 408 | 425 | 429);
        }
        match err {
            genai::Error::WebModelCall { webc_error, .. }
            | genai::Error::WebAdapterCall { webc_error, .. } => match webc_error {
                genai::webc::Error::Reqwest(err) => err.is_timeout() || err.is_connect(),
                _ => false,
            },
            genai::Error::WebStream { .. } => true,
            _ => false,
        }
    }
}

/// The `Agent` struct represents an agent that interacts with a chat model.
/// It maintains a history of chat messages, a set of tools, and a context.
///
/// As `Context` you can provide any structure. Such object will not be used by
/// `Agent` itself, but it will be passed in unmodified state as reference to any
/// `AgentTool` trait, that was registered to be used.
#[derive(Clone)]
pub struct Agent {
    /// Reference to GenAI Client
    client: Client,

    // tool_box: impl ToolBox,
    history: Vec<ChatMessage>,
}

impl Agent {
    /// Creates a new `Agent` instance.
    ///
    /// This creation method will create a new agent instance with a default GenAI client.
    /// genai picks the provider from the model name and reads the provider's API key
    /// from its usual environment variable.
    ///
    /// # Arguments
    ///
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Errors
    ///
    /// Returns an error when the GenAI client cannot be created.
    pub fn new(system: &str) -> Result<Self, AgentError> {
        let client = Client::new().map_err(AgentError::ClientInit)?;

        Ok(Self::new_with_client(client, system))
    }

    /// Creates a new `Agent` instance with provided GenAI Client
    ///
    /// # Arguments
    ///
    /// * `client` - User provided GenAI Client
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Returns
    ///
    /// A new `Agent` instance.
    pub fn new_with_client(client: Client, system: &str) -> Self {
        Self {
            client,
            history: vec![ChatMessage::system(system.trim())],
        }
    }

    /// Creates a new `Agent` instance that talks to an OpenAI-compatible endpoint
    /// (OpenAI, OpenRouter, LiteLLM, Ollama, ...).
    ///
    /// Every model name is sent through genai's OpenAI adapter to `base_url`, authenticated
    /// with `api_key`. Model names are passed as they are, so names like `openai/gpt-4.1-mini`
    /// or `vendor/model:free` work. A name with another genai namespace, such as
    /// `anthropic::claude-sonnet-4-5`, fails with an adapter mismatch error.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the API, for example `https://openrouter.ai/api/v1/`.
    /// * `api_key` - API key sent with every request.
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Errors
    ///
    /// Returns an error when the GenAI client cannot be created.
    pub fn new_with_url(base_url: &str, api_key: &str, system: &str) -> Result<Self, AgentError> {
        Self::new_with_provider(AdapterKind::OpenAI, Some(base_url), api_key, system)
    }

    /// Creates a new `Agent` instance bound to one genai provider (adapter), with an explicit
    /// API key and, optionally, a custom endpoint.
    ///
    /// Every model name is sent through the adapter for `provider`, so bare names work without a
    /// genai namespace. For example, with [`AdapterKind::OpenRouter`] the model
    /// `nvidia/nemotron-3.5-super-vl:free` goes to OpenRouter as is. A name with another genai
    /// namespace, such as `anthropic::claude-sonnet-4-5`, fails with an adapter mismatch error.
    ///
    /// ```no_run
    /// use agentai::{AdapterKind, Agent};
    ///
    /// # fn main() -> Result<(), agentai::AgentError> {
    /// let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    /// // Native OpenRouter adapter with its default endpoint
    /// let agent = Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, "You are helpful")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `provider` - genai adapter to use, for example [`AdapterKind::OpenRouter`],
    ///   [`AdapterKind::Anthropic`] or [`AdapterKind::OpenAI`].
    /// * `base_url` - Endpoint override. `None` keeps the provider's default endpoint, for
    ///   example `https://openrouter.ai/api/v1/` for OpenRouter.
    /// * `api_key` - API key sent with every request.
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Errors
    ///
    /// Returns an error when the GenAI client cannot be created.
    pub fn new_with_provider(
        provider: AdapterKind,
        base_url: Option<&str>,
        api_key: &str,
        system: &str,
    ) -> Result<Self, AgentError> {
        let auth = AuthData::from_single(api_key);
        let builder = Client::builder().with_adapter_kind(provider);
        let builder = match base_url {
            Some(base_url) => builder.append_provider_config(
                provider,
                (Endpoint::from_owned(base_url.to_string()), auth),
            ),
            None => builder.append_provider_config(provider, auth),
        };
        let client = builder.build().map_err(AgentError::ClientInit)?;
        Ok(Self::new_with_client(client, system))
    }

    /// Runs the agent with the given model and prompt.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for the chat.
    /// * `prompt` - The prompt to send to the chat model.
    ///
    /// # Returns
    ///
    /// A result containing the deserialized response.
    ///
    /// # Errors
    ///
    /// Returns an [`AgentError`]:
    /// - [`Llm`](AgentError::Llm) when the request to the model fails (see
    ///   [`AgentError::is_transient`]);
    /// - [`Schema`](AgentError::Schema) when the schema of `D` cannot be generated;
    /// - [`InvalidAnswer`](AgentError::InvalidAnswer) when the answer does not match `D`;
    /// - [`MaxIterationsReached`](AgentError::MaxIterationsReached) when there is no final answer;
    /// - [`Tool`](AgentError::Tool) when a tool fails with a fatal error. Recoverable tool errors
    ///   ([`ToolError::is_recoverable`]) are sent to the model instead.
    ///
    /// ## Structured Output
    /// Type returned by this function is responsible for setting LLM response into structured output
    ///
    /// For more information go to [crate::structured_output]
    pub async fn run<D>(
        &mut self,
        model: &str,
        prompt: &str,
        toolbox: Option<&dyn ToolBox>,
    ) -> Result<D, AgentError>
    where
        D: DeserializeOwned + JsonSchema + 'static,
    {
        // TODO change returned type
        // Need to create new type that will provide not only response structure,
        // but also statistics and reasoning.
        debug!("Agent Question: {prompt}");
        // Add new request to history
        // TODO: Create new history trait
        // This will allow on configuring behaviour of messages. When doing multi-agent
        // approach we could decide what history is being used, should we save all messages etc.
        // TODO: What to do when message have images? Should we send them only once?
        self.history.push(ChatMessage::user(prompt));

        // Prepare chat options
        // TODO: Allow to provide chat options to GenAI
        // This should be be part
        let mut chat_opts = ChatOptions::default().with_temperature(0.2);

        let is_answer_string = TypeId::of::<String>() == TypeId::of::<D>();
        if !is_answer_string {
            // If answer type is more complex then add response format to request options
            let mut response_schema = serde_json::to_value(schema_for!(D))
                .map_err(|err| AgentError::Schema(err.to_string()))?;
            let obj = response_schema.as_object_mut().ok_or_else(|| {
                AgentError::Schema("JSON schema of the answer type is not an object".to_string())
            })?;
            // Schemars attaches additional fields and not every LLM accepts them (Gemini)
            obj.remove("$schema");
            obj.remove("title");
            chat_opts = chat_opts.with_response_format(JsonSpec::new("ResponseFormat", json!(obj)));
        }

        // TODO move it to config structure
        let max_iterations = 5;

        for iteration in 0..max_iterations {
            debug!("Agent iteration: {iteration}");
            // Create chat request
            let mut chat_req = ChatRequest::new(self.history.clone());
            if let Some(toolbox) = toolbox {
                let definitions =
                    toolbox
                        .tools_definitions()
                        .map_err(|source| AgentError::Tool {
                            tool: String::new(),
                            source,
                        })?;
                chat_req = chat_req.with_tools(definitions);
            }
            let chat_resp = self
                .client
                .exec_chat(model, chat_req, Some(&chat_opts))
                .await
                .map_err(AgentError::Llm)?;

            // Collect tool calls before the response is consumed. Some models return text
            // (e.g. "Let me check...") together with tool calls; the tool calls win.
            let tool_calls: Vec<ToolCall> = chat_resp.tool_calls().into_iter().cloned().collect();

            if !tool_calls.is_empty() {
                // Keeps text, reasoning and provider continuation data next to the tool calls
                if let Some(assistant_msg) = chat_resp.into_assistant_message_for_tool_use() {
                    self.history.push(assistant_msg);
                }
                let mut fatal: Option<AgentError> = None;
                for tool_call in tool_calls {
                    if fatal.is_some() {
                        // The run is stopping. Every tool call still needs a response, or the
                        // history would be rejected by the provider on the next `run`.
                        self.history.push(
                            ToolResponse::from_tool_call(
                                &tool_call,
                                "Tool was not executed because an earlier tool call failed"
                                    .to_string(),
                            )
                            .into(),
                        );
                        continue;
                    }
                    trace!(
                        "Tool request: {} with arguments: {}",
                        tool_call.fn_name, tool_call.fn_arguments
                    );
                    let content = match toolbox {
                        Some(toolbox) => {
                            match toolbox
                                .call_tool(
                                    tool_call.fn_name.clone(),
                                    tool_call.fn_arguments.clone(),
                                )
                                .await
                            {
                                Ok(result) => {
                                    trace!("Tool result: {result}");
                                    result
                                }
                                Err(err) if err.is_recoverable() => {
                                    // Recoverable errors go back to the model, so it can react
                                    // to them. Some MCP servers return important information
                                    // as errors.
                                    trace!("Tool error: {err}");
                                    err.to_string()
                                }
                                Err(source) => {
                                    // Retrying cannot help, so the run stops
                                    debug!("Fatal error in tool '{}': {source}", tool_call.fn_name);
                                    let message = format!("Tool failed: {source}");
                                    fatal = Some(AgentError::Tool {
                                        tool: tool_call.fn_name.clone(),
                                        source,
                                    });
                                    message
                                }
                            }
                        }
                        None => {
                            debug!(
                                "Model requested tool '{}' but no toolbox was provided",
                                tool_call.fn_name
                            );
                            format!(
                                "Tool '{}' is not available: no tools were provided",
                                tool_call.fn_name
                            )
                        }
                    };
                    self.history
                        .push(ToolResponse::from_tool_call(&tool_call, content).into());
                }
                if let Some(err) = fatal {
                    return Err(err);
                }
                continue;
            }

            let texts = chat_resp.content.into_texts();
            if texts.is_empty() {
                // Nothing usable in this response (for example only reasoning), ask again
                debug!("Agent iteration {iteration}: response has no text and no tool calls");
                continue;
            }
            // Structured output may be split across several text parts
            let raw = texts.join("");
            debug!("Agent Answer: {raw}");
            self.history.push(ChatMessage::assistant(raw.clone()));
            let resp = if is_answer_string {
                // TODO: Workaround when choosing String as response type. Because we are
                // expecting D: DeserializeOwned then we can't return String directly.
                // To workaround this I escape content and later deserialize it using
                // serde_json::from_str to correct "struct" (String)
                Value::String(raw.clone()).to_string()
            } else {
                raw.clone()
            };
            return from_str(&resp).map_err(|source| AgentError::InvalidAnswer { raw, source });
        }

        Err(AgentError::MaxIterationsReached(max_iterations))
    }
}
