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
use genai::resolver::{AuthData, Endpoint};
use std::sync::Arc;
use thiserror::Error;

mod core;
mod model;
mod request;
mod stream;

pub use model::{ChatOptions, Provider, ReasoningEffort, StopReason, ToolChoice, Usage};
pub use request::{AgentRequest, Complete, RunOptions, RunResult, Streaming};
pub use stream::{AgentEvent, AgentStream};

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
    /// The LLM client could not be built (TLS, configuration).
    #[error("failed to create the LLM client: {0}")]
    ClientInit(#[source] LlmError),
    /// The request to the model failed (network, HTTP status, provider error).
    #[error("LLM request failed: {0}")]
    Llm(#[source] LlmError),
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
    /// The final answer was cut off by the token limit (for example `max_tokens` in
    /// [`ChatOptions`]). Holds the truncated text. Raise the limit or ask for a shorter answer.
    #[error("the model answer was truncated by the token limit")]
    MaxTokens {
        /// Truncated text returned by the model.
        raw: String,
    },
    /// No model was set, neither with [`Agent::with_model`] nor for the run
    /// ([`AgentRequest::with_model`]).
    #[error("no model set: use Agent::with_model or AgentRequest::with_model")]
    MissingModel,
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
        match self {
            Self::Llm(err) => err.is_transient(),
            _ => false,
        }
    }
}

/// Error of the LLM client: a failed request, or a client that could not be built.
///
/// The message describes the failure, [`status`](LlmError::status) gives the HTTP status when
/// the provider answered with one, and [`is_transient`](LlmError::is_transient) tells whether a
/// retry may succeed.
#[derive(Error, Debug)]
#[error("{source}")]
pub struct LlmError {
    status: Option<u16>,
    transient: bool,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl LlmError {
    /// HTTP status returned by the provider, if any.
    pub fn status(&self) -> Option<u16> {
        self.status
    }

    /// Returns `true` when trying again may succeed: HTTP 408, 425, 429 or 5xx, a timeout or a
    /// connection problem.
    pub fn is_transient(&self) -> bool {
        self.transient
    }
}

impl LlmError {
    pub(crate) fn from_genai(err: genai::Error) -> Self {
        let status = err.status().map(|status| status.as_u16());
        let transient = match status {
            Some(status) => status >= 500 || matches!(status, 408 | 425 | 429),
            None => match &err {
                genai::Error::WebModelCall { webc_error, .. }
                | genai::Error::WebAdapterCall { webc_error, .. } => match webc_error {
                    genai::webc::Error::Reqwest(err) => err.is_timeout() || err.is_connect(),
                    _ => false,
                },
                genai::Error::WebStream { .. } => true,
                _ => false,
            },
        };
        Self {
            status,
            transient,
            source: Box::new(err),
        }
    }
}

/// Default number of model calls in one run.
pub const DEFAULT_MAX_ITERATIONS: usize = 5;

/// Default temperature sent with every request, unless [`Agent::with_chat_options`] replaces it.
pub const DEFAULT_TEMPERATURE: f64 = 0.2;

/// The `Agent` struct represents an agent that interacts with a chat model.
///
/// An agent is configuration only: the LLM client, the system prompt, the default model and
/// toolbox, how many model calls one run may make, and the [`ChatOptions`] sent with every
/// request. It does not keep the conversation. You own the history, pass it to
/// [`AgentRequest::with_history`] and append [`RunResult::new_messages`] to it, so one agent can
/// serve many conversations at once. Cloning an agent is cheap.
///
/// ```no_run
/// use agentai::{Agent, ChatOptions, ReasoningEffort};
///
/// # fn main() -> Result<(), agentai::AgentError> {
/// let agent = Agent::new("You are a useful assistant")?
///     .with_model("gpt-4o")
///     .with_max_iterations(10)
///     .with_chat_options(
///         ChatOptions::default()
///             .with_temperature(0.0)
///             .with_max_tokens(2000)
///             .with_reasoning_effort(ReasoningEffort::Low),
///     );
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Agent {
    /// LLM client
    client: Client,

    /// System prompt sent with every request. It is not part of the history.
    system: String,

    /// Model used when the run does not set one
    model: Option<String>,

    /// Toolbox used when the run does not set one
    toolbox: Option<Arc<dyn ToolBox>>,

    /// Maximum number of model calls in one run
    max_iterations: usize,

    /// Options sent with every request
    chat_options: ChatOptions,
}

impl Agent {
    /// Creates a new `Agent` that picks the provider from the model name.
    ///
    /// `gpt-*` models go to OpenAI, `claude-*` to Anthropic, `gemini-*` to Gemini, and so on; a
    /// `provider::model` prefix such as `open_router::vendor/model` selects a provider
    /// explicitly. The API key is read from the provider's usual environment variable, for
    /// example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `OPEN_ROUTER_API_KEY`.
    ///
    /// # Arguments
    ///
    /// * `system` - The system prompt sent with every request.
    ///
    /// # Errors
    ///
    /// Returns an error when the LLM client cannot be created.
    pub fn new(system: &str) -> Result<Self, AgentError> {
        let client =
            Client::new().map_err(|err| AgentError::ClientInit(LlmError::from_genai(err)))?;
        Ok(Self::from_client(client, system))
    }

    /// Creates a new `Agent` that talks to an OpenAI-compatible endpoint
    /// (OpenAI, OpenRouter, LiteLLM, Ollama, ...).
    ///
    /// Every model name is sent to `base_url` with the OpenAI Chat Completions API,
    /// authenticated with `api_key`. Model names are passed as they are, so names like
    /// `openai/gpt-4.1-mini` or `vendor/model:free` work.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the API, for example `https://openrouter.ai/api/v1/`.
    /// * `api_key` - API key sent with every request.
    /// * `system` - The system prompt sent with every request.
    ///
    /// # Errors
    ///
    /// Returns an error when the LLM client cannot be created.
    pub fn new_with_url(base_url: &str, api_key: &str, system: &str) -> Result<Self, AgentError> {
        Self::new_with_provider(Provider::OpenAI, Some(base_url), api_key, system)
    }

    /// Creates a new `Agent` bound to one provider, with an explicit API key and, optionally, a
    /// custom endpoint.
    ///
    /// Every model name is sent to `provider` as it is. For example, with
    /// [`Provider::OpenRouter`] the model `nvidia/nemotron-3.5-super-vl:free` goes to OpenRouter.
    ///
    /// ```no_run
    /// use agentai::{Agent, Provider};
    ///
    /// # fn main() -> Result<(), agentai::AgentError> {
    /// let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    /// // OpenRouter with its default endpoint
    /// let agent = Agent::new_with_provider(Provider::OpenRouter, None, &api_key, "You are helpful")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `provider` - provider to use, for example [`Provider::OpenRouter`],
    ///   [`Provider::Anthropic`] or [`Provider::OpenAI`].
    /// * `base_url` - Endpoint override. `None` keeps the provider's default endpoint, for
    ///   example `https://openrouter.ai/api/v1/` for OpenRouter.
    /// * `api_key` - API key sent with every request.
    /// * `system` - The system prompt sent with every request.
    ///
    /// # Errors
    ///
    /// Returns an error when the LLM client cannot be created.
    pub fn new_with_provider(
        provider: Provider,
        base_url: Option<&str>,
        api_key: &str,
        system: &str,
    ) -> Result<Self, AgentError> {
        let adapter = provider.into_genai();
        let auth = AuthData::from_single(api_key);
        let builder = Client::builder().with_adapter_kind(adapter);
        let builder = match base_url {
            Some(base_url) => builder.append_provider_config(
                adapter,
                (Endpoint::from_owned(base_url.to_string()), auth),
            ),
            None => builder.append_provider_config(adapter, auth),
        };
        let client = builder
            .build()
            .map_err(|err| AgentError::ClientInit(LlmError::from_genai(err)))?;
        Ok(Self::from_client(client, system))
    }

    fn from_client(client: Client, system: &str) -> Self {
        Self {
            client,
            system: system.trim().to_string(),
            model: None,
            toolbox: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            chat_options: ChatOptions::default().with_temperature(DEFAULT_TEMPERATURE),
        }
    }

    /// Sets the model used by every run, unless the run sets its own with
    /// [`AgentRequest::with_model`]. Without a model, a run returns [`AgentError::MissingModel`].
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the toolbox used by every run, unless the run sets its own with
    /// [`AgentRequest::with_toolbox`]. Combine several toolboxes with
    /// [`ToolBoxSet`](crate::tool::ToolBoxSet).
    pub fn with_toolbox(self, toolbox: impl ToolBox) -> Self {
        self.with_toolbox_arc(Arc::new(toolbox))
    }

    /// Same as [`with_toolbox`](Agent::with_toolbox), for a toolbox you already share in an
    /// [`Arc`], for example between several agents.
    pub fn with_toolbox_arc(mut self, toolbox: Arc<dyn ToolBox>) -> Self {
        self.toolbox = Some(toolbox);
        self
    }

    /// Sets the maximum number of model calls in one run.
    ///
    /// Every answer with tool calls uses one iteration, so raise it for tasks that need many
    /// tool rounds. When the limit is reached without a final answer, the run returns
    /// [`AgentError::MaxIterationsReached`]. With `0`, the run returns that error without
    /// calling the model. The default is [`DEFAULT_MAX_ITERATIONS`].
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the [`ChatOptions`] sent with every request, for example temperature, max tokens,
    /// [tool choice](ToolChoice) or [reasoning effort](ReasoningEffort).
    ///
    /// The options replace the defaults, including the default temperature
    /// ([`DEFAULT_TEMPERATURE`]); set `with_temperature` yourself if you want to keep one.
    /// For structured output (`D` other than `String`), the run always sets the response
    /// format itself.
    pub fn with_chat_options(mut self, chat_options: ChatOptions) -> Self {
        self.chat_options = chat_options;
        self
    }

    /// Returns the system prompt sent with every request.
    pub fn system(&self) -> &str {
        &self.system
    }

    /// Returns the model set with [`with_model`](Agent::with_model).
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Returns the toolbox set with [`with_toolbox`](Agent::with_toolbox).
    pub fn toolbox(&self) -> Option<&Arc<dyn ToolBox>> {
        self.toolbox.as_ref()
    }

    /// Returns the maximum number of model calls in one run.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the [`ChatOptions`] sent with every request.
    pub fn chat_options(&self) -> &ChatOptions {
        &self.chat_options
    }

    /// Prepares a run: sends `prompt` to the model, runs the tools the model asks for and
    /// returns the final answer as `D`.
    ///
    /// Configure the returned [`AgentRequest`] with its `with_*` methods, then `.await` it to
    /// get a [`RunResult`]. [`run_stream`](Agent::run_stream) takes the same configuration and
    /// streams the run instead.
    ///
    /// ```no_run
    /// use agentai::{Agent, Message, RunResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let mut history: Vec<Message> = Vec::new();
    ///
    /// let result: RunResult<String> = agent.run("Why is the sky blue?").with_history(&history).await?;
    /// println!("{}", result.answer);
    /// history.extend(result.new_messages);
    ///
    /// let result: RunResult<String> = agent.run("And why is it red at sunset?").with_history(&history).await?;
    /// println!("{}", result.answer);
    /// history.extend(result.new_messages);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Awaiting the run returns an [`AgentError`]:
    /// - [`MissingModel`](AgentError::MissingModel) when no model is set;
    /// - [`Llm`](AgentError::Llm) when the request to the model fails (see
    ///   [`AgentError::is_transient`]);
    /// - [`Schema`](AgentError::Schema) when the schema of `D` cannot be generated;
    /// - [`InvalidAnswer`](AgentError::InvalidAnswer) when the answer does not match `D`;
    /// - [`MaxTokens`](AgentError::MaxTokens) when the answer was cut off by the token limit;
    /// - [`MaxIterationsReached`](AgentError::MaxIterationsReached) when there is no final answer;
    /// - [`Tool`](AgentError::Tool) when a tool fails with a fatal error. Recoverable tool errors
    ///   ([`ToolError::is_recoverable`]) are sent to the model instead.
    ///
    /// Your history is never changed, so after an error you can retry with the same history.
    ///
    /// ## Structured Output
    /// The answer type `D` decides between text and structured output: with `String` the model
    /// answers in plain text, with any other type the model is asked for JSON matching its schema.
    ///
    /// For more information go to [crate::structured_output]
    pub fn run<D>(&self, prompt: impl Into<String>) -> AgentRequest<'_, D, Complete> {
        AgentRequest::new(self, prompt.into())
    }

    /// Prepares a streamed run. It takes the same configuration as [`run`](Agent::run), but
    /// awaiting it returns an [`AgentStream`] of [`AgentEvent`]s: text and reasoning as the
    /// model writes them, tool calls and their results, and finally the [`RunResult`] in
    /// [`AgentEvent::Finished`].
    ///
    /// Awaiting checks the configuration (a missing model, the schema of `D`) before anything is
    /// sent; errors during the run are the last item of the stream. The stream owns everything
    /// it needs, so it is `Send + 'static`: move it into a spawned task or return it from a web
    /// handler.
    ///
    /// ```no_run
    /// use agentai::{Agent, AgentEvent, Message};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example(agent: Agent, mut history: Vec<Message>) -> Result<(), agentai::AgentError> {
    /// let mut stream = agent.run_stream::<String>("Tell me a story").with_history(&history).await?;
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         AgentEvent::TextDelta(text) => print!("{text}"),
    ///         AgentEvent::ToolCall(call) => println!("\n[calling {}]", call.name),
    ///         AgentEvent::Finished(result) => history.extend(result.new_messages),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_stream<D>(&self, prompt: impl Into<String>) -> AgentRequest<'_, D, Streaming> {
        AgentRequest::new(self, prompt.into())
    }
}
