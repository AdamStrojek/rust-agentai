//! # Core components of AI Agents
//!
//! This module contains core components that you can use in your AI Agents
//!
//! To read more about Agents look into [crate::agent::Agent]
//!
//! To read more about structured output look into [crate::structured_output]
//!
//! To read more about tool look into [crate::tool]

use crate::prompt::IntoPrompt;
use crate::tool::{ToolBox, ToolError};
use genai::Client;
use std::sync::Arc;
use thiserror::Error;

mod core;
mod model;
mod request;
mod stream;

pub use model::{ChatOptions, ReasoningEffort, StopReason, ToolChoice, Usage};
pub use request::{AgentRequest, AgentResult, Complete, RunOptions, Streaming};
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
    /// An answer of the model was cut off by the token limit (for example `max_tokens` in
    /// [`ChatOptions`]). Holds the truncated text, which is empty when the cut-off answer had only
    /// reasoning or tool calls (the tools are not run). Raise the limit or ask for a shorter answer.
    #[error("the model answer was truncated by the token limit")]
    MaxTokens {
        /// Truncated text returned by the model.
        raw: String,
    },
    /// The agent was created without a model ([`Agent::default`]), and neither
    /// [`Agent::with_model`] nor the run ([`AgentRequest::with_model`]) set one.
    #[error("no model set: use Agent::with_model or AgentRequest::with_model")]
    MissingModel,
    /// An [event listener](crate::listener) failed, for example writing the stream with
    /// [`AgentStream::print_to`] failed.
    #[error("event listener failed: {0}")]
    Listener(#[source] crate::listener::ListenerError),
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

/// The `Agent` struct represents an agent that interacts with a chat model.
///
/// An agent is configuration only: the LLM client, the system prompt, the default model and
/// toolbox, how many model calls one run may make, and the [`ChatOptions`] sent with every
/// request. It does not keep the conversation. You own the history, pass it to
/// [`AgentRequest::with_history`] and append [`AgentResult::new_messages`] to it, so one agent can
/// serve many conversations at once. Cloning an agent is cheap.
///
/// ```no_run
/// use agentai::{Agent, ChatOptions, ReasoningEffort};
///
/// # fn main() -> Result<(), agentai::AgentError> {
/// let agent = Agent::new("gpt-4o", "You are a useful assistant")?
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
    /// Creates a new `Agent` for `model`, picking the provider from the model name.
    ///
    /// `gpt-*` models go to OpenAI, `claude-*` to Anthropic, `gemini-*` to Gemini, and so on; a
    /// `provider::model` prefix such as `open_router::vendor/model` selects a provider
    /// explicitly. The API key is read from the provider's usual environment variable, for
    /// example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `OPEN_ROUTER_API_KEY`. For an endpoint the
    /// model name cannot select, such as your own OpenAI-compatible server, use
    /// [`from_client`](Agent::from_client).
    ///
    /// ```no_run
    /// use agentai::Agent;
    ///
    /// # fn main() -> Result<(), agentai::AgentError> {
    /// let agent = Agent::new("claude-haiku-5-5", "You are a useful assistant")?;
    /// // OpenRouter, with the key from `OPEN_ROUTER_API_KEY`
    /// let agent = Agent::new("open_router::nvidia/nemotron-3.5-super-vl:free", "You are a useful assistant")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `model` - The model used by every run, unless the run sets its own with
    ///   [`AgentRequest::with_model`].
    /// * `system` - The system prompt sent with every request. Strongly recommended, but it may be
    ///   empty, and then no system prompt is sent.
    ///
    /// # Errors
    ///
    /// Returns an error when the LLM client cannot be created.
    pub fn new(model: &str, system: &str) -> Result<Self, AgentError> {
        let client =
            Client::new().map_err(|err| AgentError::ClientInit(LlmError::from_genai(err)))?;
        Ok(Self::from_parts(client, Some(model.to_string()), system))
    }

    /// Creates a new `Agent` for `model` that sends its requests through your own
    /// [`genai`] client, for full control over providers, endpoints, API keys and HTTP settings.
    ///
    /// Use it when [`new`](Agent::new) is not enough, for example for an OpenAI-compatible
    /// server (LiteLLM, vLLM, a local Ollama, ...). Build the client with [`crate::genai`], the
    /// version `agentai` uses:
    ///
    /// ```no_run
    /// use agentai::Agent;
    /// use agentai::genai::Client;
    /// use agentai::genai::adapter::AdapterKind;
    /// use agentai::genai::resolver::{AuthData, Endpoint};
    ///
    /// let client = Client::builder()
    ///     // Every model name goes to this endpoint with the OpenAI Chat Completions API
    ///     .with_adapter_kind(AdapterKind::OpenAI)
    ///     .append_provider_config(
    ///         AdapterKind::OpenAI,
    ///         (
    ///             Endpoint::from_static("http://localhost:4000/v1/"),
    ///             AuthData::from_single("my-key"),
    ///         ),
    ///     )
    ///     .build()
    ///     .expect("LLM client");
    /// let agent = Agent::from_client(client, "openai/gpt-4.1-mini", "You are a useful assistant");
    /// ```
    ///
    /// # Arguments
    ///
    /// * `client` - The genai client every request is sent with.
    /// * `model` - The model used by every run, unless the run sets its own with
    ///   [`AgentRequest::with_model`].
    /// * `system` - The system prompt sent with every request. Strongly recommended, but it may be
    ///   empty, and then no system prompt is sent.
    pub fn from_client(client: Client, model: &str, system: &str) -> Self {
        Self::from_parts(client, Some(model.to_string()), system)
    }

    fn from_parts(client: Client, model: Option<String>, system: &str) -> Self {
        Self {
            client,
            system: system.trim().to_string(),
            model,
            toolbox: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            chat_options: ChatOptions::default(),
        }
    }

    /// Sets the model used by every run, unless the run sets its own with
    /// [`AgentRequest::with_model`]. An agent created with [`Agent::default`] has no model, and
    /// a run without one returns [`AgentError::MissingModel`].
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the system prompt sent with every request. An empty system prompt is not sent.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into().trim().to_string();
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
    /// For structured output (`D` other than `String`), the run always sets the response
    /// format itself.
    pub fn with_chat_options(mut self, chat_options: ChatOptions) -> Self {
        self.chat_options = chat_options;
        self
    }

    /// Returns the system prompt sent with every request. It is empty when none is sent.
    pub fn system(&self) -> &str {
        &self.system
    }

    /// Returns the model used by every run. Only an agent created with [`Agent::default`] may have
    /// none.
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
    /// The prompt is text (`&str`, `String`) or a [`Prompt`](crate::Prompt) with images or
    /// other media for vision models; see [`IntoPrompt`].
    ///
    /// Configure the returned [`AgentRequest`] with its `with_*` methods, then `.await` it to
    /// get a [`AgentResult`]. [`run_stream`](Agent::run_stream) takes the same configuration and
    /// streams the run instead.
    ///
    /// ```no_run
    /// use agentai::{Agent, Message, AgentResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let mut history: Vec<Message> = Vec::new();
    ///
    /// let result: AgentResult<String> = agent.run("Why is the sky blue?").with_history(&history).await?;
    /// println!("{}", result.answer);
    /// history.extend(result.new_messages);
    ///
    /// let result: AgentResult<String> = agent.run("And why is it red at sunset?").with_history(&history).await?;
    /// println!("{}", result.answer);
    /// history.extend(result.new_messages);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Awaiting the run returns an [`AgentError`]:
    /// - [`MissingModel`](AgentError::MissingModel) when neither the agent nor the run has a
    ///   model;
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
    pub fn run<D>(&self, prompt: impl IntoPrompt) -> AgentRequest<'_, D, Complete> {
        AgentRequest::new(self, prompt.into_prompt())
    }

    /// Prepares a streamed run. It takes the same configuration as [`run`](Agent::run), but
    /// awaiting it returns an [`AgentStream`] of [`AgentEvent`]s: text and reasoning as the
    /// model writes them, tool calls and their results, and finally the [`AgentResult`] in
    /// [`AgentEvent::Finished`].
    ///
    /// Awaiting checks the configuration (a missing model, the schema of `D`) before anything is
    /// sent; errors during the run are the last item of the stream. The stream owns everything
    /// it needs, so it is `Send + 'static`: move it into a spawned task or return it from a web
    /// handler. Read it with `StreamExt` from the `futures-util` (or `futures`) crate, which you
    /// add to your own dependencies.
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
    pub fn run_stream<D>(&self, prompt: impl IntoPrompt) -> AgentRequest<'_, D, Streaming> {
        AgentRequest::new(self, prompt.into_prompt())
    }
}

impl Default for Agent {
    /// Creates an `Agent` without a model and without a system prompt. The provider is picked
    /// from the model name, as in [`Agent::new`].
    ///
    /// Set the model with [`with_model`](Agent::with_model) or on each run
    /// ([`AgentRequest::with_model`]); otherwise a run returns [`AgentError::MissingModel`]. A
    /// system prompt is strongly recommended: set it with [`with_system`](Agent::with_system).
    ///
    /// ```no_run
    /// use agentai::Agent;
    ///
    /// let agent = Agent::default()
    ///     .with_model("gpt-4o")
    ///     .with_system("You are a useful assistant");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the LLM client cannot be created. [`Agent::new`] returns that error instead.
    fn default() -> Self {
        let client = Client::new().expect("failed to create the LLM client");
        Self::from_parts(client, None, "")
    }
}
