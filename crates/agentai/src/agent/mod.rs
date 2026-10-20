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
/// A message of the conversation history, re-exported from genai. The history is yours: pass it
/// to [`AgentRun::history`] and append [`RunResult::new_messages`] to it after a run.
pub use genai::chat::ChatMessage;
/// Model options (temperature, max tokens, tool choice, reasoning effort, ...) used by
/// [`Agent::with_chat_options`], re-exported from genai.
pub use genai::chat::ChatOptions;
/// Why the model stopped, re-exported from genai. See [`RunResult::stop_reason`].
pub use genai::chat::StopReason;
/// A tool call requested by the model, re-exported from genai.
pub use genai::chat::ToolCall;
/// Token usage, re-exported from genai. See [`RunResult::usage`].
pub use genai::chat::Usage;
use genai::resolver::{AuthData, Endpoint};
use std::sync::Arc;
use thiserror::Error;

mod core;
mod run;
mod stream;

pub use run::{AgentRun, RunOptions, RunResult};
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
    /// The final answer was cut off by the token limit (for example `max_tokens` in
    /// [`ChatOptions`]). Holds the truncated text. Raise the limit or ask for a shorter answer.
    #[error("the model answer was truncated by the token limit")]
    MaxTokens {
        /// Truncated text returned by the model.
        raw: String,
    },
    /// No model was set, neither with [`Agent::with_model`] nor for the run
    /// ([`AgentRun::model`]).
    #[error("no model set: use Agent::with_model or AgentRun::model")]
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

/// Default number of model calls in one [`Agent::run`].
pub const DEFAULT_MAX_ITERATIONS: usize = 5;

/// Default temperature sent with every request, unless [`Agent::with_chat_options`] replaces it.
pub const DEFAULT_TEMPERATURE: f64 = 0.2;

/// The `Agent` struct represents an agent that interacts with a chat model.
///
/// An agent is configuration only: the genai client, the system prompt, the default model and
/// toolbox, how many model calls one [`run`](Agent::run) may make, and the [`ChatOptions`] sent
/// with every request. It does not keep the conversation. You own the history, pass it to
/// [`AgentRun::history`] and append [`RunResult::new_messages`] to it, so one agent can serve
/// many conversations at once. Cloning an agent is cheap.
///
/// ```no_run
/// use agentai::Agent;
/// use agentai::genai::chat::{ChatOptions, ReasoningEffort};
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
    /// Reference to GenAI Client
    client: Client,

    /// System prompt sent with every request. It is not part of the history.
    system: String,

    /// Model used when the run does not set one
    model: Option<String>,

    /// Toolbox used when the run does not set one
    toolbox: Option<Arc<dyn ToolBox>>,

    /// Maximum number of model calls in one `run`
    max_iterations: usize,

    /// Options sent with every request
    chat_options: ChatOptions,
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
    /// * `system` - The system prompt sent with every request.
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
    /// * `system` - The system prompt sent with every request.
    ///
    /// # Returns
    ///
    /// A new `Agent` instance.
    pub fn new_with_client(client: Client, system: &str) -> Self {
        Self {
            client,
            system: system.trim().to_string(),
            model: None,
            toolbox: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            chat_options: ChatOptions::default().with_temperature(DEFAULT_TEMPERATURE),
        }
    }

    /// Sets the model used by every [`run`](Agent::run), unless the run sets its own with
    /// [`AgentRun::model`]. Without a model, a run returns [`AgentError::MissingModel`].
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the toolbox used by every [`run`](Agent::run), unless the run sets its own with
    /// [`AgentRun::toolbox`]. Combine several toolboxes with
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

    /// Sets the maximum number of model calls in one [`run`](Agent::run).
    ///
    /// Every answer with tool calls uses one iteration, so raise it for tasks that need many
    /// tool rounds. When the limit is reached without a final answer, `run` returns
    /// [`AgentError::MaxIterationsReached`]. With `0`, `run` returns that error without
    /// calling the model. The default is [`DEFAULT_MAX_ITERATIONS`].
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the [`ChatOptions`] sent with every request, for example temperature, max tokens,
    /// [tool choice](genai::chat::ToolChoice) or [reasoning effort](genai::chat::ReasoningEffort).
    ///
    /// The options replace the defaults, including the default temperature
    /// ([`DEFAULT_TEMPERATURE`]); set `with_temperature` yourself if you want to keep one.
    /// For structured output (`run::<D>` with `D` other than `String`), `run` always sets the
    /// response format, so a response format given here is replaced.
    ///
    /// Options set on a custom genai client passed to [`new_with_client`](Agent::new_with_client)
    /// still apply. genai uses them for every field these options leave unset.
    pub fn with_chat_options(mut self, chat_options: ChatOptions) -> Self {
        self.chat_options = chat_options;
        self
    }

    /// Returns the maximum number of model calls in one [`run`](Agent::run).
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the [`ChatOptions`] sent with every request.
    pub fn chat_options(&self) -> &ChatOptions {
        &self.chat_options
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

    /// Starts a run: sends `prompt` to the model, runs the tools the model asks for and
    /// returns the final answer as `D`.
    ///
    /// The returned [`AgentRun`] is configured with builder methods and executed with
    /// `.await`, or streamed with [`AgentRun::stream`]. Pass the conversation so far with [`AgentRun::history`], and change the model,
    /// toolbox or options for this run only with [`AgentRun::model`], [`AgentRun::toolbox`] or
    /// [`AgentRun::with_options`].
    ///
    /// ```no_run
    /// use agentai::{Agent, ChatMessage, RunResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let mut history: Vec<ChatMessage> = Vec::new();
    ///
    /// let result: RunResult<String> = agent.run("Why is the sky blue?").history(&history).await?;
    /// println!("{}", result.answer);
    /// history.extend(result.new_messages);
    ///
    /// let result: RunResult<String> = agent.run("And why is it red at sunset?").history(&history).await?;
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
    pub fn run<D>(&self, prompt: impl Into<String>) -> AgentRun<'_, D> {
        AgentRun::new(self, prompt.into())
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
    /// * `system` - The system prompt sent with every request.
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
    /// * `system` - The system prompt sent with every request.
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
}
