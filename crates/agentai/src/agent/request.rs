//! A single run of an [`Agent`]: the [`AgentRequest`] builder, per-run options and the result.

use super::core::{ResolvedRun, finish, response_format, run_loop};
use super::stream::AgentStream;
use super::{Agent, AgentError, ChatOptions, StopReason, Usage};
use crate::message::Message;
use crate::tool::ToolBox;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::any::TypeId;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

/// Settings that replace the [`Agent`] configuration for one run.
///
/// Every field left as `None` keeps the agent's setting. Keep a `RunOptions` around to reuse
/// the same overrides, for example a cheaper model for simple questions, and pass it to
/// [`AgentRequest::with_options`].
///
/// ```no_run
/// use agentai::{Agent, RunOptions, AgentResult};
///
/// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
/// let cheap = RunOptions::default().with_model("gpt-4.1-mini").with_max_iterations(2);
///
/// let result: AgentResult<String> = agent.run("Say hello").with_options(cheap.clone()).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct RunOptions {
    /// Model to use instead of [`Agent::with_model`].
    pub model: Option<String>,
    /// Toolbox to use instead of [`Agent::with_toolbox`].
    pub toolbox: Option<Arc<dyn ToolBox>>,
    /// Iteration limit to use instead of [`Agent::with_max_iterations`].
    pub max_iterations: Option<usize>,
    /// Options to send instead of [`Agent::with_chat_options`]. They replace the agent's
    /// options as a whole, they are not merged with them.
    pub chat_options: Option<ChatOptions>,
}

impl RunOptions {
    /// Sets the model for the run.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the toolbox for the run.
    pub fn with_toolbox(self, toolbox: impl ToolBox) -> Self {
        self.with_toolbox_arc(Arc::new(toolbox))
    }

    /// Sets the toolbox for the run, for a toolbox you already share in an [`Arc`].
    pub fn with_toolbox_arc(mut self, toolbox: Arc<dyn ToolBox>) -> Self {
        self.toolbox = Some(toolbox);
        self
    }

    /// Sets the maximum number of model calls in the run.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Sets the [`ChatOptions`] for the run.
    pub fn with_chat_options(mut self, chat_options: ChatOptions) -> Self {
        self.chat_options = Some(chat_options);
        self
    }

    /// Returns these options with every field set in `other` replaced by its value.
    fn merge(self, other: RunOptions) -> Self {
        Self {
            model: other.model.or(self.model),
            toolbox: other.toolbox.or(self.toolbox),
            max_iterations: other.max_iterations.or(self.max_iterations),
            chat_options: other.chat_options.or(self.chat_options),
        }
    }
}

/// The result of a successful run.
#[derive(Debug)]
#[non_exhaustive]
pub struct AgentResult<D> {
    /// The final answer, parsed into `D`.
    pub answer: D,
    /// The messages added in this run, in order: your prompt, the model's tool calls, the tool
    /// results and the final answer. The history you passed and the system prompt are not
    /// included. Append them to your history to continue the conversation:
    /// `history.extend(result.new_messages)`.
    pub new_messages: Vec<Message>,
    /// Tokens used by all model calls of the run together.
    pub usage: Usage,
    /// Why the model stopped writing the final answer.
    pub stop_reason: Option<StopReason>,
    /// Reasoning the model returned with the final answer, if the provider sends it.
    pub reasoning_content: Option<String>,
    /// Number of model calls made in the run.
    pub iterations: usize,
}

/// Mode of an [`AgentRequest`] created by [`Agent::run`]: awaiting it returns a [`AgentResult`].
#[derive(Debug, Clone, Copy)]
pub struct Complete;

/// Mode of an [`AgentRequest`] created by [`Agent::run_stream`]: awaiting it returns an
/// [`AgentStream`].
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

/// A prepared run of an [`Agent`], created by [`Agent::run`] or [`Agent::run_stream`].
///
/// Both create the same request with the same `with_*` methods; only the result of `.await`
/// differs. The mode `M` is [`Complete`] for `run` (a [`AgentResult`]) and [`Streaming`] for
/// `run_stream` (an [`AgentStream`]). Nothing is sent to the model before the request is
/// awaited.
#[must_use = "a request does nothing until it is awaited"]
pub struct AgentRequest<'a, D = String, M = Complete> {
    agent: &'a Agent,
    prompt: String,
    history: &'a [Message],
    options: RunOptions,
    marker: PhantomData<fn() -> (D, M)>,
}

impl<'a, D, M> AgentRequest<'a, D, M> {
    pub(super) fn new(agent: &'a Agent, prompt: String) -> Self {
        Self {
            agent,
            prompt,
            history: &[],
            options: RunOptions::default(),
            marker: PhantomData,
        }
    }

    /// Sets the conversation so far. The prompt is sent after these messages.
    ///
    /// The history is only read, never changed. Add [`AgentResult::new_messages`] to it yourself
    /// once the run succeeds. The system prompt is not part of it: the agent sends its own.
    pub fn with_history(mut self, history: &'a [Message]) -> Self {
        self.history = history;
        self
    }

    /// Uses this model instead of the agent's model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.options.model = Some(model.into());
        self
    }

    /// Uses this toolbox instead of the agent's toolbox.
    pub fn with_toolbox(self, toolbox: impl ToolBox) -> Self {
        self.with_toolbox_arc(Arc::new(toolbox))
    }

    /// Uses this toolbox, shared in an [`Arc`], instead of the agent's toolbox.
    pub fn with_toolbox_arc(mut self, toolbox: Arc<dyn ToolBox>) -> Self {
        self.options.toolbox = Some(toolbox);
        self
    }

    /// Uses this iteration limit instead of the agent's limit.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.options.max_iterations = Some(max_iterations);
        self
    }

    /// Sends these [`ChatOptions`] instead of the agent's options.
    pub fn with_chat_options(mut self, chat_options: ChatOptions) -> Self {
        self.options.chat_options = Some(chat_options);
        self
    }

    /// Applies a set of overrides. Fields set in `options` replace the ones set before.
    pub fn with_options(mut self, options: RunOptions) -> Self {
        self.options = self.options.merge(options);
        self
    }

    /// Combines the agent configuration with the overrides of this run and checks it.
    fn resolve(self) -> Result<(ResolvedRun, Vec<Message>, String), AgentError>
    where
        D: JsonSchema + 'static,
    {
        let agent = self.agent;
        let options = self.options;
        let model = options
            .model
            .or_else(|| agent.model.clone())
            .ok_or(AgentError::MissingModel)?;
        let is_answer_string = TypeId::of::<String>() == TypeId::of::<D>();
        let run = ResolvedRun {
            client: agent.client.clone(),
            system: agent.system.clone(),
            model,
            toolbox: options.toolbox.or_else(|| agent.toolbox.clone()),
            max_iterations: options.max_iterations.unwrap_or(agent.max_iterations),
            chat_options: options
                .chat_options
                .unwrap_or_else(|| agent.chat_options.clone()),
            response_format: if is_answer_string {
                None
            } else {
                Some(response_format::<D>()?)
            },
        };
        Ok((run, self.history.to_vec(), self.prompt))
    }
}

impl<D> IntoFuture for AgentRequest<'_, D, Complete>
where
    D: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Output = Result<AgentResult<D>, AgentError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'static>>;

    fn into_future(self) -> Self::IntoFuture {
        let resolved = self.resolve();
        Box::pin(async move {
            let (run, history, prompt) = resolved?;
            finish(run_loop::<D>(run, history, prompt, false)).await
        })
    }
}

impl<D> IntoFuture for AgentRequest<'_, D, Streaming>
where
    D: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Output = Result<AgentStream<D>, AgentError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'static>>;

    fn into_future(self) -> Self::IntoFuture {
        let resolved = self.resolve();
        Box::pin(async move {
            let (run, history, prompt) = resolved?;
            Ok(AgentStream::new(run_loop::<D>(run, history, prompt, true)))
        })
    }
}
