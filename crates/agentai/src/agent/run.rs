//! A single run of an [`Agent`]: per-run options, the [`AgentRun`] builder and its result.

use super::core::{ResolvedRun, finish, run_loop};
use super::stream::AgentStream;
use super::{Agent, AgentError, ChatMessage, ChatOptions, StopReason, Usage};
use crate::tool::ToolBox;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

/// Settings that replace the [`Agent`] configuration for one run.
///
/// Like [`Agent`] and [`AgentRun`], it is configured with `with_*` methods.
///
/// Every field left as `None` keeps the agent's setting. Keep a `RunOptions` around to reuse
/// the same overrides, for example a cheaper model for simple questions, and pass it to
/// [`AgentRun::with_options`].
///
/// ```no_run
/// use agentai::{Agent, RunOptions, RunResult};
///
/// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
/// let cheap = RunOptions::default().with_model("gpt-4.1-mini").with_max_iterations(2);
///
/// let result: RunResult<String> = agent.run("Say hello").with_options(cheap.clone()).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
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
pub struct RunResult<D> {
    /// The final answer, parsed into `D`.
    pub answer: D,
    /// The messages added in this run, in order: your prompt, the model's tool calls, the tool
    /// results and the final answer. The history you passed and the system prompt are not
    /// included. Append them to your history to continue the conversation:
    /// `history.extend(result.new_messages)`.
    pub new_messages: Vec<ChatMessage>,
    /// Tokens used by all model calls of the run together.
    pub usage: Usage,
    /// Why the model stopped writing the final answer.
    pub stop_reason: Option<StopReason>,
    /// Reasoning the model returned with the final answer, if the provider sends it.
    pub reasoning_content: Option<String>,
    /// Number of model calls made in the run.
    pub iterations: usize,
}

/// A configured run of an [`Agent`], created by [`Agent::run`].
///
/// Set the history and per-run options with the builder methods, then `.await` it to get a
/// [`RunResult`], or call [`stream`](AgentRun::stream) to receive the answer as it is written.
/// Nothing is sent to the model before the run is awaited or the stream is polled.
#[must_use = "a run does nothing until it is awaited or streamed"]
pub struct AgentRun<'a, D = String> {
    agent: &'a Agent,
    prompt: String,
    history: &'a [ChatMessage],
    options: RunOptions,
    answer: PhantomData<fn() -> D>,
}

impl<'a, D> AgentRun<'a, D> {
    pub(super) fn new(agent: &'a Agent, prompt: String) -> Self {
        Self {
            agent,
            prompt,
            history: &[],
            options: RunOptions::default(),
            answer: PhantomData,
        }
    }

    /// Sets the conversation so far. The prompt is sent after these messages.
    ///
    /// The history is only read, never changed. Add [`RunResult::new_messages`] to it yourself
    /// once the run succeeds. Do not put the system prompt in it: the agent sends its own.
    pub fn with_history(mut self, history: &'a [ChatMessage]) -> Self {
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

    /// Streams the run: text and reasoning as the model writes them, tool calls and their
    /// results, and finally the [`RunResult`] in [`AgentEvent::Finished`](super::AgentEvent::Finished).
    ///
    /// Every model call of the loop is streamed, and the run works like an awaited run: tools
    /// are called, recoverable tool errors are sent to the model, and structured output is
    /// parsed at the end. The stream owns everything it needs, so it is `Send + 'static`.
    ///
    /// ```no_run
    /// use agentai::{Agent, AgentEvent, ChatMessage};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example(agent: Agent, mut history: Vec<ChatMessage>) -> Result<(), agentai::AgentError> {
    /// let mut stream = agent.run::<String>("Tell me a story").with_history(&history).stream();
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         AgentEvent::TextDelta(text) => print!("{text}"),
    ///         AgentEvent::ToolCall(call) => println!("\n[calling {}]", call.fn_name),
    ///         AgentEvent::Finished(result) => history.extend(result.new_messages),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Errors are the same as for an awaited run (see [`Agent::run`]); an error is the last item
    /// of the stream.
    pub fn stream(self) -> AgentStream<D>
    where
        D: DeserializeOwned + JsonSchema + Send + 'static,
    {
        match self.resolve() {
            Ok((run, history, prompt)) => {
                AgentStream::new(run_loop::<D>(run, history, prompt, true))
            }
            Err(err) => AgentStream::new(futures_util::stream::once(async move { Err(err) })),
        }
    }

    /// Combines the agent configuration with the overrides of this run.
    fn resolve(self) -> Result<(ResolvedRun, Vec<ChatMessage>, String), AgentError> {
        let agent = self.agent;
        let options = self.options;
        let model = options
            .model
            .or_else(|| agent.model.clone())
            .ok_or(AgentError::MissingModel)?;
        let run = ResolvedRun {
            client: agent.client.clone(),
            system: agent.system.clone(),
            model,
            toolbox: options.toolbox.or_else(|| agent.toolbox.clone()),
            max_iterations: options.max_iterations.unwrap_or(agent.max_iterations),
            chat_options: options
                .chat_options
                .unwrap_or_else(|| agent.chat_options.clone()),
        };
        Ok((run, self.history.to_vec(), self.prompt))
    }
}

impl<D> IntoFuture for AgentRun<'_, D>
where
    D: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Output = Result<RunResult<D>, AgentError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'static>>;

    fn into_future(self) -> Self::IntoFuture {
        let resolved = self.resolve();
        Box::pin(async move {
            let (run, history, prompt) = resolved?;
            finish(run_loop::<D>(run, history, prompt, false)).await
        })
    }
}
