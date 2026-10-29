//! Streaming a run: [`AgentEvent`] and [`AgentStream`].

use super::core::finish;
use super::{AgentError, AgentResult};
use crate::listener::{AsyncEventListener, EventListener, Printer, RunEvent};
use crate::message::ToolCall;
use crate::tool::ToolOutput;
use futures_util::{Stream, StreamExt};
use std::io::Write;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Something that happened during a streamed run, see [`Agent::run_stream`](super::Agent::run_stream).
///
/// The last event of a successful run is always [`Finished`](AgentEvent::Finished).
#[derive(Debug)]
#[non_exhaustive]
pub enum AgentEvent<D> {
    /// A piece of the model's text, in the order it arrives. Text the model writes before it
    /// asks for tools is streamed too, so a turn with tool calls can produce text deltas
    /// that are not part of the final answer. With structured output the deltas are pieces of
    /// the JSON answer.
    TextDelta(String),
    /// A piece of the model's reasoning, if the provider streams it.
    ReasoningDelta(String),
    /// The model asked for a tool call. Sent before the tool runs.
    ToolCall(ToolCall),
    /// A tool call finished, and its result will be sent to the model.
    ToolResult {
        /// Id of the tool call, the same as in [`ToolCall::id`].
        call_id: String,
        /// Name of the tool.
        tool: String,
        /// Result sent to the model: the tool's output, or the error message.
        content: ToolOutput,
        /// `true` when the tool failed. A recoverable error is sent to the model, which can
        /// try again; a fatal error ends the stream with [`AgentError::Tool`] after this event.
        is_error: bool,
    },
    /// The run finished. Holds the same [`AgentResult`] that awaiting the run returns.
    Finished(AgentResult<D>),
}

/// Stream of [`AgentEvent`]s of one run, returned by awaiting
/// [`Agent::run_stream`](super::Agent::run_stream).
///
/// It is `Send + 'static`: you can move it into a spawned task or return it from a web handler,
/// for example as server-sent events. An error ends the stream. Dropping the stream cancels the
/// run; your history is not changed by a cancelled or failed run.
///
/// Read it with [`StreamExt`](futures_util::StreamExt), pass it to an
/// [`EventListener`](crate::listener::EventListener) with [`listen`](AgentStream::listen), write it
/// to the console or any other writer with [`print_to`](AgentStream::print_to) (or
/// [`print_to_async`](AgentStream::print_to_async) / `print_to_tokio` for an async writer), or call
/// [`finish`](AgentStream::finish) to skip the remaining events and get the result.
#[must_use = "streams do nothing unless polled"]
pub struct AgentStream<D> {
    inner: Pin<Box<dyn Stream<Item = Result<AgentEvent<D>, AgentError>> + Send>>,
}

impl<D> AgentStream<D> {
    pub(super) fn new(
        inner: impl Stream<Item = Result<AgentEvent<D>, AgentError>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }

    /// Reads the remaining events and returns the result of the run.
    pub async fn finish(self) -> Result<AgentResult<D>, AgentError> {
        finish(self).await
    }

    /// Passes every event of the run to `listener` as it arrives and returns the result.
    ///
    /// See [`listener`](crate::listener) for the ready-made [`Printer`] and how to write your own
    /// listener, for example for a GUI. An error from the listener stops the run with
    /// [`AgentError::Listener`].
    ///
    /// ```no_run
    /// use agentai::listener::Printer;
    /// use agentai::{Agent, AgentResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let printer = Printer::new(std::io::stderr()).with_tool_results(true);
    /// let result: AgentResult<String> = agent.run_stream("What time is it?").await?.listen(printer).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn listen(
        mut self,
        mut listener: impl EventListener,
    ) -> Result<AgentResult<D>, AgentError> {
        while let Some(event) = self.inner.next().await {
            let event = event?;
            listener
                .on_event(&event.as_run_event())
                .map_err(AgentError::Listener)?;
            if let AgentEvent::Finished(result) = event {
                return Ok(result);
            }
        }
        unreachable!("the agent loop ends with a result or an error")
    }

    /// Same as [`listen`](AgentStream::listen), for an [`AsyncEventListener`].
    pub async fn listen_async(
        mut self,
        mut listener: impl AsyncEventListener,
    ) -> Result<AgentResult<D>, AgentError> {
        while let Some(event) = self.inner.next().await {
            let event = event?;
            listener
                .on_event(&event.as_run_event())
                .await
                .map_err(AgentError::Listener)?;
            if let AgentEvent::Finished(result) = event {
                return Ok(result);
            }
        }
        unreachable!("the agent loop ends with a result or an error")
    }

    /// Writes the run to `out` as it arrives and returns its result: the model's text as it is
    /// streamed, and each tool call on its own line.
    ///
    /// `out` is anything that implements [`std::io::Write`]: [`std::io::stdout()`],
    /// [`std::io::stderr()`], a [`File`](std::fs::File), a `Vec<u8>`, ... It is a shortcut for
    /// [`listen`](AgentStream::listen) with a [`Printer`] with the default settings; to change
    /// what is written, build the [`Printer`] yourself. The writes are blocking calls inside the
    /// async run, which is fine for stdout, stderr and files; for async writers, or writers that
    /// can block for long such as sockets and pipes, use
    /// [`print_to_async`](AgentStream::print_to_async) or
    /// [`print_to_tokio`](AgentStream::print_to_tokio).
    ///
    /// ```no_run
    /// use agentai::{Agent, AgentResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let result: AgentResult<String> = agent
    ///     .run_stream("Tell me a story")
    ///     .await?
    ///     .print_to(std::io::stdout())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of the run, and [`AgentError::Listener`] when writing to `out` fails.
    pub async fn print_to(self, out: impl Write + Send) -> Result<AgentResult<D>, AgentError> {
        self.listen(Printer::new(out)).await
    }

    /// Same as [`print_to`](AgentStream::print_to), for a writer that implements
    /// [`futures::io::AsyncWrite`](futures_util::io::AsyncWrite), the runtime-neutral async I/O
    /// trait used by smol and async-std, among others.
    pub async fn print_to_async(
        self,
        out: impl futures_util::io::AsyncWrite + Unpin + Send,
    ) -> Result<AgentResult<D>, AgentError> {
        self.listen_async(Printer::new(out)).await
    }

    /// Same as [`print_to`](AgentStream::print_to), for a tokio writer:
    /// `tokio::io::stdout()`, `tokio::io::stderr()`, `tokio::fs::File`, a TCP stream, ...
    ///
    /// ```no_run
    /// use agentai::{Agent, AgentResult};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// let result: AgentResult<String> = agent
    ///     .run_stream("Tell me a story")
    ///     .await?
    ///     .print_to_tokio(tokio::io::stdout())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "tokio")]
    pub async fn print_to_tokio(
        self,
        out: impl tokio::io::AsyncWrite + Unpin + Send,
    ) -> Result<AgentResult<D>, AgentError> {
        self.listen_async(Printer::tokio(out)).await
    }
}

impl<D> AgentEvent<D> {
    /// The event as a listener sees it: borrowed and without the typed answer.
    pub fn as_run_event(&self) -> RunEvent<'_> {
        match self {
            AgentEvent::TextDelta(text) => RunEvent::TextDelta(text),
            AgentEvent::ReasoningDelta(text) => RunEvent::ReasoningDelta(text),
            AgentEvent::ToolCall(call) => RunEvent::ToolCall(call),
            AgentEvent::ToolResult {
                call_id,
                tool,
                content,
                is_error,
            } => RunEvent::ToolResult {
                call_id,
                tool,
                content,
                is_error: *is_error,
            },
            AgentEvent::Finished(result) => RunEvent::Finished {
                new_messages: &result.new_messages,
                usage: &result.usage,
                stop_reason: result.stop_reason.as_ref(),
                iterations: result.iterations,
            },
        }
    }
}

impl<D> Stream for AgentStream<D> {
    type Item = Result<AgentEvent<D>, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
