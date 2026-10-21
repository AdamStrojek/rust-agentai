//! Streaming a run: [`AgentEvent`] and [`AgentStream`].

use super::core::finish;
use super::{AgentError, RunResult};
use crate::message::ToolCall;
use futures_util::Stream;
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
        content: String,
        /// `true` when the tool failed. A recoverable error is sent to the model, which can
        /// try again; a fatal error ends the stream with [`AgentError::Tool`] after this event.
        is_error: bool,
    },
    /// The run finished. Holds the same [`RunResult`] that awaiting the run returns.
    Finished(RunResult<D>),
}

/// Stream of [`AgentEvent`]s of one run, returned by awaiting
/// [`Agent::run_stream`](super::Agent::run_stream).
///
/// It is `Send + 'static`: you can move it into a spawned task or return it from a web handler,
/// for example as server-sent events. An error ends the stream. Dropping the stream cancels the
/// run; your history is not changed by a cancelled or failed run.
///
/// Read it with [`StreamExt`](futures_util::StreamExt), or call [`finish`](AgentStream::finish)
/// to skip the remaining events and get the result.
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
    pub async fn finish(self) -> Result<RunResult<D>, AgentError> {
        finish(self).await
    }
}

impl<D> Stream for AgentStream<D> {
    type Item = Result<AgentEvent<D>, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
