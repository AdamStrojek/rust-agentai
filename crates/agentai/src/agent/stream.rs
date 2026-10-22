//! Streaming a run: [`AgentEvent`] and [`AgentStream`].

use super::core::finish;
use super::{AgentError, AgentResult};
use crate::message::ToolCall;
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
        content: String,
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
/// Read it with [`StreamExt`](futures_util::StreamExt), call [`print_to`](AgentStream::print_to)
/// (or [`print_to_async`](AgentStream::print_to_async) / `print_to_tokio` for an async writer) to
/// write the answer to the console or any other writer as it arrives, or call
/// [`finish`](AgentStream::finish) to skip the remaining events and get the result.
#[must_use = "streams do nothing unless polled"]
pub struct AgentStream<D> {
    inner: Pin<Box<dyn Stream<Item = Result<AgentEvent<D>, AgentError>> + Send>>,
    print_options: PrintOptions,
}

impl<D> AgentStream<D> {
    pub(super) fn new(
        inner: impl Stream<Item = Result<AgentEvent<D>, AgentError>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            print_options: PrintOptions::default(),
        }
    }

    /// Reads the remaining events and returns the result of the run.
    pub async fn finish(self) -> Result<AgentResult<D>, AgentError> {
        finish(self).await
    }

    /// Sets what [`print_to`](AgentStream::print_to), [`print_to_async`](AgentStream::print_to_async)
    /// and `print_to_tokio` write besides the model's text.
    ///
    /// ```no_run
    /// use agentai::{Agent, PrintOptions};
    ///
    /// # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
    /// agent
    ///     .run_stream::<String>("What time is it?")
    ///     .await?
    ///     .with_print_options(PrintOptions::default().with_tool_results(true))
    ///     .print_to(std::io::stderr())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_print_options(mut self, options: PrintOptions) -> Self {
        self.print_options = options;
        self
    }

    /// Writes the run to `out` as it arrives and returns its result.
    ///
    /// The model's text is written as it is streamed, and each tool call is written on its own
    /// line ([`with_print_options`](AgentStream::with_print_options) chooses what else is written).
    /// `out` is anything that implements [`std::io::Write`]: [`std::io::stdout()`],
    /// [`std::io::stderr()`], a [`File`](std::fs::File), a `Vec<u8>`, ... The writer is flushed
    /// after every piece, so the text shows up right away. For an asynchronous writer use
    /// [`print_to_async`](AgentStream::print_to_async) or `print_to_tokio`.
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
    /// The errors of the run, and [`AgentError::Output`] when writing to `out` fails.
    pub async fn print_to(mut self, mut out: impl Write) -> Result<AgentResult<D>, AgentError> {
        let mut printer = Printer::default();
        while let Some(event) = self.inner.next().await {
            let (text, result) = printer.render(event?, &self.print_options);
            if !text.is_empty() {
                out.write_all(text.as_bytes())
                    .and_then(|()| out.flush())
                    .map_err(AgentError::Output)?;
            }
            if let Some(result) = result {
                return Ok(result);
            }
        }
        unreachable!("the agent loop ends with a result or an error")
    }

    /// Same as [`print_to`](AgentStream::print_to), for an asynchronous writer that implements
    /// [`futures::io::AsyncWrite`](futures_util::io::AsyncWrite).
    ///
    /// This is the runtime-neutral async I/O trait used by smol and async-std, among others. For a
    /// tokio writer (`tokio::io::stdout()`, `tokio::fs::File`, a TCP stream), use `print_to_tokio`
    /// (feature `tokio`), or wrap it with `tokio_util::compat`.
    ///
    /// # Errors
    ///
    /// The errors of the run, and [`AgentError::Output`] when writing to `out` fails.
    pub async fn print_to_async(
        mut self,
        mut out: impl futures_util::io::AsyncWrite + Unpin,
    ) -> Result<AgentResult<D>, AgentError> {
        use futures_util::io::AsyncWriteExt;

        let mut printer = Printer::default();
        while let Some(event) = self.inner.next().await {
            let (text, result) = printer.render(event?, &self.print_options);
            if !text.is_empty() {
                out.write_all(text.as_bytes())
                    .await
                    .map_err(AgentError::Output)?;
                out.flush().await.map_err(AgentError::Output)?;
            }
            if let Some(result) = result {
                return Ok(result);
            }
        }
        unreachable!("the agent loop ends with a result or an error")
    }

    /// Same as [`print_to`](AgentStream::print_to), for a tokio writer that implements
    /// [`tokio::io::AsyncWrite`]: `tokio::io::stdout()`, `tokio::io::stderr()`,
    /// `tokio::fs::File`, a TCP stream, ...
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
    ///
    /// # Errors
    ///
    /// The errors of the run, and [`AgentError::Output`] when writing to `out` fails.
    #[cfg(feature = "tokio")]
    pub async fn print_to_tokio(
        mut self,
        mut out: impl tokio::io::AsyncWrite + Unpin,
    ) -> Result<AgentResult<D>, AgentError> {
        use tokio::io::AsyncWriteExt;

        let mut printer = Printer::default();
        while let Some(event) = self.inner.next().await {
            let (text, result) = printer.render(event?, &self.print_options);
            if !text.is_empty() {
                out.write_all(text.as_bytes())
                    .await
                    .map_err(AgentError::Output)?;
                out.flush().await.map_err(AgentError::Output)?;
            }
            if let Some(result) = result {
                return Ok(result);
            }
        }
        unreachable!("the agent loop ends with a result or an error")
    }
}

/// What [`AgentStream::print_to`] and the other `print_to_*` methods write besides the model's
/// text. Set it with [`AgentStream::with_print_options`].
///
/// ```
/// use agentai::PrintOptions;
///
/// let options = PrintOptions::default().with_tool_results(true).with_reasoning(true);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PrintOptions {
    /// Write each tool call with its arguments. On by default.
    pub tool_calls: bool,
    /// Write each tool result. Off by default.
    pub tool_results: bool,
    /// Write the model's reasoning, if the provider streams it. Off by default.
    pub reasoning: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            tool_calls: true,
            tool_results: false,
            reasoning: false,
        }
    }
}

impl PrintOptions {
    /// Sets whether tool calls are written.
    pub fn with_tool_calls(mut self, tool_calls: bool) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Sets whether tool results are written.
    pub fn with_tool_results(mut self, tool_results: bool) -> Self {
        self.tool_results = tool_results;
        self
    }

    /// Sets whether the model's reasoning is written.
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }
}

/// What was written last, to put tool calls and reasoning on lines of their own.
#[derive(Default, PartialEq)]
enum Printed {
    /// Nothing yet, or a complete line.
    #[default]
    LineStart,
    Text,
    Reasoning,
}

/// Turns stream events into the text that the `print_to*` methods write.
#[derive(Default)]
struct Printer {
    last: Printed,
}

impl Printer {
    /// Returns the text to write for `event`, and the result when the run finished.
    fn render<D>(
        &mut self,
        event: AgentEvent<D>,
        options: &PrintOptions,
    ) -> (String, Option<AgentResult<D>>) {
        let mut out = String::new();
        match event {
            AgentEvent::TextDelta(text) => {
                if self.last == Printed::Reasoning {
                    self.end_line(&mut out);
                }
                self.push_text(&mut out, &text, Printed::Text);
            }
            AgentEvent::ReasoningDelta(text) if options.reasoning => {
                if self.last != Printed::Reasoning {
                    self.end_line(&mut out);
                    out.push_str("[reasoning] ");
                }
                self.push_text(&mut out, &text, Printed::Reasoning);
            }
            AgentEvent::ToolCall(call) if options.tool_calls => {
                self.end_line(&mut out);
                out.push_str(&format!("[tool call] {}({})\n", call.name, call.arguments));
            }
            AgentEvent::ToolResult {
                tool,
                content,
                is_error,
                ..
            } if options.tool_results => {
                self.end_line(&mut out);
                let label = if is_error {
                    "tool error"
                } else {
                    "tool result"
                };
                out.push_str(&format!("[{label}] {tool}: {content}\n"));
            }
            AgentEvent::Finished(result) => {
                self.end_line(&mut out);
                return (out, Some(result));
            }
            _ => {}
        }
        (out, None)
    }

    fn push_text(&mut self, out: &mut String, text: &str, kind: Printed) {
        if text.is_empty() {
            return;
        }
        out.push_str(text);
        self.last = if text.ends_with('\n') {
            Printed::LineStart
        } else {
            kind
        };
    }

    /// Ends the current line, if one is open.
    fn end_line(&mut self, out: &mut String) {
        if self.last != Printed::LineStart {
            out.push('\n');
            self.last = Printed::LineStart;
        }
    }
}

impl<D> Stream for AgentStream<D> {
    type Item = Result<AgentEvent<D>, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
