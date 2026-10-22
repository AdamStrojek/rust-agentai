//! # Event listeners
//!
//! An [`EventListener`] (or [`AsyncEventListener`]) receives every event of a streamed run as it
//! happens: text and reasoning as the model writes them, tool calls and their results, and the
//! end of the run. Pass one to [`AgentStream::listen`](crate::AgentStream::listen) or
//! [`AgentStream::listen_async`](crate::AgentStream::listen_async).
//!
//! [`Printer`] is the ready-made listener that writes a run to the console, a file or any other
//! writer. [`AgentStream::print_to`](crate::AgentStream::print_to) and its async variants create
//! one with the default settings; build your own to change what is written:
//!
//! ```no_run
//! use agentai::listener::Printer;
//! use agentai::{Agent, AgentResult};
//!
//! # async fn example(agent: Agent) -> Result<(), agentai::AgentError> {
//! let printer = Printer::new(std::io::stderr()).with_tool_results(true);
//! let result: AgentResult<String> = agent.run_stream("What time is it?").await?.listen(printer).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Implement the traits yourself to show a run somewhere else, for example in a GUI:
//!
//! ```
//! use agentai::listener::{EventListener, ListenerError, RunEvent};
//!
//! #[derive(Default)]
//! struct Transcript {
//!     text: String,
//! }
//!
//! impl EventListener for Transcript {
//!     fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
//!         if let RunEvent::TextDelta(text) = event {
//!             self.text.push_str(text);
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use crate::agent::{StopReason, Usage};
use crate::message::{Message, ToolCall};
use async_trait::async_trait;
use std::io::Write;

/// Error returned by a listener. It stops the run with
/// [`AgentError::Listener`](crate::AgentError::Listener).
pub type ListenerError = Box<dyn std::error::Error + Send + Sync>;

/// An event of a streamed run, as a listener sees it.
///
/// It borrows from the stream's [`AgentEvent`](crate::AgentEvent) and does not depend on the type
/// of the answer, so one listener works for runs with any answer type. The parsed answer itself is
/// returned by [`AgentStream::listen`](crate::AgentStream::listen).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RunEvent<'a> {
    /// A piece of the model's text. See [`AgentEvent::TextDelta`](crate::AgentEvent::TextDelta).
    TextDelta(&'a str),
    /// A piece of the model's reasoning, if the provider streams it.
    ReasoningDelta(&'a str),
    /// The model asked for a tool call. Sent before the tool runs.
    ToolCall(&'a ToolCall),
    /// A tool call finished.
    ToolResult {
        /// Id of the tool call.
        call_id: &'a str,
        /// Name of the tool.
        tool: &'a str,
        /// Result sent to the model: the tool's output, or the error message.
        content: &'a str,
        /// `true` when the tool failed.
        is_error: bool,
    },
    /// The run finished. Always the last event of a successful run.
    Finished {
        /// The messages added in this run.
        new_messages: &'a [Message],
        /// Tokens used by all model calls of the run together.
        usage: &'a Usage,
        /// Why the model stopped writing the final answer.
        stop_reason: Option<&'a StopReason>,
        /// Number of model calls made in the run.
        iterations: usize,
    },
}

/// Receives the events of a streamed run. See the [module documentation](self).
pub trait EventListener: Send {
    /// Called for every event, in order. An error stops the run.
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError>;
}

/// Same as [`EventListener`], for listeners that need to `.await`, for example to write to an
/// async writer.
#[async_trait]
pub trait AsyncEventListener: Send {
    /// Called for every event, in order. An error stops the run.
    async fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError>;
}

impl<L: EventListener + ?Sized> EventListener for &mut L {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        (**self).on_event(event)
    }
}

impl<L: EventListener + ?Sized> EventListener for Box<L> {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        (**self).on_event(event)
    }
}

/// Listener that writes a run to a writer: the model's text as it arrives, and each tool call on
/// its own line, for example `[tool call] get_weather({"city":"Paris"})`.
///
/// It is an [`EventListener`] for a [`std::io::Write`] (stdout, stderr, a file, a `Vec<u8>`) and an
/// [`AsyncEventListener`] for a [`futures::io::AsyncWrite`](futures_util::io::AsyncWrite) (the
/// runtime-neutral async I/O trait). For a tokio writer, use [`Printer::tokio`] (feature
/// `tokio`). The writer is flushed after every piece, so the text shows up right away.
///
/// As an [`EventListener`], the printer writes with blocking calls inside the async run. That is
/// fine for stdout, stderr and files; for a writer that can block for long, such as a socket or a
/// pipe, use an async writer instead.
///
/// By default it writes the text and the tool calls; the `with_*` methods choose what else is
/// written.
///
/// ```
/// use agentai::listener::Printer;
///
/// let printer = Printer::new(std::io::stdout())
///     .with_tool_results(true)
///     .with_reasoning(true);
/// ```
#[derive(Debug)]
pub struct Printer<W> {
    out: W,
    tool_calls: bool,
    tool_results: bool,
    reasoning: bool,
    /// A line was started and not ended yet.
    line_open: bool,
    /// The last thing written was reasoning (it may span several lines).
    in_reasoning: bool,
}

impl<W> Printer<W> {
    /// Creates a printer that writes to `out`.
    pub fn new(out: W) -> Self {
        Self {
            out,
            tool_calls: true,
            tool_results: false,
            reasoning: false,
            line_open: false,
            in_reasoning: false,
        }
    }

    /// Sets whether each tool call is written with its arguments. On by default.
    pub fn with_tool_calls(mut self, tool_calls: bool) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Sets whether each tool result is written. Off by default.
    pub fn with_tool_results(mut self, tool_results: bool) -> Self {
        self.tool_results = tool_results;
        self
    }

    /// Sets whether the model's reasoning is written, if the provider streams it. Off by default.
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Returns the writer.
    pub fn into_inner(self) -> W {
        self.out
    }

    /// Returns the text to write for `event`.
    fn render(&mut self, event: &RunEvent<'_>) -> String {
        let mut out = String::new();
        match *event {
            RunEvent::TextDelta(text) => {
                if self.in_reasoning {
                    self.end_block(&mut out);
                }
                self.push_text(&mut out, text);
            }
            RunEvent::ReasoningDelta(text) if self.reasoning => {
                if !self.in_reasoning {
                    self.end_block(&mut out);
                    out.push_str("[reasoning] ");
                    self.line_open = true;
                    self.in_reasoning = true;
                }
                self.push_text(&mut out, text);
            }
            RunEvent::ToolCall(call) if self.tool_calls => {
                self.end_block(&mut out);
                out.push_str(&format!("[tool call] {}({})\n", call.name, call.arguments));
            }
            RunEvent::ToolResult {
                tool,
                content,
                is_error,
                ..
            } if self.tool_results => {
                self.end_block(&mut out);
                let label = if is_error {
                    "tool error"
                } else {
                    "tool result"
                };
                out.push_str(&format!("[{label}] {tool}: {content}\n"));
            }
            RunEvent::Finished { .. } => self.end_block(&mut out),
            _ => {}
        }
        out
    }

    fn push_text(&mut self, out: &mut String, text: &str) {
        if text.is_empty() {
            return;
        }
        out.push_str(text);
        self.line_open = !text.ends_with('\n');
    }

    /// Ends the current line, if one is open, and the current reasoning block.
    fn end_block(&mut self, out: &mut String) {
        if self.line_open {
            out.push('\n');
            self.line_open = false;
        }
        self.in_reasoning = false;
    }
}

impl<W: Write + Send> EventListener for Printer<W> {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        let text = self.render(event);
        if !text.is_empty() {
            self.out.write_all(text.as_bytes())?;
            self.out.flush()?;
        }
        Ok(())
    }
}

#[async_trait]
impl<W: futures_util::io::AsyncWrite + Unpin + Send> AsyncEventListener for Printer<W> {
    async fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        use futures_util::io::AsyncWriteExt;

        let text = self.render(event);
        if !text.is_empty() {
            self.out.write_all(text.as_bytes()).await?;
            self.out.flush().await?;
        }
        Ok(())
    }
}

#[cfg(feature = "tokio")]
mod tokio_writer {
    use super::Printer;
    use std::io;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Adapter that lets a [`Printer`] write to a [`tokio::io::AsyncWrite`]. Created by
    /// [`Printer::tokio`].
    #[derive(Debug)]
    pub struct TokioWriter<W>(W);

    impl<W> TokioWriter<W> {
        /// Returns the tokio writer.
        pub fn into_inner(self) -> W {
            self.0
        }
    }

    impl<W: tokio::io::AsyncWrite + Unpin> futures_util::io::AsyncWrite for TokioWriter<W> {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            Pin::new(&mut self.0).poll_write(cx, buf)
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.0).poll_flush(cx)
        }

        fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.0).poll_shutdown(cx)
        }
    }

    impl<W> Printer<TokioWriter<W>> {
        /// Creates a printer that writes to a tokio writer: `tokio::io::stdout()`,
        /// `tokio::io::stderr()`, `tokio::fs::File`, a TCP stream, ...
        pub fn tokio(out: W) -> Self {
            Printer::new(TokioWriter(out))
        }
    }
}

#[cfg(feature = "tokio")]
pub use tokio_writer::TokioWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Usage;

    fn print(printer: Printer<Vec<u8>>, events: &[RunEvent<'_>]) -> String {
        let mut printer = printer;
        for event in events {
            EventListener::on_event(&mut printer, event).unwrap();
        }
        String::from_utf8(printer.into_inner()).unwrap()
    }

    fn finished(usage: &Usage) -> RunEvent<'_> {
        RunEvent::Finished {
            new_messages: &[],
            usage,
            stop_reason: None,
            iterations: 1,
        }
    }

    #[test]
    fn reasoning_over_several_lines_has_one_prefix() {
        let usage = Usage::default();
        let output = print(
            Printer::new(Vec::new()).with_reasoning(true),
            &[
                RunEvent::ReasoningDelta("a\n"),
                RunEvent::ReasoningDelta("b"),
                RunEvent::TextDelta("answer"),
                finished(&usage),
            ],
        );
        assert_eq!(output, "[reasoning] a\nb\nanswer\n");
    }

    #[test]
    fn reasoning_after_a_complete_line_starts_on_a_new_line() {
        let usage = Usage::default();
        let output = print(
            Printer::new(Vec::new()).with_reasoning(true),
            &[
                RunEvent::TextDelta("intro\n"),
                RunEvent::ReasoningDelta("think"),
                RunEvent::TextDelta("done\n"),
                finished(&usage),
            ],
        );
        assert_eq!(output, "intro\n[reasoning] think\ndone\n");
    }

    #[test]
    fn hidden_reasoning_writes_nothing() {
        let usage = Usage::default();
        let output = print(
            Printer::new(Vec::new()),
            &[
                RunEvent::ReasoningDelta("think"),
                RunEvent::TextDelta("answer"),
                finished(&usage),
            ],
        );
        assert_eq!(output, "answer\n");
    }
}
