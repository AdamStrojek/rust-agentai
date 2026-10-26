//! Event Listener Example
//!
//! This example shows two ways of following a streamed run with event listeners:
//!
//! 1. a configured `Printer`, the ready-made listener, which here writes the answer, the tool
//!    calls and the tool results to stderr;
//! 2. a custom `EventListener` that keeps its own state, the way a GUI would: it collects the
//!    answer, counts the tool calls and reports the token usage at the end.

use agentai::listener::{EventListener, ListenerError, Printer, RunEvent};
use agentai::tool::buildin::CurrentDateAndTimeToolBox;
use agentai::{Agent, AgentResult};
use anyhow::Result;
use std::time::Instant;

const SYSTEM: &str = "You are helpful assistant. Use tools when you need the current date or time.";

/// A listener with its own state. A GUI would update its widgets in `on_event` instead.
struct RunMonitor {
    started: Instant,
    answer: String,
    tool_calls: Vec<String>,
}

impl RunMonitor {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            answer: String::new(),
            tool_calls: Vec::new(),
        }
    }
}

impl EventListener for RunMonitor {
    fn on_event(&mut self, event: &RunEvent<'_>) -> Result<(), ListenerError> {
        match event {
            RunEvent::TextDelta(text) => self.answer.push_str(text),
            RunEvent::ToolCall(call) => {
                // Text written before a tool call is not part of the final answer
                self.answer.clear();
                self.tool_calls.push(call.name.clone());
                println!(
                    "[{:>5} ms] calling {}",
                    self.started.elapsed().as_millis(),
                    call.name
                );
            }
            RunEvent::ToolResult { tool, is_error, .. } => {
                let status = if *is_error { "failed" } else { "done" };
                println!(
                    "[{:>5} ms] {tool} {status}",
                    self.started.elapsed().as_millis()
                );
            }
            RunEvent::Finished {
                usage, iterations, ..
            } => {
                println!(
                    "[{:>5} ms] finished after {iterations} model calls and {} tool calls, {} tokens",
                    self.started.elapsed().as_millis(),
                    self.tool_calls.len(),
                    usage.total_tokens
                );
            }
            // `RunEvent` is #[non_exhaustive]; reasoning and future events are ignored here
            _ => {}
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(SYSTEM)?
        .with_model(&model)
        .with_toolbox(CurrentDateAndTimeToolBox {});

    // 1. The ready-made Printer, configured to also show the tool results, writing to stderr
    let printer = Printer::new(std::io::stderr()).with_tool_results(true);
    let result: AgentResult<String> = agent
        .run_stream("What day of the week is it today?")
        .await?
        .listen(printer)
        .await?;
    println!("Answer: {}\n", result.answer);

    // 2. A custom listener. Pass it by `&mut` to read its state after the run.
    let mut monitor = RunMonitor::new();
    let result: AgentResult<String> = agent
        .run_stream("What time is it in Tokyo and in New York?")
        .await?
        .listen(&mut monitor)
        .await?;
    println!("\nAnswer collected by the listener: {}", monitor.answer);
    println!("Tools called: {}", monitor.tool_calls.join(", "));
    println!("Answer returned by the run: {}", result.answer);

    Ok(())
}
