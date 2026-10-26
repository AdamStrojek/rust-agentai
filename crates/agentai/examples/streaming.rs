//! Streaming Example
//!
//! This example streams the agent's answer and prints it as the model writes it. The question
//! needs the built-in date and time tools, so the stream also shows the tool calls and their
//! results before the final answer.

use agentai::tool::buildin::CurrentDateAndTimeToolBox;
use agentai::{Agent, AgentEvent, Message};
use anyhow::Result;
use futures_util::StreamExt;
use std::io::Write;

const SYSTEM: &str = "You are helpful assistant. Use tools when you need the current date or time.";

#[tokio::main]
async fn main() -> Result<()> {
    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(&model, SYSTEM)?.with_toolbox(CurrentDateAndTimeToolBox {});

    let mut history: Vec<Message> = Vec::new();
    let question = "What day of the week is it today? Answer with a short poem about it.";
    println!("Question: {question}\n");

    // The stream is `Send + 'static`, so it could also be moved into a spawned task.
    // Reading the events yourself gives full control. To just write the stream to stdout (or a
    // file, or any other `std::io::Write`), replace the loop with
    // `stream.print_to(std::io::stdout()).await?` (`print_to_async` and `print_to_tokio` for async
    // writers). To change what is written, or to show the run elsewhere (a GUI, a log), pass a
    // configured `agentai::listener::Printer` or your own `EventListener` to `stream.listen(..)`.
    let mut stream = agent
        .run_stream::<String>(question)
        .with_history(&history)
        .await?;
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::TextDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            AgentEvent::ToolCall(call) => {
                println!("[tool call] {}({})", call.name, call.arguments)
            }
            AgentEvent::ToolResult { tool, content, .. } => {
                println!("[tool result] {tool}: {content}")
            }
            AgentEvent::Finished(result) => {
                println!(
                    "\n\nDone after {} model calls, {:?} tokens",
                    result.iterations, result.usage.total_tokens
                );
                history.extend(result.new_messages);
            }
            _ => {}
        }
    }

    println!("The history has {} messages", history.len());

    Ok(())
}
