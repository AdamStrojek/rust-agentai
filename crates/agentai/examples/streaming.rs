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
    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    let model =
        std::env::var("AGENTAI_MODEL").unwrap_or("nvidia/nemotron-3.5-super-vl:free".to_string());

    let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?
        .with_model(&model)
        .with_toolbox(CurrentDateAndTimeToolBox {});

    let mut history: Vec<Message> = Vec::new();
    let question = "What day of the week is it today? Answer with a short poem about it.";
    println!("Question: {question}\n");

    // The stream is `Send + 'static`, so it could also be moved into a spawned task
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
