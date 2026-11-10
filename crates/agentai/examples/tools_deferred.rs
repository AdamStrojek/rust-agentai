//! Deferred Tool Execution Example
//!
//! This example shows how to write a background tool with [`ToolReturn::deferred`] and run it with
//! [`TokioDeferredTaskToolBox`].
//!
//! Run with your OpenRouter API key:
//! ```bash
//! `OPEN_ROUTER_API_KEY="your-key" cargo run --example tools_deferred
//! ```

use agentai::tool::{
    TokioDeferredTaskToolBox, Tool, ToolBox, ToolBoxSet, ToolError, ToolOutput, ToolResult,
    ToolReturn, toolbox,
};
use agentai::{Agent, AgentEvent, Message, ToolCall};
use anyhow::Result;
use futures_util::StreamExt;
use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::io::Write;
use std::time::Duration;

const SYSTEM: &str = "\
You are a helpful assistant with access to a background tool (`slow_task`) and deferred task \
management tools (`list_deferred_tasks`, `wait_deferred_task`, `get_deferred_task`). \
When `slow_task` starts a job in the background, use `wait_deferred_task` (or `get_deferred_task`) \
with the returned task ID to collect its result.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Debug,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;

    // Uses OpenRouter by default (reads `OPEN_ROUTER_API_KEY` from the environment).
    // Override the model with `AGENTAI_MODEL` if desired.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or_else(|_| "open_router::nvidia/nemotron-3.5-super-vl:free".to_string());

    // `TokioDeferredTaskToolBox` spawns deferred tasks onto Tokio and exposes
    // `list_deferred_tasks`, `get_deferred_task`, and `wait_deferred_task`.
    // Set a 3-second default timeout so tasks longer than 3s time out non-destructively first.
    let deferred = TokioDeferredTaskToolBox::new().with_default_timeout(Duration::from_secs(3));

    let mut tools = ToolBoxSet::new();
    tools.add_tool(SlowToolBox);
    tools.add_tool(deferred.clone());

    let agent = Agent::new(&model, SYSTEM)?
        .with_toolbox(tools)
        .with_max_iterations(12);

    let mut history: Vec<Message> = Vec::new();
    let mut handles = Vec::new();

    let prompt = "\
Call `slow_task` 3 times with sleep durations: 8 second, 4 seconds, and 10 seconds. \
Then wait for each task to finish using `wait_deferred_task` with a 1-second timeout \
(`timeout_seconds: 1`) until all 3 results are collected, and report what happened on each wait.";

    println!("Prompt: {prompt}\n");

    let mut stream = agent
        .run_stream::<String>(prompt)
        .with_history(&history)
        .with_deferred_tasks(handles)
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::TextDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            AgentEvent::ToolCall(call) => {
                println!("\n[tool call] {} ({}): {}", call.name, call.id, call.arguments);
            }
            AgentEvent::ToolResult {
                call_id,
                tool,
                content,
                ..
            } => {
                println!("[tool result] {tool} ({call_id}): {content}");
            }
            AgentEvent::DeferredTask(task) => {
                println!("[spawning deferred task] {} ({})", task.id, task.tool);
                deferred.spawn(task);
            }
            AgentEvent::Finished(result) => {
                println!(
                    "\n\nDone after {} model calls ({} tokens, {} tasks still running)",
                    result.iterations,
                    result.usage.total_tokens,
                    result.deferred_tasks.len(),
                );
                history.extend(result.new_messages);
                handles = result.deferred_tasks;
            }
            _ => {}
        }
    }

    let _ = handles;
    Ok(())
}

struct SlowToolBox;

#[toolbox]
impl SlowToolBox {
    /// Starts a slow background task that sleeps for `sleep_seconds` before producing its result.
    #[tool]
    fn slow_task(
        &self,
        call: &ToolCall,
        /// How many seconds the background task should sleep before finishing.
        sleep_seconds: u64,
    ) -> ToolResult<ToolReturn> {
        let task_id = call.id.clone();
        Ok(ToolReturn::deferred(
            format!(
                "Started slow_task in the background (task_id: '{task_id}', sleep: {sleep_seconds}s)."
            ),
            async move {
                tokio::time::sleep(Duration::from_secs(sleep_seconds)).await;
                Ok(ToolOutput::from_text(format!(
                    "slow_task '{task_id}' completed after sleeping for {sleep_seconds}s!"
                )))
            },
        ))
    }
}
