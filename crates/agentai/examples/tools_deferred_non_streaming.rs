//! Non-Streaming Deferred Tool Execution Example
//!
//! This example shows how deferred background tasks work with non-streaming [`Agent::run`]:
//! 1. During **Run 1**, the model starts background tasks and replies to the user right away.
//!    Because `agent.run()` does not yield stream events mid-run, the unstarted tasks are returned
//!    in [`AgentResult::unstarted_deferred_tasks`](agentai::AgentResult::unstarted_deferred_tasks)
//!    and spawned after `run()` finishes.
//! 2. Once the tasks complete in the background, **Run 2** starts with [`Prompt::new()`](agentai::Prompt::new)
//!    (no user prompt) and `.with_deferred_tasks(handles)`. The agent loop appends the completion
//!    notifications and the model calls `get_deferred_task` to fetch and summarize the results.
//!
//! Run with your OpenRouter API key:
//! ```bash
//! OPEN_ROUTER_API_KEY="your-key" cargo run --example tools_deferred_non_streaming
//! ```

use agentai::tool::{
    TokioDeferredTaskToolBox, Tool, ToolBox, ToolBoxSet, ToolError, ToolOutput, ToolResult,
    ToolReturn, toolbox,
};
use agentai::{Agent, Message, Prompt, ToolCall};
use anyhow::Result;
use log::LevelFilter;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::time::Duration;

const SYSTEM: &str = "\
You are a helpful assistant with access to a background tool (`slow_task`) and deferred task \
management tools (`list_deferred_tasks`, `wait_deferred_task`, `get_deferred_task`). \
When asked to start background tasks without waiting, call `slow_task` and immediately reply to \
the user confirming that the tasks have been started (do not call `wait_deferred_task` or \
`get_deferred_task` until you are notified that a task has finished).";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Debug,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;

    let model = std::env::var("AGENTAI_MODEL")
        .unwrap_or_else(|_| "open_router::nvidia/nemotron-3.5-super-vl:free".to_string());

    let deferred = TokioDeferredTaskToolBox::new();

    let mut tools = ToolBoxSet::new();
    tools.add_tool(SlowToolBox);
    tools.add_tool(deferred.clone());

    let agent = Agent::new(&model, SYSTEM)?
        .with_toolbox(tools)
        .with_max_iterations(8);

    let mut history: Vec<Message> = Vec::new();

    // --- Run 1: start background tasks using non-streaming `agent.run()` ---
    let prompt = "Start two `slow_task` jobs (2 seconds and 4 seconds) and reply right away \
                  without waiting for them to finish.";
    println!("=== Run 1 Prompt ===\n{prompt}\n");

    let run1 = agent.run::<String>(prompt).await?;
    println!("=== Run 1 Answer ===\n{}\n", run1.answer);
    println!(
        "Run 1 finished: {} unstarted task(s), {} active handle(s)",
        run1.unstarted_deferred_tasks.len(),
        run1.deferred_tasks.len(),
    );

    // Spawn the unstarted tasks collected during `agent.run()`
    for task in run1.unstarted_deferred_tasks {
        println!("[spawning unstarted task] {} ({})", task.id, task.tool);
        deferred.spawn(task);
    }

    history.extend(run1.new_messages);
    let handles = run1.deferred_tasks;

    // Wait for the background tasks to finish between runs
    println!("\nWaiting for background tasks to finish...");
    while handles.iter().any(|h| !h.is_finished()) {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    println!("All background tasks have finished!\n");

    // --- Run 2: wake the agent with an empty prompt and the finished handles ---
    println!("=== Run 2 (Notification-only, no user prompt) ===\n");
    let run2 = agent
        .run::<String>(Prompt::new())
        .with_history(&history)
        .with_deferred_tasks(handles)
        .await?;

    println!("=== Run 2 Answer ===\n{}", run2.answer);

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
                "Queued slow_task in the background (task_id: '{task_id}', sleep: {sleep_seconds}s)."
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
