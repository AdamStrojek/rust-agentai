//! OpenRouter Example
//!
//! This example uses OpenRouter through the `open_router::` namespace of the model name.
//! The API key is read from the `OPEN_ROUTER_API_KEY` environment variable.

use agentai::tool::buildin::CurrentDateAndTimeToolBox;
use agentai::{Agent, AgentResult};
use anyhow::Result;
use log::{LevelFilter, info};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant. Use tools when you need the current date or time.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let question = "What day of the week is it today?";

    info!("Question: {}", question);

    // The `open_router::` prefix sends the model to OpenRouter
    let agent = Agent::new("open_router::nvidia/nemotron-3.5-super-vl:free", SYSTEM)?
        .with_toolbox(CurrentDateAndTimeToolBox {});

    let result: AgentResult<String> = agent.run(question).await?;

    info!("Answer: {}", result.answer);

    Ok(())
}
