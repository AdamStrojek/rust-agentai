//! OpenRouter Example
//!
//! This example uses genai's native OpenRouter adapter through `Agent::new_with_provider`.
//! No base URL is needed: the adapter knows OpenRouter's endpoint, and every model name
//! (for example `nvidia/nemotron-3.5-super-vl:free`) is sent to OpenRouter as is.

use agentai::tool::buildin::CurrentDateAndTimeToolBox;
use agentai::{AdapterKind, Agent};
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

    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let model =
        std::env::var("AGENTAI_MODEL").unwrap_or("nvidia/nemotron-3.5-super-vl:free".to_string());

    // `None` keeps the default OpenRouter endpoint
    let mut agent = Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, SYSTEM)?;
    let tools = CurrentDateAndTimeToolBox {};

    let answer: String = agent.run(&model, question, Some(&tools)).await?;

    info!("Answer: {}", answer);

    Ok(())
}
