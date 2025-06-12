use agentai::Agent;
use anyhow::Result;
use log::{info, LevelFilter};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::sync::Arc;
use agentai::tool::mcp::McpToolBox;

const SYSTEM: &str =
    "You are helpful assistant.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let model = std::env::var("AGENTAI_MODEL").unwrap_or("gpt-4o-mini".to_string());

    let question = "What is current time in Poland??";

    info!("Question: {}", question);

    let mut agent = Agent::new(SYSTEM);

    let mcp_tools = McpToolBox::new("uvx", ["mcp-server-time", "--local-timezone", "UTC"], None).await?;

    let answer: Answer = agent.run(&model, question, Some(Arc::new(mcp_tools))).await?;

    info!("{:#?}", answer);

    Ok(())
}

#[allow(dead_code)]
#[derive(Deserialize, JsonSchema, Debug)]
struct Answer {
    // It is always good idea to include thinking field for LLM's debugging
    /// In this field provide your thinking steps
    #[serde(rename = "_thinking")]
    thinking: String,

    /// In this field provide answer
    answer: String,
}
