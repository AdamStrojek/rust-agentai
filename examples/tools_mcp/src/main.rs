use agentai::Agent;
use anyhow::Result;
use genai::Client;
use log::{info, LevelFilter};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::env;
use agentai::mcp::McpClient;

const MODEL: &str = "gpt-4o-mini";

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

    let model = env::var("AGENTAI_MODEL").unwrap_or(MODEL.to_owned());

    // Creating GenAI client
    let client = Client::default();

    let question = "What is current time in Poland??";

    info!("Question: {}", question);

    let mut agent = Agent::new(&client, SYSTEM, &());

    let mcp_tools = McpClient::new("uvx", ["mcp-server-time"]).await?;
    for agent_tool in mcp_tools.tools().await? {
        agent.add_tool(agent_tool);
    }

    let answer: Answer = agent.run(&model, question).await?;

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
