use agentai::websearch::WebSearchTool;
use agentai::Agent;
use anyhow::Result;
use genai::Client;
use log::{info, LevelFilter};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::env;
use std::sync::Arc;

const MODEL: &str = "gpt-4o-mini";

const SYSTEM: &str =
    "You are helpful assistant. You goal is to search for information requested by user,\
in result you will receive 5 sites, provide summary based on titles and descriptions.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let api_key = env::var("BRAVE_API_KEY")?;
    let web_search_tool = WebSearchTool::new(&api_key);

    // Creating GenAI client
    let client = Client::default();

    let question = "Search me for tools that can be used with terminal and Rust";

    info!("Question: {}", question);

    let mut agent = Agent::new(&client, SYSTEM, &());
    agent.add_tool(Arc::new(web_search_tool));

    let answer: Answer = agent.run(MODEL, question).await?;

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
