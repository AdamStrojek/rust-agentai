use agentai::tool::{
    web::{WebFetchToolBox, WebSearchToolBox},
    ToolBoxSet,
};
use agentai::Agent;
use anyhow::Result;
use log::{info, LevelFilter};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::env;

const SYSTEM: &str = "You are helpful assistant. You goal is to provide user answer based on search
    results and content of found pages. You can use provided tools to achieve that.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    dotenvy::dotenv()?;
    info!("Starting AgentAI");

    let api_key = env::var("BRAVE_API_KEY")?;
    let web_search_tool = WebSearchToolBox::new(&api_key);
    let web_fetch_tool = WebFetchToolBox::new();

    let mut toolbox = ToolBoxSet::new();
    toolbox.add_tool(web_search_tool);
    toolbox.add_tool(web_fetch_tool);

    let question = "Search me for tools that can be used with terminal and Rust";

    info!("Question: {}", question);

    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("openai/gpt-4.1-mini".to_string());

    let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM);

    let answer: Answer = agent.run(&model, question, Some(&toolbox)).await?;

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
