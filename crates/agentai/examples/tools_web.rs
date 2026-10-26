use agentai::tool::{
    ToolBoxSet,
    web::{WebFetchToolBox, WebSearchToolBox},
};
use agentai::{Agent, AgentResult};
use anyhow::Result;
use log::{LevelFilter, info};
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
    info!("Starting AgentAI");

    let api_key = env::var("BRAVE_API_KEY")?;
    let web_search_tool = WebSearchToolBox::new(&api_key);
    let web_fetch_tool = WebFetchToolBox::new();

    let mut toolbox = ToolBoxSet::new();
    toolbox.add_tool(web_search_tool);
    toolbox.add_tool(web_fetch_tool);

    let question = "Search me for tools that can be used with terminal and Rust";

    info!("Question: {}", question);

    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(&model, SYSTEM)?.with_toolbox(toolbox);

    let result: AgentResult<Answer> = agent.run(question).await?;

    info!("{:#?}", result.answer);

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
