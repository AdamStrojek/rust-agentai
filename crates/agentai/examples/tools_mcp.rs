use agentai::tool::mcp::McpToolBox;
use agentai::{Agent, AgentResult};
use anyhow::Result;
use log::{LevelFilter, info};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant.";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let question = "What is current time in Poland??";

    info!("Question: {}", question);

    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(SYSTEM)?.with_model(&model);

    let mcp_tools =
        McpToolBox::new("uvx", ["mcp-server-time", "--local-timezone", "UTC"], None).await?;

    // The toolbox can also be set for one run only
    let result: AgentResult<Answer> = agent.run(question).with_toolbox(mcp_tools).await?;

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
