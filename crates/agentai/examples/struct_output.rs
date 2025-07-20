use agentai::Agent;
use anyhow::Result;
use log::{info, LevelFilter};
use schemars::JsonSchema;
use serde::Deserialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let question = "Why sky is blue?";

    info!("Question: {}", question);

    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("openai/gpt-4.1-mini".to_string());

    let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM);

    let answer: Answer = agent.run(&model, question, None).await?;

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
