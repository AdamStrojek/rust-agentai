use agentai::Agent;
use anyhow::Result;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    // Remember about providing what model you want to use
    // Some models may require providing authorization details
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("gemini-2.0-flash".to_string());

    let question = "Why sky is blue?";

    info!("Question: {}", question);

    let mut agent = Agent::new(SYSTEM);

    let answer: String = agent.run(&model, question, None).await?;

    info!("Answer: {}", answer);

    Ok(())
}
