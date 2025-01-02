use agentai::Agent;
use anyhow::Result;
use genai::Client;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

// Remember about providing what model you want to use
// For some models you can provide API keys using environment variables
const MODEL: &str = "gpt-4o-mini";

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

    // Creating GenAI client
    let client = Client::default();

    let question = "Why sky is blue?";

    info!("Question: {}", question);

    let mut agent = Agent::new(&client, SYSTEM, &());

    let answer: String = agent.run(MODEL, question).await?;

    info!("Answer: {}", answer);

    Ok(())
}
