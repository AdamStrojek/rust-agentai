use agentai::Agent;
use genai::ClientBuilder;
use anyhow::Result;
use genai::chat::ChatOptions;
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

    let chat_options = ChatOptions::default().with_temperature(0.0).with_max_tokens(20);
    let client = ClientBuilder::default().with_chat_options(chat_options).build();

    info!("Question: {}", question);

    let mut agent = Agent::new_with_client(client, SYSTEM, &());

    let answer: String = agent.run(&model, question).await?;

    info!("Answer: {}", answer);

    Ok(())
}
