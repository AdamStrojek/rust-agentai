use agentai::{Agent, ChatMessage, RunResult};
use anyhow::Result;
use log::{LevelFilter, info};
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

    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    let model =
        std::env::var("AGENTAI_MODEL").unwrap_or("nvidia/nemotron-3.5-super-vl:free".to_string());

    let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?.with_model(&model);

    // The agent does not remember the conversation: you keep the history and pass it to
    // every run.
    let mut history: Vec<ChatMessage> = Vec::new();

    let question = "Why sky is blue?";
    info!("Question: {}", question);
    let result: RunResult<String> = agent.run(question).history(&history).await?;
    info!("Answer: {}", result.answer);
    // Add the question and the answer to the history to continue the conversation
    history.extend(result.new_messages);

    let question = "Summarize your previous answer in one sentence.";
    info!("Question: {}", question);
    let result: RunResult<String> = agent.run(question).history(&history).await?;
    info!("Answer: {}", result.answer);
    history.extend(result.new_messages);

    info!(
        "The history has {} messages, the last run used {:?} tokens",
        history.len(),
        result.usage.total_tokens
    );

    Ok(())
}
