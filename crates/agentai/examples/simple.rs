use agentai::{Agent, AgentResult, Message};
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

    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(&model, SYSTEM)?;

    // The agent does not remember the conversation: you keep the history and pass it to
    // every run.
    let mut history: Vec<Message> = Vec::new();

    let question = "Why sky is blue?";
    info!("Question: {}", question);
    let result: AgentResult<String> = agent.run(question).with_history(&history).await?;
    info!("Answer: {}", result.answer);
    // Add the question and the answer to the history to continue the conversation
    history.extend(result.new_messages);

    let question = "Summarize your previous answer in one sentence.";
    info!("Question: {}", question);
    let result: AgentResult<String> = agent.run(question).with_history(&history).await?;
    info!("Answer: {}", result.answer);
    history.extend(result.new_messages);

    info!(
        "The history has {} messages, the last run used {:?} tokens",
        history.len(),
        result.usage.total_tokens
    );

    Ok(())
}
