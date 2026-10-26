use agentai::{Agent, AgentResult, Prompt};
use anyhow::Result;
use log::{LevelFilter, info};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant";

// The project logo, published on GitHub and read from the repository files
const LOGO_URL: &str =
    "https://raw.githubusercontent.com/AdamStrojek/rust-agentai/master/assets/logo-docs.png";
const LOGO_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/logo-docs.png"
));

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    // The model has to support images (vision)
    // The provider comes from the model name, for example `claude-haiku-5-5` or
    // `omlx::Qwen3.8-27B-4bit`; its API key and endpoint come from the provider's environment
    // variables, for example `ANTHROPIC_API_KEY` or `OMLX_ENDPOINT`.
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("claude-haiku-5-5".to_string());

    let agent = Agent::new(SYSTEM)?.with_model(&model);

    // An image by URL: the provider downloads it
    let prompt = Prompt::text("Describe this logo in one sentence.").with_image_url(LOGO_URL);
    let result: AgentResult<String> = agent.run(prompt).await?;
    info!("Logo from URL: {}", result.answer);

    // The same image from its bytes: it is sent base64 encoded inside the request
    let prompt =
        Prompt::text("What colours does this logo use?").with_image_bytes("image/png", LOGO_PNG);
    let result: AgentResult<String> = agent.run(prompt).await?;
    info!("Logo from bytes: {}", result.answer);

    Ok(())
}
