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

    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    // The model has to support images (vision)
    let model =
        std::env::var("AGENTAI_MODEL").unwrap_or("nvidia/nemotron-3.5-super-vl:free".to_string());

    let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?.with_model(&model);

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
