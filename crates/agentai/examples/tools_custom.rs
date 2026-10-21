//! Custom Agent Tool Implementation Example
//!
//! This example demonstrates how to create a custom tool using the `#[toolbox]` and `#[tool()]` macros
//! provided by the `agentai` crate. This tool will be used by the AI agent to fetch content from a URL.
//!

use agentai::tool::{Tool, ToolBox, ToolError, ToolResult, toolbox};
use agentai::{Agent, AgentResult};
use anyhow::Error;
use log::{LevelFilter, info};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant. You goal is to provide summary for provided site. Limit you answer to 3 sentences.";

#[tokio::main]
async fn main() -> Result<(), Error> {
    TermLogger::init(
        LevelFilter::Trace,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    let question = "For what I can use this library? https://raw.githubusercontent.com/AdamStrojek/rust-agentai/refs/heads/master/README.md";

    info!("Question: {}", question);

    let toolbox = UrlFetcherToolBox {};

    dbg!(toolbox.tools_definitions()?);

    let base_url = std::env::var("AGENTAI_BASE_URL")?;
    let api_key = std::env::var("AGENTAI_API_KEY")?;
    let model =
        std::env::var("AGENTAI_MODEL").unwrap_or("nvidia/nemotron-3.5-super-vl:free".to_string());

    let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?
        .with_model(&model)
        .with_toolbox(toolbox);

    let result: AgentResult<String> = agent.run(question).await?;

    info!("Answer: {}", result.answer);

    Ok(())
}

// This structure represents our custom tool set. The `#[toolbox]` macro
// is applied to the `impl` block for this struct. It discovers methods
// annotated with `#[tool()]` and automatically generates the necessary
// `ToolBox` trait implementation, including `name`, `description`,
// `schema`, and `call` methods based on the annotated functions.
//
// For this example, `UrlFetcherToolBox` itself doesn't need to store
// any state, but it could if your tools required it.
struct UrlFetcherToolBox {}

// The `#[toolbox]` macro is applied to the `impl` block for `UrlFetcherToolBox`.
// It processes the methods within this block to create the tool definitions.
#[toolbox]
impl UrlFetcherToolBox {
    // The `#[tool]` macro annotates methods that should be exposed as tools
    // to the AI agent. The macro automatically generates the necessary metadata
    // (name, description, schema) for the tool based on the function signature
    // and documentation comments.

    // The tool name will be derived from the function name (`web_fetch`).
    // The description will be taken from this documentation comment.
    // The schema will be generated from the function arguments (here, `url: String`).
    // The body of this function will be executed when the AI agent decides to use the tool.
    #[tool]
    /// This tool allow to fetch resource from provided URL
    async fn web_fetch(
        &self,
        /// Use this field to provide URL of file to download
        url: String,
    ) -> ToolResult {
        // Use reqwest to fetch the content from the provided URL.
        // Errors the model can react to, such as a bad URL, are `ToolError::Failed` or
        // `ToolError::InvalidArguments`: the agent sends them to the model, which can then try
        // again. Use `ToolError::Configuration` or `ToolError::fatal(..)` for problems that
        // retrying cannot fix: the run then stops and returns `AgentError::Tool`.
        reqwest::get(&url)
            .await
            .map_err(|e| ToolError::Failed(format!("Request to {url} failed: {e}")))?
            .text()
            .await
            .map_err(|e| ToolError::Failed(format!("Reading {url} failed: {e}")))
    }
}
