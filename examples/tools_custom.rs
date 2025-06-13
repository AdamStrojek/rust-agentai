//! Custom Agent Tool Implementation Example
//!
//! This example demonstrates how to create a custom tool using the `#[toolbox]` and `#[tool()]` macros
//! provided by the `agentai` crate. This tool will be used by the AI agent to fetch content from a URL.
//!

use agentai::Agent;
use agentai::tool::{ToolBox, Tool, ToolError, toolbox};
use anyhow::Error;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::sync::Arc;

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

    let model = std::env::var("AGENTAI_MODEL").unwrap_or("gpt-4o-mini".to_string());

    let question =
        "For what I can use this library? https://raw.githubusercontent.com/AdamStrojek/rust-agentai/refs/heads/master/README.md";

    info!("Question: {}", question);

    let mut agent = Agent::new(SYSTEM);

    let answer: String = agent.run(&model, question, Some(Arc::new(UrlFetcherToolBox {}))).await?;
    // let answer: String = agent.run(&model, question, None).await?;

    info!("Answer: {}", answer);

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
    // The `#[tool()]` macro annotates methods that should be exposed as tools
    // to the AI agent. The macro automatically generates the necessary metadata
    // (name, description, schema) for the tool based on the function signature
    // and documentation comments.

    // The tool name will be derived from the function name (`web_fetch`).
    // The description will be taken from this documentation comment.
    // The schema will be generated from the function arguments (here, `url: String`).
    // The body of this function will be executed when the AI agent decides to use the tool.
    #[tool()]
    async fn web_fetch(
        &self,
        /// Use this field to provide URL of file to download
        url: String
    ) -> Result<String, ToolError> {
        // Use reqwest to fetch the content from the provided URL.
        // The `?` operator handles potential errors from the get and text methods.
        Ok(
            reqwest::get(url).await.map_err(|e| anyhow::Error::new(e))?
                .text().await.map_err(|e| anyhow::Error::new(e))?
        )
    }
}
