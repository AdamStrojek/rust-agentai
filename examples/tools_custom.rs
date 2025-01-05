//! Custom Agent Tool Implementation Example
//!
//! This example is pretty similar to [Tools Search](crate::examples::tools_search), but here we focus
//! on demonstrating how to create your own tool that can be used later in your AI Agent
//!

use agentai::{Agent, AgentTool};
use anyhow::{Context, Result};
use async_trait::async_trait;
use genai::Client;
use log::{info, LevelFilter};
use serde_json::{json, Value};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::sync::Arc;

const MODEL: &str = "gpt-4o-mini";

const SYSTEM: &str = "You are helpful assistant. You goal is to provide summary for provided site. Limit you answer to 3 sentences.";

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

    let question =
        "For what I can use this library? https://raw.githubusercontent.com/AdamStrojek/rust-agentai/refs/heads/master/README.md";

    info!("Question: {}", question);

    let mut agent = Agent::new(&client, SYSTEM, &());

    // Remember to register your tool!
    agent.add_tool(Arc::new(UrlFetcherTool {}));

    let answer: String = agent.run(MODEL, question).await?;

    info!("Answer: {}", answer);

    Ok(())
}

/// This structure provides "base" for trait implementation. For this example, we don't store
/// any extra information, but if you want, you can store things that differ between two instances
/// that are Context independent
///
/// Please be aware that even though this tool is provided as Arc to `Agent` you need to ensure
/// thread safe operations within the tool itself
struct UrlFetcherTool {}

/// Here is the main implementation of Agent Tool.All methods are required to be implemented.
/// This example doesn't focus on providing Context for tool, so skipping it,
/// Important notice: we use #[async_trait] to be able to use async method within trait without
/// a need for Boxing and Pinning them.
#[async_trait]
impl<CTX> AgentTool<CTX> for UrlFetcherTool {
    /// All tools require returning name of it. All next three methods are provided to LLM
    /// to be aware of the capabilities of the tool. If LLM is not triggering your tool,
    /// then check information provided here
    /// Most of LLMs require providing tool name in format: `^[a-zA-Z0-9_-]+$`
    fn name(&self) -> String {
        "Web_Fetch".to_string()
    }

    fn description(&self) -> String {
        "This tool allows you fetching any page for provided URL address".to_string()
    }

    fn schema(&self) -> Value {
        // Parameters description is provided in JSON Schema format
        json!({
            "type": "object",
            "properties": {
                "url": {
                  "description": "URL for site that should be fetched",
                  "type": "string"
                },
            },
            "required": [ "url" ]
        })
    }

    /// This is the main body of your tool. It will be executed each time LLM thinks it can be used.
    /// For this example, context parameter can be safely ignored.
    /// `params` contains parsed JSON structure that match to your description
    /// from `schema()` method
    async fn call(&self, _: &CTX, params: Value) -> Result<String> {
        let url = params["url"].as_str().context("Missing URL argument")?;
        Ok(reqwest::get(url).await?.text().await?)
    }
}
